from typing import Dict, Optional, Union
from collections import OrderedDict
import sys
sys.path.append('/data/jsh/MADELEINE-main/madeleine')
import torch # type: ignore
from torch import nn # type: ignore
from models.abmil import BatchedABMIL
import pdb
from einops import rearrange # type: ignore
import numpy as np
import torch.nn.functional as F # type: ignore

# global magic numbers
HE_POSITION = 0

def create_model(
    model_cfg: Union[str, Dict],
    device: Union[str, torch.device] = 'cpu',
    checkpoint_path: Optional[str] = None,
    ):
    
   # set up MADELEINE model
    model = MADELEINE(
        config=model_cfg,
        stain_encoding=False,
    ).to(device)
    
    # restore wsi embedder for downstream slide embedding extraction.
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, weights_only=False)
        sd = list(state_dict.keys())
        contains_module = any('module' in entry for entry in sd)
        
        if not contains_module:
            model.load_state_dict(state_dict, strict=True)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        print("* Loaded weights successfully!")
            
    return model

class MADELEINE(nn.Module):
    def __init__(self, config, stain_encoding=False,):
        # stain_encoding=True
        super(MADELEINE, self).__init__()
        self.config = config
        self.modalities = config.MODALITIES
        self.stain_encoding = stain_encoding

        if self.stain_encoding: # True
            self.stain_encoding_dim = 32
            self.embedding = nn.Embedding(len(self.modalities), self.stain_encoding_dim)
        else:
            self.stain_encoding_dim = 0

        if self.config.wsi_encoder == "abmil":  # True
            # processing any WSI needs this
            pre_params = {
                'input_dim': self.config.patch_embedding_dim + self.stain_encoding_dim,
                'hidden_dim': self.config.wsi_encoder_hidden_dim
            }
            
            # params for the main MADELEINE model
            attention_params = {
                'model': 'ABMIL',
                'params': {
                    'input_dim': self.config.wsi_encoder_hidden_dim,
                    'hidden_dim': 512,
                    'dropout': True, 
                    'activation': self.config.activation,
                    'n_heads' : self.config.n_heads, 
                    'n_classes': 1
                }
            }
            
            # pre-attn projection
            self.token_projector = nn.Linear(
                attention_params['params']['hidden_dim'] * attention_params['params']['n_heads'],
                128
            )
            # patch aggregator
            self.wsi_embedders = ABMILEmbedder(pre_params, attention_params)
            
            # post-attention network
            self.projector = nn.Linear(
                attention_params['params']['hidden_dim'] * attention_params['params']['n_heads'],
                attention_params['params']['hidden_dim']
            )

        else:
            raise ValueError('Unsupported wsi_encoder. Must be "abmil". Now is {}.'.format(self.config.wsi_encoder))
        

    def encode_he(self, feats, device): # 没有运行
        feats = feats.to(device)
        bs, _, _ = feats.shape
        n_mod = 1
        feats = feats.unsqueeze(dim=1)
        HE_embedding = self.wsi_embedders(feats[:, HE_POSITION, : :], return_attention=False) 
        d_out, n_heads = HE_embedding.shape[-2], HE_embedding.shape[-1]
        HE_embedding = HE_embedding.view(bs*n_mod, d_out * n_heads)
        HE_embedding = self.projector(HE_embedding)
        HE_embedding = HE_embedding.view(bs, n_mod, d_out)
        return HE_embedding.squeeze(dim=1)

    
    def forward(self, data, device, train=True, n_views = 1, custom_stain_idx=None, return_attention=False, inference_patch_embedding=False):
        # unpack and put on device
        all_wsi_feats = data['feats'].to(device)
        slide_id = data['slide_ids']
        # store embeds
        all_embeddings = {}
        all_token_embeddings = {}
        
        # get the HE embedding (always at pos 0)
        if train:
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            all_wsi_feats = all_wsi_feats.view(bs*n_mod, n_tokens, d_in)

            # add stain specific encodings if asked for 
            if self.stain_encoding: # True
                stain_indicator = []
                for i in range(n_mod):
                    stain_indicator += [i]*bs
                stain_indicator = torch.LongTensor([stain_indicator]).to(device)
                stain_encoding = self.embedding(stain_indicator).squeeze()
                stain_encoding = torch.repeat_interleave(stain_encoding.unsqueeze(1), repeats=all_wsi_feats.shape[1], dim=1)
                all_wsi_feats = torch.cat([all_wsi_feats, stain_encoding], axis=-1) # [21, 2048, 544]
            
            # forward
            slide_embeddings, token_embeddings = self.wsi_embedders(all_wsi_feats, return_preattn_feats=True, n_views=n_views)
            ## ([b*7, 3, 512, 4])(num_view=3, heads=4)   ([b*7, num_patch, 512, heads])
        
            # re-arrange tokens into bs, n_mod, n_tokens, d                                                                      
            token_embeddings = token_embeddings.view(bs * n_mod, n_tokens, -1)          
            token_embeddings = token_embeddings.view(bs, n_mod, n_tokens, -1)           
            token_embeddings = self.token_projector(token_embeddings)          # ([3, 7, 2048, 128])                  
            
            # apply post-attn network to all embeddings
            d_out, n_heads = slide_embeddings.shape[-2], slide_embeddings.shape[-1]     
            slide_embeddings = slide_embeddings.view(bs*n_mod, -1, d_out * n_heads)        
            slide_embeddings = self.projector(slide_embeddings)                         
            slide_embeddings = slide_embeddings.view(bs, n_mod, -1, d_out)     # ([3, 7, 3, 512])  
            
            # format output 
            for idx, modality in enumerate(self.modalities):    # 对三个views进行混合
                
                slide_emb = slide_embeddings[:, idx, :, :]             
                token_emb = token_embeddings[:, idx, :]                
                if modality == "HE":
                    slide_emb = slide_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod-1)            
                    token_emb = token_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod-1)          
                # print(modality)   
                # print(slide_emb.shape)
                # print(token_emb.shape)
                # print()
                all_embeddings[modality] = slide_emb 
                all_token_embeddings[modality] = token_emb

            return all_embeddings, all_token_embeddings     # [bs, n_mod, d_out]; [bs, n_mod, n_tokens, d_model]

        elif inference_patch_embedding:     # inference return patch embeddings
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            all_wsi_feats = all_wsi_feats.view(bs*n_mod, n_tokens, d_in)

            # add stain specific encodings if asked for 
            if self.stain_encoding: # True
                stain_indicator = []
                for i in range(n_mod):
                    stain_indicator += [i]*bs
                stain_indicator = torch.LongTensor([stain_indicator]).to(device)
                stain_encoding = self.embedding(stain_indicator).squeeze()
                stain_encoding = torch.repeat_interleave(stain_encoding.unsqueeze(1), repeats=all_wsi_feats.shape[1], dim=1)
                all_wsi_feats = torch.cat([all_wsi_feats, stain_encoding], axis=-1) # [21, 2048, 544]
            
            # forward
            token_embeddings = self.wsi_embedders(all_wsi_feats, return_preattn_feats=True, n_views=n_views, inference_patch_embedding=inference_patch_embedding)
            ## ([b*7, 3, 512, 4])(num_view=3, heads=4)   ([b*7, num_patch, 512, heads])
            all_embeddings = None

            # re-arrange tokens into bs, n_mod, n_tokens, d                                                                      
            token_embeddings = token_embeddings.view(bs * n_mod, n_tokens, -1)          
            token_embeddings = token_embeddings.view(bs, n_mod, n_tokens, -1)           
            token_embeddings = self.token_projector(token_embeddings)          # ([3, 7, 2048, 128])                  

            # # apply post-attn network to all embeddings
            # d_out, n_heads = slide_embeddings.shape[-2], slide_embeddings.shape[-1]     
            # slide_embeddings = slide_embeddings.view(bs*n_mod, -1, d_out * n_heads)        
            # slide_embeddings = self.projector(slide_embeddings)                         
            # slide_embeddings = slide_embeddings.view(bs, n_mod, -1, d_out)     # ([3, 7, 3, 512])  
            
            # format output 
            for idx, modality in enumerate(self.modalities):    # 对三个views进行混合
                # slide_emb = slide_embeddings[:, idx, :, :]             
                token_emb = token_embeddings[:, idx, :]  
                if modality == "HE":    # HE embedding会复制6份
                    # slide_emb = slide_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod-1)            
                    token_emb = token_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod-1)          

                # all_embeddings[modality] = slide_emb 
                all_token_embeddings[modality] = token_emb.cpu()

            return all_embeddings, all_token_embeddings, slide_id     # [bs, n_mod, d_out]; [bs, n_mod, n_tokens, d_model]            


        # handle multiple stains and stain encodings during eval
        elif not train and not return_attention:
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            
            for stain_idx in range(n_mod):

                # if requesting a specific stain
                if custom_stain_idx:
                    stain_name = self.modalities[custom_stain_idx]
                # if you want just the zero index
                else:
                    stain_name = self.modalities[stain_idx]

                # ok with stain_idx as n_mods is 1 always
                curr_stain_feats = all_wsi_feats[:, stain_idx, : :]

                # if stain_encodings, then add those to curr_stain_feats
                if self.stain_encoding:

                    # if requesting specific stain then get its key
                    if custom_stain_idx:
                        key = custom_stain_idx
                    else:
                        key = stain_idx

                    stain_indicator = torch.LongTensor([[key]*bs]).to(device)
                    stain_encoding = self.embedding(stain_indicator)
                    stain_encoding = torch.repeat_interleave(stain_encoding, repeats=n_tokens, dim=1)
                    curr_stain_feats = torch.cat([curr_stain_feats, stain_encoding], axis=-1)
                
                # get the model output 
                stain_embedding = self.wsi_embedders(curr_stain_feats)

                # apply post-attn network 
                d_out, n_heads = stain_embedding.shape[-2], stain_embedding.shape[-1]
                stain_embedding = stain_embedding.view(bs*n_mod, d_out * n_heads)
                stain_embedding = self.projector(stain_embedding)
                stain_embedding = stain_embedding.view(bs, n_mod, d_out)

                # save
                all_embeddings[stain_name] = stain_embedding
            
            return all_embeddings
        
        # if returning attention 
        else:
            stain_name = "HE"
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            HE_embedding, raw_attention = self.wsi_embedders(all_wsi_feats[:, HE_POSITION, : :], return_attention=True) 
            d_out, n_heads = HE_embedding.shape[-2], HE_embedding.shape[-1]
            HE_embedding = HE_embedding.view(bs*n_mod, d_out * n_heads)
            HE_embedding = self.projector(HE_embedding)
            HE_embedding = HE_embedding.view(bs, n_mod, d_out)

            # save
            return HE_embedding, raw_attention

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output.

    Attributes:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output.
        blocks (nn.Sequential): The sequential blocks of the MLP model.

    """

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks=nn.Sequential(
            self.build_block(in_dim=self.input_dim, out_dim=int(self.input_dim)),
            self.build_block(in_dim=int(self.input_dim), out_dim=int(self.input_dim)),
            nn.Linear(in_features=int(self.input_dim), out_features=self.output_dim),
        )
        
    def build_block(self, in_dim, out_dim):
        """
        Build a block of the MLP model.

        Args:
            in_dim (int): The dimensionality of the input features for the block.
            out_dim (int): The dimensionality of the output for the block.

        Returns:
            nn.Sequential: The sequential block of the MLP model.

        """
        return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
        )

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.blocks(x)
        return x


class ProjHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the ProjHead module.

        Args:
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super(ProjHead, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=int(self.input_dim)),
                nn.LayerNorm(int(self.input_dim)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(self.input_dim) ,out_features=self.output_dim),
        )
        
    def forward(self, x):
        """
        Perform forward pass of the ProjHead module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.layers(x)
        return x

class ABMILEmbedder(nn.Module):
    """
    ABMIL. 
    """

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = 'regular',
    ) -> None:
        """
        """
        super(ABMILEmbedder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_attention_params = pre_attention_params
        self.attention_params = attention_params
        self.n_heads = attention_params['params']["n_heads"]

        # 1- build pre-attention params 
        self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        # 3- set aggregation type 
        self.agg_type = aggregation  

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params 
        """
        self.pre_attn = nn.Sequential(
            nn.Linear(params['input_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_dim'], params['hidden_dim']), # expanding by n_classes
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_dim'], params['hidden_dim']*self.n_heads), # expanding by n_classes
            nn.LayerNorm(params['hidden_dim']*self.n_heads),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_attention_params(self, attn_model='ABMIL', params=None):
        """
        Build attention params 
        """
        if attn_model == 'ABMIL':
            self.attn = nn.ModuleList([BatchedABMIL(**params).to(self.device) for i in range(self.n_heads)])
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL')


    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False, 
        return_preattn_feats: bool = False,
        n_views = 1,
        inference_patch_embedding=False
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens 
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """

        # pre-attention common for all stains, shared across all heads
        embeddings = self.pre_attn(bags)    # [b*7, num_patch. 512+32] --> [b*7, num_patch, 2048] patch_level features 

        if self.n_heads > 1:
            embeddings = rearrange(embeddings, 'b t (e c) -> b t e c',c=self.n_heads)
        else:
            embeddings = embeddings.unsqueeze(-1) # for consistency later on

        # for returning, save embeddings
        token_embeddings = embeddings # ([b*7, num_patch, 512, heads])

        if inference_patch_embedding:
            return token_embeddings


        # individual attentions for each stain
        attention = []
        raw_attention = []
        for i, attn_net in enumerate(self.attn):
            processed_attention, untouched_attention = attn_net(embeddings[:, :, :, i], return_raw_attention = True)
            attention.append(processed_attention)
            raw_attention.append(untouched_attention)
        attention = torch.stack(attention, dim=-1) # return post softmax attention
        raw_attention = torch.stack(raw_attention, dim=-1) # return post softmax attention

        if self.agg_type == 'regular':
            
            if n_views == 1:
                slide_embeddings = embeddings * attention
                slide_embeddings = torch.sum(slide_embeddings, dim=1)

            else:   # True  

                # 1. compute the whole view slide embeddings 
                slide_embeddings_wholeView = embeddings * attention # [21, 2048, 512, 4] * [21, 2048, 1, 4]=[21, 2048, 512, 4])
                slide_embeddings_wholeView = torch.sum(slide_embeddings_wholeView, dim=1)
                slide_embeddings_wholeView = slide_embeddings_wholeView.unsqueeze(1)    # ([21, 1, 512, 4])
                
                # 2. compute two intra views
                all_indices = np.arange(embeddings.shape[1])
                np.random.shuffle(all_indices)
                midpoint = len(all_indices) // 2
                list_of_indices = [all_indices[:midpoint], all_indices[midpoint:]]
                try:
                    embeddings = torch.cat([embeddings[:, indices, :, :].unsqueeze(1) for indices in list_of_indices], dim=1) 
                except:
                    pdb.set_trace()
                attention = torch.cat([F.softmax(raw_attention[:, indices], dim=1).unsqueeze(1) for indices in list_of_indices], dim=1) # ([21, 2, 1024, 1, 4])
                embeddings = embeddings * attention # ([21, 2, 1024, 512, 4])
                slide_embeddings_smallViews = torch.sum(embeddings, dim=2)  # ([21, 2, 512, 4])

                # 3. concat all views
                slide_embeddings = torch.concat([slide_embeddings_wholeView, slide_embeddings_smallViews], dim=1)   # ([21, 3, 512, 4])

        else:
            raise NotImplementedError('Agg type not supported. Options are "regular".')
        
        if return_attention:
            return slide_embeddings, raw_attention
        
        if return_preattn_feats:
            return slide_embeddings, token_embeddings

        return slide_embeddings


def load_checkpoint(args, ssl_model, path_to_checkpoint=None):
    """
    Loads a checkpoint file and updates the state of the SSL model.

    Args:
        args (Namespace): The command-line arguments.
        ssl_model (nn.Module): The SSL model to update.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint file is corrupted or incompatible with the model.

    """
    import os
    # load checkpoint
    if path_to_checkpoint is not None:
        state_dict = torch.load(path_to_checkpoint)
    else:
        state_dict = torch.load(os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
    
    # load weights into model
    try:
        ssl_model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        ssl_model.load_state_dict(new_state_dict)
        print('Model loaded by removing module in state dict...')
    
    return ssl_model

if __name__ == '__main__':
    import argparse

    config_dict = {
        "wsi_encoder_hidden_dim": 512,
        "use_stain_encoding": True,
        "patch_embedding_dim": 1536,
        "activation": 'softmax',
        "n_heads" : 4,
        "MODALITIES": ['HE','CD3', 'CD4', 'CD8', 'CD19', 'CD68', 'Foxp3'],
        "wsi_encoder": "abmil"
    }

    args = argparse.Namespace(**config_dict)
    model = MADELEINE(args, stain_encoding=True)   # when inference
    path_to_checkpoint = '/data/jsh/MADELEINE-main/results/DEBUG_dc8debec3197e38b1d2a16adad9c7d07/model_88.pt'
    load_checkpoint(args, model, path_to_checkpoint=path_to_checkpoint)
