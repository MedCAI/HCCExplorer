import numpy as np
import pandas as pd
import torch
import os
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.transforms import Polar
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

HCC_TYPES = ['CD3', 'CD4', 'CD8', 'CD19', 'CD68', 'Foxp3']

class MLDataset_Graph_Surv(Dataset):

    def __init__(self, model_name='mil', csv_path = 'label.csv', data_dir=None, shuffle = False, seed = 7,
                  n_bins = 4, recur_label_col = None, eps=1e-6, feature_dim=512, multi_hcc=False, root_hcc=None):
        super(MLDataset_Graph_Surv, self).__init__()
        self.model_name = model_name
        self.seed = seed
        self.data_dir = data_dir
        self.recur_label_col = recur_label_col
        self.polar_transform = Polar()
        self.feature_dim = feature_dim
        self.multi_hcc = multi_hcc
        self.root_hcc = root_hcc

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        columns_to_keep = ['id', 'recurrent', 'living_status', 'OS', 'DFS', 'slide_path']
        new_df = slide_data[columns_to_keep]
        wsi_id_list = [id.split('.svs')[0] for id in new_df['slide_path'].to_list()]
        new_df.insert(1, 'wsi_id', wsi_id_list)

        slide_data = new_df
        patients_df = slide_data.copy()
        recur_uncensored_df = patients_df[patients_df['living_status']==1]

        # recur prediction
        disc_labels, q_bins = pd.qcut(recur_uncensored_df[recur_label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[recur_label_col].max() + eps
        q_bins[0] = slide_data[recur_label_col].min() - eps
        disc_labels, q_bins = pd.cut(patients_df[recur_label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'recur_label', disc_labels.values.astype(int))
        slide_data = patients_df

        recur_label_dict = {}     
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                recur_label_dict.update({(i, c):key_count})
                key_count+=1
        for i in slide_data.index:
            key = slide_data.loc[i, 'recur_label']
            slide_data.at[i, 'disc_recur_label'] = (int)(key)
            censorship = slide_data.loc[i, 'living_status']
            key = (key, int(censorship))
            slide_data.at[i, 'recur_label'] = recur_label_dict[key] 
        self.recur_label_dict = recur_label_dict
        self.slide_data = slide_data
        
    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['wsi_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split_Graph(self.model_name, df_slice, recur_label_col=self.recur_label_col,data_dir=self.data_dir,
                                         feature_dim=self.feature_dim, multi_hcc=self.multi_hcc, root_hcc=self.root_hcc)
        else:
            split = None
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
        return train_split, val_split

    def __getitem__(self, idx):
        case_id = self.slide_data['id'][idx]
        wsi_id = self.slide_data['wsi_id'][idx]
        Y_recur = self.slide_data['disc_recur_label'][idx]
        event_time = self.slide_data[self.recur_label_col][idx]
        e = self.slide_data['living_status'][idx]

        data_dir = self.data_dir
        wsi_path  = os.path.join(data_dir, 'pt_files', '{}'.format(wsi_id)+'.pt')
        data_origin = torch.load(wsi_path)
        transfer = T.ToSparseTensor()
        
        data_re = Data(x=data_origin.x[:,:self.feature_dim], edge_index=data_origin.edge_index)
        mock_data = Data(x=data_origin.x[:,:self.feature_dim], edge_index=data_origin.edge_index, pos=data_origin.pos)        
        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr


        data = transfer(data_re)
        data.edge_attr = polar_edge_attr
        data.pos = data_origin.pos
        
        data.edge_index = to_undirected(data_origin.edge_index)
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index)
             
        from BatchWSI import BatchWSI      
        data = BatchWSI.from_data_list([data], update_cat_dims={'edge_latent': 1})
        
        if self.multi_hcc:
            hcc = torch.load(os.path.join(self.root_hcc, wsi_id+'.pt'))
            modal_keys = list(hcc.keys())
            modal_feats = [hcc[k] for k in modal_keys]  # list of [N, 128]
            hcc_features = torch.stack(modal_feats, dim=1)  # [N, M, 128]                

            return (data, Y_recur, event_time, e, case_id, wsi_id, hcc_features)
        else:
            return (data, Y_recur, event_time, e, case_id, wsi_id)

    def __len__(self) -> int:
        return len(self.slide_data)

    def getlabel(self, ids):
        return (self.slide_data['recur_label'][ids])

 
class Generic_Split_Graph(MLDataset_Graph_Surv):
    def __init__(self, model_name, slide_data, recur_label_col='survival_month', data_dir=None, feature_dim=None, multi_hcc=False, root_hcc=None):
        self.model_name = model_name
        self.use_h5 = False
        self.slide_data = slide_data
        self.recur_label_col = recur_label_col
        self.data_dir = data_dir
        self.recur_discclass_num = len(self.slide_data['recur_label'].value_counts())
        self.slide_recur_ids = [[] for i in range(self.recur_discclass_num)]
        self.polar_transform = Polar()
        self.feature_dim = feature_dim
        self.multi_hcc = multi_hcc
        self.root_hcc = root_hcc
        for i in range(self.recur_discclass_num):
            self.slide_recur_ids[i] = np.where(self.slide_data['recur_label'] == i)[0]
    
    def len(self):
        return len(self.slide_data)