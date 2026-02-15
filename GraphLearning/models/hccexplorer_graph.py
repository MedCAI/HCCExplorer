import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from models.pre_layer import preprocess
from models.post_layer import postprocess

class GCN(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
				 dropout=0.5, save_mem=True, use_bn=True):
		super(GCN, self).__init__()

		self.convs = nn.ModuleList()
		self.convs.append(
			GCNConv(in_channels, hidden_channels, cached=not save_mem))

		self.bns = nn.ModuleList()
		self.bns.append(nn.BatchNorm1d(hidden_channels))
		for _ in range(num_layers - 2):
			self.convs.append(
				GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
			self.bns.append(nn.BatchNorm1d(hidden_channels))
		self.convs.append(
			GCNConv(hidden_channels, out_channels, cached=not save_mem))

		self.dropout = dropout
		self.activation = F.relu
		self.use_bn = use_bn

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		for bn in self.bns:
			bn.reset_parameters()

	def forward(self, x, edge_index):
		edge_weight=None
		for i, conv in enumerate(self.convs[:-1]):
			if edge_weight is None:
				x = conv(x, edge_index)
			else:
				x=conv(x,edge_index,edge_weight)
			if self.use_bn:
				x = self.bns[i](x)
			x = self.activation(x)
			x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.convs[-1](x, edge_index)
		return x
	
def full_attention_conv(qs, ks, vs, output_attn=False):
	# normalize input
	qs = qs / torch.norm(qs, p=2)  # [N, H, M]
	ks = ks / torch.norm(ks, p=2)  # [L, H, M]
	N = qs.shape[0]

	# numerator
	kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
	attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
	attention_num += N * vs

	# denominator
	all_ones = torch.ones([ks.shape[0]]).to(ks.device)
	ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
	attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

	# attentive aggregated results
	attention_normalizer = torch.unsqueeze(
		attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
	attention_normalizer += torch.ones_like(attention_normalizer) * N
	attn_output = attention_num / attention_normalizer  # [N, H, D]

	# compute attention for visualization if needed
	if output_attn:
		attention=torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1) #[N, N]
		normalizer=attention_normalizer.squeeze(dim=-1).mean(dim=-1,keepdims=True) #[N,1]
		attention=attention/normalizer

	if output_attn:
		return attn_output, attention
	else:
		return attn_output

class TransConvLayer(nn.Module):
	'''
	transformer with fast attention
	'''
	def __init__(self, in_channels,
				 out_channels,
				 num_heads,
				 use_weight=True):
		super().__init__()
		self.Wk = nn.Linear(in_channels, out_channels * num_heads)
		self.Wq = nn.Linear(in_channels, out_channels * num_heads)
		if use_weight:
			self.Wv = nn.Linear(in_channels, out_channels * num_heads)

		self.out_channels = out_channels
		self.num_heads = num_heads
		self.use_weight = use_weight

	def reset_parameters(self):
		self.Wk.reset_parameters()
		self.Wq.reset_parameters()
		if self.use_weight:
			self.Wv.reset_parameters()

	def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
		query = self.Wq(query_input).reshape(-1,
											 self.num_heads, self.out_channels)
		key = self.Wk(source_input).reshape(-1,
											self.num_heads, self.out_channels)
		if self.use_weight:
			value = self.Wv(source_input).reshape(-1,
												  self.num_heads, self.out_channels)
		else:
			value = source_input.reshape(-1, 1, self.out_channels)

		# compute full attentive aggregation
		if output_attn:
			attention_output, attn = full_attention_conv(
				query, key, value, output_attn)  # [N, H, D]
		else:
			attention_output = full_attention_conv(
				query, key, value)  # [N, H, D]

		final_output = attention_output
		final_output = final_output.mean(dim=1)

		if output_attn:
			return final_output, attn
		else:
			return final_output

class TransConv(nn.Module):
	def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
				 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False, jk=False):
		super().__init__()
		self.jk = jk
		self.convs = nn.ModuleList()
		self.fcs = nn.ModuleList()
		self.fcs.append(nn.Linear(in_channels, hidden_channels))
		self.bns = nn.ModuleList()
		self.bns.append(nn.LayerNorm(hidden_channels))
		for i in range(num_layers):
			self.convs.append(
				TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
			self.bns.append(nn.LayerNorm(hidden_channels))

		self.dropout = dropout
		self.activation = F.relu
		self.use_bn = use_bn
		self.residual = use_residual
		self.alpha = alpha
		self.use_act=use_act

	def reset_parameters(self):
		for conv in self.convs:
			conv.reset_parameters()
		for bn in self.bns:
			bn.reset_parameters()
		for fc in self.fcs:
			fc.reset_parameters()

	def forward(self, x, edge_index):
		edge_weight = None
		layer_ = []
		x_local = 0
		x = self.fcs[0](x)
		if self.use_bn:
			x = self.bns[0](x)
		x = self.activation(x)
		x = F.dropout(x, p=self.dropout, training=self.training)

		layer_.append(x)

		for i, conv in enumerate(self.convs):
			x = conv(x, x, edge_index, edge_weight)
			if self.residual:
				x = self.alpha * x + (1-self.alpha) * layer_[i]
			if self.use_bn:
				x = self.bns[i+1](x)
			if self.use_act:
				x = self.activation(x) 
			x = F.dropout(x, p=self.dropout, training=self.training)
			layer_.append(x)
			if self.jk:
				x_local = x_local + x
			else:
				x_local = x
		x = x_local
		return x

	def get_attentions(self, x):
		layer_, attentions = [], []
		x = self.fcs[0](x)
		if self.use_bn:
			x = self.bns[0](x)
		x = self.activation(x)
		layer_.append(x)
		for i, conv in enumerate(self.convs):
			x, attn = conv(x, x, output_attn=True)
			attentions.append(attn)
			if self.residual:
				x = self.alpha * x + (1 - self.alpha) * layer_[i]
			if self.use_bn:
				x = self.bns[i + 1](x)
			layer_.append(x)
		return torch.stack(attentions, dim=0)

class hccexplorer_graph(nn.Module):
	def __init__(self, Argument, input_dim, hidden_channels=256, n_classes=4, num_layers=2, global_layers=2, num_heads=1, another_channels=512,
				 alpha=0.5, dropout=0.15, use_bn=True, use_residual=True, use_weight=True, use_graph=True, return_atten=False, return_channel_atten=False,
				 use_act=False, graph_weight=0.8, aggregate='add', jk=False, preprocess_flag=False, beta=-1, device='cuda'):
		super().__init__()
		
		self.device = device
		self.preprocess_flag = preprocess_flag
		self.return_atten = return_atten
		self.return_channel_atten = return_channel_atten
		if preprocess_flag == True:
			Argument.initial_dim = input_dim
			input_dim = input_dim*2
			self.preprocess = preprocess(Argument)       
			self.postprocess = postprocess(64, 3, 64, (Argument.MLP_layernum-1), dropout)
			if aggregate=='add':
				self.fc=nn.Linear(int(hidden_channels/4),n_classes)
			elif aggregate=='cat':
				self.fc=nn.Linear(int(2*hidden_channels/4),n_classes)
			else:
				raise ValueError(f'Invalid aggregate type:{aggregate}')
		else:
			if aggregate=='add':
				self.fc=nn.Linear(hidden_channels,n_classes)
			elif aggregate=='cat':
				self.fc=nn.Linear(2*hidden_channels,n_classes)
			else:
				raise ValueError(f'Invalid aggregate type:{aggregate}')
		input_dim1 = 128*7 

		self.gnn = GCN(in_channels=input_dim1,
		  hidden_channels=hidden_channels,
		  out_channels=hidden_channels,
		  num_layers=global_layers,
		  dropout=dropout)
		
		self.trans_conv=TransConv(input_dim1,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight,jk=jk)
		self.use_graph=use_graph
		self.graph_weight=graph_weight
		self.use_act=use_act
		self.aggregate=aggregate

		self.params1=list(self.trans_conv.parameters())
		self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
		self.params2.extend(list(self.fc.parameters()) )

		self.he_projector = nn.Sequential(
			nn.Linear(input_dim, 512),
			nn.ReLU(),
			nn.Linear(512, 128)
		)
		
		layers = 3
		self.pre_lns = torch.nn.ModuleList()
		self.local_convs = torch.nn.ModuleList()
		self.h_lins = torch.nn.ModuleList()
		self.lins = torch.nn.ModuleList()
		self.lns = torch.nn.ModuleList()
		self.beta = beta
		if self.beta < 0:
			self.betas = torch.nn.Parameter(torch.zeros(layers,128))
		else:
			self.betas = torch.nn.Parameter(torch.ones(layers,128)*self.beta)
		for _ in range(layers):
			self.pre_lns.append(torch.nn.LayerNorm(128))
			self.h_lins.append(torch.nn.Linear(128, 128))
			self.local_convs.append(GATConv(in_channels=128, out_channels=64, heads=2, 
						concat=True, add_self_loops=False, bias=False))
			self.lins.append(torch.nn.Linear(128, 128))
			self.lns.append(torch.nn.LayerNorm(128))
			self.dropout = dropout

	def _build_edge_index_full_connected(self, num_modalities):
		src, dst = [], []
		for i in range(num_modalities):
			for j in range(num_modalities):
				if i != j:
					src.append(i)
					dst.append(j)
		return torch.tensor([src, dst], dtype=torch.long).to(self.device)

	def _build_edge_index_single_connected(self, num_modalities):
		src = []
		dst = []
		for j in range(1, num_modalities):
			src.append(0)
			dst.append(j)
		return torch.tensor([src, dst], dtype=torch.long).to(self.device)

	def _batch_edge_index(self, edge_index, batch_size, num_nodes_per_graph):
		edge_indices = []
		for i in range(batch_size):
			offset = i * num_nodes_per_graph
			edge = edge_index + offset
			edge_indices.append(edge)
		return torch.cat(edge_indices, dim=1).to(self.device)

	def process_hcc(self, patch_modal_feats, return_attention_weights=False):
		N, M, D = patch_modal_feats.shape
		x = patch_modal_feats.view(N * M, D)
		edge_index = self._build_edge_index_full_connected(num_modalities=M)
		edge_index_batch = self._batch_edge_index(edge_index, N, M)
		x_local = 0
		for i, local_conv in enumerate(self.local_convs):
			h = self.h_lins[i](x)
			h = F.leaky_relu(h)
			x, (edge_index_out, attn_weights) = local_conv(x, edge_index_batch,
						return_attention_weights=return_attention_weights) 
			beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
			if self.beta < 0:
				beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
			else:
				beta = self.betas[i].unsqueeze(0)
			x = (1-beta)*self.lns[i](h*x) + beta*x
			x_local = x_local + x

		fused_feat = x_local.view(N, -1)
		if return_attention_weights:
			return fused_feat, attn_weights.mean(dim=1)
		return fused_feat

	def forward(self,**kwargs):
		data = kwargs['wsi']      
		multi_hcc = kwargs['hcc']

		x = data.x 
		batch = data.batch
		edge_index = data.edge_index
		
		if self.preprocess_flag:
			x, preprocess_edge_attr = self.preprocess(data)

		modal_feats = multi_hcc
		
		he_projected = self.he_projector(x)      
		he_feat = he_projected.unsqueeze(1)
		patch_modal_feats = torch.cat([he_feat, modal_feats], dim=1)

		if self.return_channel_atten:
			x, atten = self.process_hcc(patch_modal_feats, self.return_channel_atten)
			
		else:
			x = self.process_hcc(patch_modal_feats, self.return_channel_atten) # ([20831, 896])

		x1=self.trans_conv(x,edge_index)
		
		if self.return_atten:
			return self.get_attentions(x)
		
		if self.use_graph:
			x2=self.gnn(x,edge_index)
			if self.aggregate=='add':
				x=self.graph_weight*x2+(1-self.graph_weight)*x1
			else:
				x=torch.cat((x1,x2),dim=1)
		else:
			x=x1
		
		global_features = global_mean_pool(x, batch) 
		if self.preprocess_flag:
			global_features = self.postprocess(global_features, data.batch)
		logits=self.fc(global_features)
		hazards = torch.sigmoid(logits)
		S = torch.cumprod(1 - hazards, dim=1)

		if self.return_channel_atten:
			return logits, hazards, S, atten
		else:
			return logits, hazards, S
	
	def get_attentions(self, x):
		attns=self.trans_conv.get_attentions(x)
		attn_mean_heads = attns.mean(dim=0)
		node_attention = attn_mean_heads.mean(dim=1)

	def reset_parameters(self):
		self.trans_conv.reset_parameters()
		if self.use_graph:
			self.gnn.reset_parameters()