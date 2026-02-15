import torch
from dataset_graph_recur import MLDataset_Graph_Recur
from dataset_graph_surv import MLDataset_Graph_Surv
from utils.tools import *
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler, RandomSampler
from utils.tools import make_weights_for_balanced_classes_split
from utils.collate import collate_MT_graph, collate_MT_graph_hcc

class MLDataLoader_Graph(DataLoader):
    def __init__(self, model_name, csv_path, data_dir, split_dir, fold, task='recur', shuffle=True, num_workers=4,
                  training=True, multi_hcc=False, root_hcc=None, feature_dim=1536):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.task = task
        self.num_workers = num_workers
        self.multi_hcc=multi_hcc
        if task == 'recur':
            self.dataset = MLDataset_Graph_Recur(model_name=model_name,
                                                csv_path = csv_path,
		    								    data_dir= data_dir,
		    								    shuffle = False, 
		    								    n_bins=4,
		    								    recur_label_col = 'DFS')
        elif task == 'survival':
            self.dataset = MLDataset_Graph_Surv(model_name=model_name,
                                                csv_path = csv_path,
                                                data_dir= data_dir,
                                                shuffle = False, 
                                                n_bins=4,
                                                recur_label_col = 'OS',
                                                feature_dim=feature_dim,
                                                multi_hcc=multi_hcc,
                                                root_hcc=root_hcc)  
        train_dataset, val_dataset = self.dataset.return_splits(from_id=False, csv_path='{}/split{}.csv'.format(split_dir, fold))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_split_loader(self, split_dataset, weighted = True, batch_size=1):
        """
            return either the validation loader or training loader 
        """
        if self.multi_hcc:
            collate = collate_MT_graph_hcc
        else:
            collate = collate_MT_graph

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {'num_workers': self.num_workers} if device.type == "cuda" else {}

        if weighted:
            weights = make_weights_for_balanced_classes_split(self.task, split_dataset)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        return loader

    def get_dataloader(self):
        train_dataloader = self.get_split_loader(self.train_dataset, weighted=True)
        val_dataloader = self.get_split_loader(self.val_dataset, weighted=False)
        return train_dataloader, val_dataloader