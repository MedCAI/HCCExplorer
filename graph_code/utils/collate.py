import torch
import numpy as np
from torch_geometric.data import Data

def collate_MT(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label_surv = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    e = torch.FloatTensor([item[3] for item in batch])
    case_id = np.array([item[4] for item in batch])
    wsi_id = np.array([item[5] for item in batch])
    return [img, label_surv, event_time, e, case_id, wsi_id]


def collate_MT_graph(batch):
    imgs = [item[0] for item in batch if isinstance(item[0], Data)]
    label_surv = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    e = torch.FloatTensor([item[3] for item in batch])
    case_id = np.array([item[4] for item in batch])
    wsi_id = np.array([item[5] for item in batch])
    return [imgs[0], label_surv, event_time, e, case_id, wsi_id]


def collate_MT_graph_hcc(batch):
    imgs = [item[0] for item in batch if isinstance(item[0], Data)]
    hccs = [item[6] for item in batch]
    label_surv = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    e = torch.FloatTensor([item[3] for item in batch])
    case_id = np.array([item[4] for item in batch])
    wsi_id = np.array([item[5] for item in batch])
    return [imgs[0], hccs[0], label_surv, event_time, e, case_id, wsi_id]