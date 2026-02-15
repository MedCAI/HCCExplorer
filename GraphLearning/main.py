import os
import torch
import argparse
import torch.optim as optim
from utils.tools import read_yaml
from dataset.dataloader_graph import MLDataLoader_Graph
from utils.trainer_hcc import Trainer_hcc
import importlib
import inspect

cpu_num = 8
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


def init_model(args, cfg):
	name = cfg.Model['name']
	try:
		Model = getattr(importlib.import_module(f'models.{name}'), name)
	except:
		raise ValueError('Invalid Module File Name or Invalid Class Name!')
	class_args = inspect.getfullargspec(Model.__init__).args[1:]    # ['i_classifier', 'b_classifier']

	args_dict = {}
	for _args in class_args:
		if _args in cfg.Model.keys():
			args_dict[_args] = cfg.Model[_args]
   
	if args.reset_parameter:
		for _args in class_args:
			if hasattr(args, _args):
				args_dict[_args] = getattr(args, _args)
			
	args_dict['Argument'] = args 
	model = Model(**args_dict)
	if args.multi_gpu_mode == 'DataParallel':
		model = torch.nn.DataParallel(model)
	return model

def init_dataloader(args, cfg, i):
	data_loader = MLDataLoader_Graph(cfg.Model['name'], args.csv_path, args.data_dir, args.split_dir, task=cfg.Model.task, fold=i,
									multi_hcc=args.multi_hcc, root_hcc=args.root_hcc, feature_dim=cfg.Model.input_dim)
	return data_loader

def main(args, cfg):

	for i in range(0, args.folds):
		if args.test_phase:
			i = args.test_fold
		data_loader = init_dataloader(args, cfg, i)
		train_dataloader, val_dataloader = data_loader.get_dataloader()

		model = init_model(args, cfg)
		model.to(args.device)
		if cfg.Optimizer.opt == 'adam':
			optimizer = optim.Adam(model.parameters(), lr=cfg.Optimizer.lr, weight_decay=cfg.Optimizer.weight_decay)
		elif cfg.Optimizer.opt == 'sgd':
			optimizer =  optim.SGD(model.parameters(), lr=cfg.Optimizer.lr, momentum=cfg.Optimizer.momentum)
		else:
			optimizer = optim.RAdam(model.parameters(), lr=cfg.Optimizer.lr, weight_decay=cfg.Optimizer.weight_decay)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.coslr_T)
		
		trainer = Trainer_hcc(model, optimizer, args, cfg, args.device, i, train_dataloader, val_dataloader, lr_scheduler=lr_scheduler)
		trainer.train(i)


if __name__ == '__main__':
	args = argparse.ArgumentParser(description='PyTorch Template')
	
	args.add_argument('--save_dir',type=str, help='')
	args.add_argument('--epochs',type=int, default=45, help='')
	args.add_argument('--folds',type=int, default=5, help='')
	args.add_argument('--test_fold',type=int, default=0, help='')
	args.add_argument('--csv_path',type=str, help='')
	args.add_argument('--data_dir',type=str, help='')
	args.add_argument('--split_dir',type=str, help='')
	args.add_argument('--optimizer',type=str, default='adam', help='')
	args.add_argument('--lr',type=float, default=0.0001, help='')
	args.add_argument('--weight_decay',type=float, default=0.000005)
	args.add_argument('--momentum',type=float, default=0.4)
	args.add_argument('--coslr_T', type=int, default=10)
	args.add_argument('--lr_scheduler', action='store_true', default=False)
	args.add_argument('--num_class',type=int, default=4)
	args.add_argument('--config', type=str, help='yaml file')
	args.add_argument('--batch_size',type=int, default=1, help='')
	args.add_argument('--device',type=str, default='cuda', help='')
	args.add_argument('--multi_gpu_mode',type=str, default='DataParallel', help='Gpus')
	args.add_argument('--reset_parameter', action='store_true', default=False)
	args.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
	args.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
	args.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
	args.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
	args.add_argument("--residual_connection", default="Y", help="Y/N", type=str)
	args.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
	args.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
	args.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
	args.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
	args.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
	args.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
	args.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
	args.add_argument('--multi_hcc', action='store_true', default=False, help='test')
	args.add_argument("--root_hcc", default=None, help="BatchNorm=batch/LayerNorm=layer", type=str)
	
	args = args.parse_args()
	cfg = read_yaml(args.config)

	main(args, cfg)