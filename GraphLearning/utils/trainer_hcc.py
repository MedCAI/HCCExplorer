import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.tools import EarlyStopping, get_logger
import os
from tqdm import tqdm
from utils.loss import nll_loss
from utils.metric import c_index
import pandas as pd
import gc

class Trainer_hcc():
	"""
	Trainer class
	"""
	def __init__(self, model, optimizer, args, cfg, device, fold,
				 train_data_loader, valid_data_loader=None, num_class=2, lr_scheduler = None):
		self.args = args
		self.cfg = cfg
		self.device = device
		self.model = model
		self.optimizer = optimizer
		self.data_loader = train_data_loader
		self.valid_data_loader = valid_data_loader
		self.fold = fold
		self.epochs = args.epochs
		self.writer = SummaryWriter(args.save_dir + '/{}'.format(fold), flush_secs=15)
		self.model_path = os.path.join(args.save_dir, str(fold), 'model.pt')
		self.gc = 32
		self.lr_scheduler = lr_scheduler
		self.best_valid_cindex = 0.
		self.logger = get_logger('')

		self.debug_save = True

	def _train_epoch(self, epoch, early_stopping):
		self.model.train()
	
		all_risk_scores = np.zeros((len(self.data_loader)))
		all_censorships = np.zeros((len(self.data_loader)))
		all_event_times = np.zeros((len(self.data_loader)))

		train_loss, surv_loss_log = 0., 0.

		for batch_idx, (path_features, hcc_features, Y_surv, event_time, if_event, case_id, wsi_id) in enumerate(tqdm(self.data_loader)):

			path_features = path_features.to(self.device)
			hcc_features = hcc_features.to(self.device)
			Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
			c = 1.0 - if_event.type(torch.FloatTensor).to(self.device)

			surv_logits, hazards, S = self.model(wsi=path_features, hcc=hcc_features, label=Y_surv)
			surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)
			loss = surv_loss / self.gc
			surv_loss_log += surv_loss.item()
			train_loss += loss.item()

			loss.backward()

			if (batch_idx + 1) % self.gc == 0: 
				self.optimizer.step()
				self.optimizer.zero_grad()

			risk = -torch.sum(S, dim=1).detach().cpu().numpy()
			all_risk_scores[batch_idx] = risk
			all_censorships[batch_idx] = c.item()
			all_event_times[batch_idx] = event_time

		cindex = c_index(all_censorships, all_event_times, all_risk_scores)

		train_loss /= len(self.data_loader)

		self.writer.add_scalar('train/loss', train_loss, epoch)
		self.writer.add_scalar('train/surv_loss', surv_loss_log, epoch)
		self._valid_epoch(epoch, early_stopping)


	def _valid_epoch(self, epoch, early_stopping):

		self.model.eval()
	
		all_risk_scores = np.zeros((len(self.valid_data_loader)))
		all_censorships = np.zeros((len(self.valid_data_loader)))
		all_event_times = np.zeros((len(self.valid_data_loader)))

		val_loss, surv_loss_log = 0., 0.

		for batch_idx, (path_features, hcc_features, Y_surv, event_time, c, case_id, wsi_id) in enumerate(tqdm(self.valid_data_loader)):
			hcc_features = hcc_features.to(self.device)
			path_features = path_features.to(self.device)
			Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
			c = 1.0 - c.type(torch.FloatTensor).to(self.device)

			with torch.no_grad():   
				surv_logits, hazards, S = self.model(wsi = path_features, hcc=hcc_features, label=Y_surv)
				surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)
				loss = surv_loss / self.gc
				surv_loss_log += surv_loss.item()
				val_loss += loss.item()

			risk = -torch.sum(S, dim=1).detach().cpu().numpy()
			all_risk_scores[batch_idx] = risk
			all_censorships[batch_idx] = c.item()
			all_event_times[batch_idx] = event_time

		cindex = c_index(all_censorships, all_event_times, all_risk_scores)

		val_loss /= len(self.data_loader)

		self.writer.add_scalar('val/loss', val_loss, epoch)
		self.writer.add_scalar('val/surv_loss', surv_loss_log, epoch)
		metric = cindex
		early_stopping(epoch=epoch, metric=metric, models=self.model, ckpt_name=os.path.join(self.args.save_dir, str(self.fold)))   # 在验证阶段会累积score不改变的次数


	def train(self, fold=0):
		early_stopping = EarlyStopping(warmup=2, patience=5, stop_epoch=18, verbose = True, logger=self.logger, multi_gpus=self.args.multi_gpu_mode)

		for epoch in range(0, self.epochs):
			print(f'Epoch : {epoch}:')
			self._train_epoch(epoch, early_stopping)
			if self.cfg.Lr_scheduler._if:
				self.lr_scheduler.step()
			gc.collect()
			torch.cuda.empty_cache()

	def load_model_state(self,model, state_dict_path):
		state_dict = torch.load(state_dict_path, map_location='cpu')
		new_state_dict = {}
		for k, v in state_dict.items():
			if k == "gat.lin_src.weight":
				new_state_dict["gat.lin.weight"] = v 
			elif k in ["gat.lin_dst.weight"]:
				continue
			else:
				new_state_dict[k] = v
		model.load_state_dict(new_state_dict, strict=False)