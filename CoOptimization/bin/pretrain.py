# general
import os
import time
import torch

try:
	import wandb # type: ignore
	WANDB_ERROR = False
except:
	print("wandb not installed")
	WANDB_ERROR = True
import pdb

import logging
import json

# internal
from utils.setup_components import setup, setup_DownstreamDatasets, setup_dataloader, setup_dataset, setup_losses, setup_model, setup_optim
from utils.trainer import train_loop
from utils.utils import extract_slide_level_embeddings, load_checkpoint, set_deterministic_mode


if __name__ == "__main__":
	
	# set seed
	set_deterministic_mode(SEED=42)
	
	# geenral set up 
	args = setup()
	
	# set up dataset
	dataset = setup_dataset(args)

	# set up dataloader
	dataloader = setup_dataloader(args, dataset)
	
	# set up the downstream datasets
	# val_dataloaders = setup_DownstreamDatasets(args)
	
	# set up model
	ssl_model = setup_model(args)
	
	# set up optimizers
	optimizer, scheduler, scheduler_warmup = setup_optim(args, dataloader, ssl_model)
	
	# set up losses
	loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod = setup_losses(args)

	# main training loop
	best_rank = 0.
	if args.load_checkpoint is True:
		print('loading from ...')
		ssl_model = load_checkpoint(args, ssl_model, path_to_checkpoint='/data/jsh/MADELEINE-main/results/07_400_based/model_88.pt')

	for epoch in range(args.max_epochs):
		
		print(f"\nTraining for epoch {epoch}...\n")

		start = time.time()
		ep_loss, train_rank = train_loop(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)
		
		if args.log_ml and not WANDB_ERROR:
			wandb.log({"train_loss": ep_loss, "train_rank": train_rank})

			
		log_data = {"epoch": epoch, "train_loss": ep_loss, "train_rank": train_rank}
		logging.info(json.dumps(log_data))

		end = time.time()

		print(f"\nDone with epoch {epoch}")
		print(f"Total loss = {ep_loss:.3f}")
		print(f"Train rank = {train_rank:.3f}")
		print(f"Total time = {end-start:.3f} seconds")

		# Stop training based on rank of the training samples. 
		# dont save for the first 20 epochs
		if epoch > 20 and train_rank > best_rank:
			print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
			best_rank = train_rank
			torch.save(ssl_model.state_dict(), os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
			print()
		elif args.load_checkpoint is True and epoch > 5 and train_rank > best_rank:
			print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
			best_rank = train_rank
			torch.save(ssl_model.state_dict(), os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
			print()
		elif epoch%10 == 0:
			print('saving the epoch of {}'.format(epoch))
			torch.save(ssl_model.state_dict(), os.path.join(args.RESULS_SAVE_PATH, "model_{}.pt".format(epoch)))
			print()

	# load the trained wsi_embedder model
	ssl_model = load_checkpoint(args, ssl_model)
		
	# get slide-level embeddings of downstream datasets
	# extract_slide_level_embeddings(args, val_dataloaders, ssl_model)