# general
import time

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
from utils.trainer import train_loop, valid
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

    print(f"\nTraining for epoch {0}...\n")
    
    # train
    start = time.time()
    # ep_loss, train_rank = train_loop(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler)
    ssl_model = load_checkpoint(args, ssl_model, path_to_checkpoint='results\model.pt')
    ep_loss, train_rank = valid(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, 0, dataloader, optimizer, scheduler_warmup, scheduler)
        
    if args.log_ml and not WANDB_ERROR:
        wandb.log({"train_loss": ep_loss, "train_rank": train_rank})
        
    log_data = {"epoch": 0, "train_loss": ep_loss, "train_rank": train_rank}
    logging.info(json.dumps(log_data))
    end = time.time()
    # print(f"\nDone with epoch {epoch}")
    print(f"Total loss = {ep_loss:.3f}")
    print(f"Train rank = {train_rank:.3f}")
    print(f"Total time = {end-start:.3f} seconds")
    print("\nDone with training\n")
