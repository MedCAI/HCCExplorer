import time
import pdb

# numpy
import numpy as np
import os

# torch
import torch # type: ignore

# internal
from madeleine.utils.utils import set_model_precision, smooth_rank_measure

# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 
WHOLE_VIEW_POSITION = 0


# move to utils
def calculate_losses(STAINS, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, wsi_embs, token_embs, modality_labels_withoutHE, args):
	losses = []
	atleast_two_loss_flag = False

	# HE
	# torch.Size([3, 3, 512, 6])
	# torch.Size([3, 2048, 128, 6])

	# CD3
	# torch.Size([3, 3, 512])
	# torch.Size([3, 2048, 128])

	for stain_idx, stain in enumerate(STAINS):

		stain_mask = modality_labels_withoutHE[:, stain_idx].bool() # modality_labels_withoutHE 判断数据集中是否存在这一模态（应该是判断是否所有模态都存在）

		if stain_mask.sum().item() > 1:
			# Global loss:
			if loss_fn_interMod:
				HE_for_stain = wsi_embs["HE"][:, WHOLE_VIEW_POSITION, :, stain_idx][stain_mask] # ([3, 512]) views=3, view_num=0作为global view
				stain_ind = wsi_embs[stain][:, WHOLE_VIEW_POSITION, :][stain_mask]              # ([3, 512])

				if args.global_loss == "info-nce":
					global_loss = loss_fn_interMod(query=HE_for_stain, positive_key=stain_ind, symmetric=args.symmetric_cl) # symmetric参数表示双向
				else:
					raise AssertionError("invalid global loss")

				# add to loss 
				losses.append(global_loss) 

			# Local loss:
			if loss_fn_interMod_local:
				HE_tokens = token_embs["HE"][:, :, :, stain_idx][stain_mask]    # ([3, 2048, 128])
				IHC_tokens = token_embs[stain].squeeze()[stain_mask]            # ([3, 2048, 128])
				got_loss = loss_fn_interMod_local(HE_tokens, IHC_tokens, subsample=256)
				got_loss = got_loss * args.local_loss_weight

				# add to loss 
				losses.append(got_loss)

			# Intra modality loss
			if loss_fn_intraMod:    # 1 and 2都是局部视角
				# view 1 
				HE_for_stain_view1 = wsi_embs["HE"][:, 1, :, stain_idx][stain_mask]
				stain_ind_view1 = wsi_embs[stain][:, 1, :][stain_mask]

				# view 2
				HE_for_stain_view2 = wsi_embs["HE"][:, 2, :, stain_idx][stain_mask]
				stain_ind_view2 = wsi_embs[stain][:, 2, :][stain_mask]

				# intra modal loss for HE and stain
				l_HE = loss_fn_intraMod(query=HE_for_stain_view1, positive_key=HE_for_stain_view2, symmetric=args.symmetric_cl)
				l_stain = loss_fn_intraMod(query=stain_ind_view1, positive_key=stain_ind_view2, symmetric=args.symmetric_cl)

				# add to loss 
				losses.append(l_HE)
				losses.append(l_stain)

			# there is at least one stain in addition to HE in this batch, so we keep this batch
			atleast_two_loss_flag = True
			
	if len(losses) > 0:
		loss = sum(losses)
	else:
		loss = -1
		assert loss == -1 and not atleast_two_loss_flag, "Loss should be -1 if there are no losses to calculate"
		
	return loss, atleast_two_loss_flag

# move to utils
def train_loop(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):

	if loss_fn_intraMod:
		n_views = 3
	else:
		n_views = 1
		
	ssl_model.train()
	torch_precision = set_model_precision(args.precision)

	ep_loss = 0.
	fb_time = 0.
	all_embeds = []
	
	for b_idx, data in enumerate(dataloader):
		# [b, 7, 2048, 512]; [b, 7]
		if epoch == 0 and b_idx == 0:
			print("Using precision:", torch_precision)
		
		s_fb = time.time()
		
		# clean modality labels to be without HE
		modality_labels = data['modality_labels']
		modality_labels_withoutHE = modality_labels[:, HE_POSITION+1:]
		
		# begin forward pass
		optimizer.zero_grad()
			 
		with torch.amp.autocast(device_type="cuda", dtype=torch_precision):
			# get model outputs
			wsi_embs, token_embs = ssl_model(data, device=DEVICE, n_views=n_views)
			# calculate losses
			loss, atleast_two_loss_flag = calculate_losses(args.STAINS, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, wsi_embs, token_embs, modality_labels_withoutHE, args)
			
		# get the train embeds to calculate rank
		all_embeds.extend(wsi_embs['HE'][:, WHOLE_VIEW_POSITION, :, 0].detach().to(torch.float32).cpu().numpy())
		
		# if we have a batch with only HE then continue
		if not atleast_two_loss_flag:
			print("Skipping batch with only HE")
			continue
		
		# if we have made it, then we must have had more than HE stain, so we update model
		loss.backward()
		optimizer.step()

		if epoch <= args.warmup_epochs:
			scheduler_warmup.step()
		else:
			scheduler.step()  
			
		if (b_idx % 3) == 0:
			print(str(time.time()))
			print(f"Loss for batch: {b_idx} = {loss:.3f}")
			
		ep_loss += loss.item()
		
		e_fb = time.time()
		fb_time += e_fb - s_fb
		
	# track rank on all HE slides
	all_embeds_tensor = torch.Tensor(np.array(all_embeds))
	rank = smooth_rank_measure(all_embeds_tensor)   # 判断特诊提取器的rank，zhi越大，那么特征提取能力越强
													# 判定过拟合
	return ep_loss, rank                   


def valid(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):
	# 
	if loss_fn_intraMod:
		n_views = 3
	else:
		n_views = 1
		  
	ssl_model.eval()
	torch_precision = set_model_precision(args.precision)

	ep_loss = 0.
	fb_time = 0.
	all_embeds = []
	ERROR_LOG_FILE = "/data2/jsh/MADELEINE-main/results/error_slide_ids.txt"
	LOG_FILE = "/data2/jsh/MADELEINE-main/results/slide_id_log.txt"

	# for b_idx, data in enumerate(reversed(list(dataloader))):
	for b_idx, data in enumerate(dataloader):
		slide_id = data['slide_ids']
		with open(LOG_FILE, "a") as f:
			for sid in slide_id:
				f.write(f"{sid}\n")

		save_path = os.path.join('/data2/jsh/MADELEINE-main/results/pt_files', str(slide_id[0])+'.pt')
		if os.path.exists(save_path):
			print(str(save_path)+' existing..')
			continue
		if slide_id[0] == '201266646_HE_null' or slide_id[0] == '201214704_HE_null' or slide_id[0]=='201427540_HE_6' or slide_id[0]=='201226380_HE_null':
			print('skipping '+str(slide_id))	# out of m
			continue
		if slide_id[0] == '201129220_HE_5':	# error
			print('skipping '+str(slide_id))
			continue
		if data["feats"].numel() == 0:
			print('skipping len=0 '+str(slide_id))
			with open(ERROR_LOG_FILE, "a") as f:
				for sid in slide_id:
					f.write(f"{sid}\n")
			continue

		# [b, 7, 2048, 512]; [b, 7]
		if epoch == 0 and b_idx == 0:
			print("Using precision:", torch_precision)
		
		s_fb = time.time()
		
		# clean modality labels to be without HE
		modality_labels = data['modality_labels']
		modality_labels_withoutHE = modality_labels[:, HE_POSITION+1:]
		
		# begin forward pass
		# optimizer.zero_grad()

		### choice1
		# wsi_embs, token_embs, slide_id = ssl_model(data, train=False, device=DEVICE, n_views=1, inference_patch_embedding=True)
		### choice2
		try:
			with torch.amp.autocast(device_type="cuda", dtype=torch_precision):
				# get model outputs
				with torch.no_grad():
					wsi_embs, token_embs, slide_id = ssl_model(data, train=False, device=DEVICE, n_views=1, inference_patch_embedding=True)
					# calculate losses
					# loss, atleast_two_loss_flag = calculate_losses(args.STAINS, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, wsi_embs, token_embs, modality_labels_withoutHE, args)
			torch.cuda.empty_cache()	
		except RuntimeError as e:
			if "out of memory" in str(e):
				print(f"⚠️ CUDA OOM at batch {b_idx}, skipping...")
				torch.cuda.empty_cache()
				with open(ERROR_LOG_FILE, "a") as f:
					for sid in slide_id:
						f.write(f"{sid}\tOOM\n")
			continue
	
		# get the train embeds to calculate rank
		# all_embeds.extend(wsi_embs['HE'][:, WHOLE_VIEW_POSITION, :, 0].detach().to(torch.float32).cpu().numpy())
		
		# if we have a batch with only HE then continue
		# if not atleast_two_loss_flag:
		# 	print("Skipping batch with only HE")
		# 	continue
		
		## if we have made it, then we must have had more than HE stain, so we update model
		# loss.backward()
		# optimizer.step()

		# if epoch <= args.warmup_epochs:
		#     scheduler_warmup.step()
		# else:
		#     scheduler.step()  
			
		# if (b_idx % 3) == 0:
		#     print(f"Loss for batch: {b_idx} = {loss:.3f}")
			
		# ep_loss += loss	#.item()
		
		e_fb = time.time()
		fb_time += e_fb - s_fb
		
		features = token_embs
		torch.save(features, save_path)
		# break
	# track rank on all HE slides
	# all_embeds_tensor = torch.Tensor(np.array(all_embeds))
	# rank = smooth_rank_measure(all_embeds_tensor)   # 判断特诊提取器的rank，zhi越大，那么特征提取能力越强
	# 												# 判定过拟合
	return ep_loss, rank     
