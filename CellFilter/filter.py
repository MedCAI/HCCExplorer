import os
import imageio
import shutil
import numpy as np

slide_list = ['201560526.5_Scan1',]


channels = ['DAPI', 'CD3', 'CD4', 'CD8', 'HE', 'CD19', 'CD68', 'Foxp3', 'SampleAF', 'Nuclei']
mask_root = '/data/ceiling/workspace/HCC/hover_net/output_tiles_Pannuke/'
org_root = '/data/ceiling/workspace/HCC/patches/'
save_root = '/data/ceiling/workspace/HCC/filtered_patches/'
DAPI_root = org_root + 'DAPI'

for slide_name in slide_list:
    print(slide_name)
    mask_path = os.path.join(mask_root, slide_name)
    DAPI_path = os.path.join(DAPI_root, slide_name)
    channels_path = [os.path.join(save_root, c, slide_name) for c in channels]
    for c in channels_path:
        os.makedirs(c, exist_ok=True)
    
    sum_threshold = 50
    max_threshold = 50
    count = 0
    for part_name in os.listdir(mask_path):
        if part_name[0] == '.':
            continue
        else:
            mask_part = os.path.join(mask_path, part_name, 'mask')
        for patch_name in os.listdir(mask_part):
            if patch_name[0] == '.':
                continue
            mask = imageio.imread(os.path.join(mask_part, patch_name))
            dapi = imageio.imread(os.path.join(DAPI_path, patch_name))
            if np.sum(mask / 255) < sum_threshold or np.max(dapi) < max_threshold:
                count += 1
                continue
            else:
                for step, c in enumerate(channels_path):
                    if 'Nuclei' in c:
                        shutil.copy(os.path.join(mask_part, patch_name), os.path.join(c, patch_name))
                    else:
                        shutil.copy(os.path.join(os.path.join(org_root, channels[step], slide_name, patch_name)), os.path.join(c, patch_name))
    print('filter:', count)