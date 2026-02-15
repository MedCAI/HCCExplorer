# 文件夹处理：
from valis import registration, preprocessing, feature_detectors, affine_optimizer
import pyvips
import time
import os
import numpy as np

root = './Original/'
save_root = './Coarse_Registration/'
for file_name in os.listdir(root):
    print(file_name)
    slide_src_dir = os.path.join(root, file_name)
    results_dst_dir = os.path.join(save_root, file_name)
    os.makedirs(results_dst_dir, exist_ok=True)
    registered_slide_dst_dir = "./slide_registration_example_coarse/registered_slides"
    for image_name in os.listdir(slide_src_dir):
        if 'svs' in image_name:
            # 以he为参考图像
            reference_slide = image_name
        else:
            continue
    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, results_dst_dir, 
                                   reference_img_f=reference_slide, 
                                   non_rigid_registrar_cls=None)
    
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    #　registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference")
    # Kill the JVM
registration.kill_jvm()