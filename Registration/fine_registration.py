from valis import registration, feature_detectors, non_rigid_registrars, affine_optimizer
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration
import pyvips
import time
import os
import numpy as np

name_list = ['201524599.4_Scan1', '201413211.4_Scan1', '201467173-2_Scan1', '201541831.3_Scan2', '201576225.3_Scan1',
             '201419275.3_Scan1', '201558458.3_Scan1', '201548796.4_Scan1', '201647678.4_Scan1', '201468671-2_Scan1',
             '201626042.4_Scan1', '201583373.3_Scan1', '201624571.4_Scan1',
             '201536981.3_Scan1',  '201639980.5_Scan1', '201433005.4_Scan1', '201611809.2_Scan1',
             '201638887.3_Scan1', '201644395.5_Scan1', '201575131.3_Scan1', '201638904.1_Scan1', '201427534.2_Scan1',
             '201550810.4_Scan1', '201581382.3_Scan1', '201541831.3_Scan1', '201519509.3_Scan2', '201583414.3_Scan1',
             '201541796.4_Scan1', '201403500.6_Scan1', '201461850-4_Scan1', '201500319.2_Scan1',
             '201625704.4_Scan1', '201473864-5_Scan1', '201615614.4_Scan1', '201418604.2_Scan1', '201560526.5_Scan1',
             '201433764.3_Scan2', '201472230-9_Scan1', '201404940.2_Scan1', '201419249.5_Scan1', '201547450.3_Scan1',
             '201429858.3_Scan1',]

i = 1
name = name_list[i]

slide_src_dir = "./Original/" + name
results_dst_dir = "./Fine_Registration/" + name
os.makedirs(results_dst_dir, exist_ok=True)
registered_slide_dst_dir = results_dst_dir + "/registered_slides/"
micro_reg_fraction = 0.25

# Create a Valis object and use it to register the slides in slide_src_dir
# reference_img_f=reference_slide

start = time.time()
feature_detector_cls = feature_detectors.KazeFD
non_rigid_registrar_cls = non_rigid_registrars.SimpleElastixWarper
affine_optimizer_cls = affine_optimizer.AffineOptimizerMattesMI

registrar = registration.Valis(slide_src_dir, results_dst_dir,
                               feature_detector_cls=feature_detector_cls,
                               affine_optimizer_cls=affine_optimizer_cls,
                               non_rigid_registrar_cls=non_rigid_registrar_cls)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
'''
registrar = registration.Valis(slide_src_dir, results_dst_dir,
                               micro_rigid_registrar_cls=MicroRigidRegistrar)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
'''
# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

# Perform high resolution non-rigid registration using 25% full resolution
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)
# registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference")
registration.kill_jvm()