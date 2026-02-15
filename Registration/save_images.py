import os
import numpy as np
import pandas as pd
import pathlib
import pickle
from valis import registration


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

# i=3报错
# i = -1
name = '201541831.3_Scan2'
registration.init_jvm()
registrar_f = "./Coarse_Registration/{}/{}/data/{}_registrar.pickle".format(name, name, name)
registered_slide_dst_dir = "./Coarse_Registration/{}/save_image/".format(name)
registrar = registration.load_registrar(registrar_f)
registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap", non_rigid=True, compression="jpeg")
registration.kill_jvm()