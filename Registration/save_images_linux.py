import os
import numpy as np
import pandas as pd
import pathlib
import pickle
from valis import registration
import sys

def main(i):
    print(f"Running with i = {i}")
    name_list = ['201541831.3_Scan2',
                 '201541831.3_Scan1',
                 '201625704.4_Scan1',
                 '201429858.3_Scan1']
    name = name_list[i]
    registration.init_jvm()
    registrar_f = "./Coarse_Registration/{}/{}/data/{}_registrar.pickle".format(name, name, name)
    registered_slide_dst_dir = "./Coarse_Registration/{}/save_image/".format(name)
    registrar = registration.load_registrar(registrar_f)
    registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap", non_rigid=True, compression="jpeg")
    registration.kill_jvm()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_program.py <i>")
        sys.exit(1)
    i = int(sys.argv[1])  # 从命令行读取 i
    main(i)