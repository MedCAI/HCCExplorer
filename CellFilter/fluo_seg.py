from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 单张图像分割函数
def nuclei_segment(img_path, save_path, model):
    img = imageio.imread(img_path)
    labels, _ = model.predict_instances(normalize(img))
    mask = 255 * np.uint8(labels > 0)
    imageio.imwrite(save_path, mask)

# 主程序
model = StarDist2D.from_pretrained('2D_versatile_fluo')
dapi_root = '/data/ceiling/workspace/HCC/filtered_patches/DAPI/'
save_root = './output_fluo_nuclei/'
slide_list = [
    '201550810.4_Scan1',
    '201581382.3_Scan1',
    '201541831.3_Scan1',
    '201519509.3_Scan2',
    '201583414.3_Scan1',
    '201403500.6_Scan1',
    '201461850-4_Scan1',
    '201500319.2_Scan1',
    '201625704.4_Scan1',
    '201473864-5_Scan1',
    '201615614.4_Scan1',
    '201418604.2_Scan1',
    '201560526.5_Scan1',
    '201433764.3_Scan2',
    '201472230-9_Scan1',
    '201404940.2_Scan1',
    '201419249.5_Scan1',
    '201547450.3_Scan1',
    '201429858.3_Scan1',
]

# 多线程处理
max_workers = 4  # 线程数，可以根据CPU核心数调整

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for slide_name in slide_list:
        print(slide_name)
        dapi_path = os.path.join(dapi_root, slide_name)
        save_path = os.path.join(save_root, slide_name)
        os.makedirs(save_path, exist_ok=True)

        for image_name in os.listdir(dapi_path):
            if image_name.startswith('.'):
                continue
            image_ = os.path.join(dapi_path, image_name)
            save_ = os.path.join(save_path, image_name)

            # 提交到线程池
            futures.append(executor.submit(nuclei_segment, image_, save_, model))

    # 等待所有任务完成
    for future in as_completed(futures):
        try:
            future.result()  # 可以加个try防止有异常挂掉整个池子
        except Exception as e:
            print(f"Error processing image: {e}")

print("All done.")