import os
import imageio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def crop_center(image, crop_size=(1024, 1024)):
    h, w = image.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        raise ValueError(f"输入图像尺寸太小，无法裁剪 {crop_size} 的区域。")
    start_x = w // 2 - cw // 2
    start_y = h // 2 - ch // 2
    return image[start_y:start_y+ch, start_x:start_x+cw]

def process_image(slide_name, image_name, marker_path, marker, save_root):
    dapi_image = None
    sampleaf_image = None
    
    for step, m in enumerate(marker_path):
        m_type = marker[step]
        image_path = os.path.join(m, image_name)

        if not os.path.exists(image_path):
            print(f"图像不存在: {image_path}")
            continue

        image = imageio.imread(image_path)

        if m_type == 'HE':
            crop_image = crop_center(image[:, :, :3])
        elif m_type == 'DAPI':
            dapi_image = crop_center(image)
            crop_image = np.stack((dapi_image, dapi_image, dapi_image), axis=-1)
        elif m_type == 'SampleAF':
            sampleaf_image = crop_center(image)
            crop_image = np.stack((sampleaf_image, sampleaf_image, sampleaf_image), axis=-1)
        elif m_type == 'Nuclei':
            crop_image = crop_center(image)
        else:
            channel = crop_center(image)
            if dapi_image is None or sampleaf_image is None:
                print(f"DAPI/SampleAF 缺失: {slide_name} - {image_name}")
                return
            crop_image = np.stack((dapi_image, channel, sampleaf_image), axis=-1)

        save_path = os.path.join(save_root, m_type + '_train')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, slide_name + '-' + image_name)
        imageio.imwrite(save_name, crop_image)

def main():
    marker = ['HE', 'DAPI', 'SampleAF', 'CD3', 'CD4', 'CD8', 'CD19', 'CD68', 'Foxp3', 'Nuclei']
    filter_root = '/data/ceiling/workspace/HCC/filtered_patches/'
    save_root = '/data/ceiling/workspace/HCC/CUT/datasets/HE2mIF_1024/'
    DAPI_root = os.path.join(filter_root, 'DAPI')
    slide_list = ['201524599.4_Scan1',
                  '201413211.4_Scan1',
                  '201541831.3_Scan2',
                  '201576225.3_Scan1',
                  '201419275.3_Scan1',
                  '201558458.3_Scan1',
                  '201548796.4_Scan1',
                  '201647678.4_Scan1',
                  '201468671-2_Scan1',
                  '201626042.4_Scan1',
                  '201583373.3_Scan1',
                  '201536981.3_Scan1',
                  '201639980.5_Scan1',
                  '201433005.4_Scan1',
                  '201611809.2_Scan1',
                  '201638887.3_Scan1',
                  '201644395.5_Scan1',
                  '201575131.3_Scan1',
                  '201638904.1_Scan1',
                  '201427534.2_Scan1',
                  '201550810.4_Scan1',
                  '201581382.3_Scan1',
                  '201541831.3_Scan1',
                  '201519509.3_Scan2',
                  '201583414.3_Scan1',
                  '201403500.6_Scan1',
                  '201461850-4_Scan1',
                  '201500319.2_Scan1',
                  '201625704.4_Scan1',
                  # '201473864-5_Scan1',
                  # '201615614.4_Scan1',
                  # '201418604.2_Scan1',
                  # '201560526.5_Scan1',
                  # '201433764.3_Scan2',
                  # '201472230-9_Scan1',
                  # '201404940.2_Scan1',
                  # '201419249.5_Scan1',
                  # '201547450.3_Scan1',
                  # '201429858.3_Scan1'
                 ]
    tasks = []

    with ThreadPoolExecutor(max_workers=4) as executor:  # 你可以根据CPU核心数调整
        for slide_name in slide_list:
            if slide_name.startswith('.'):
                continue
            print(slide_name)
            marker_path = [os.path.join(filter_root, m, slide_name) for m in marker]
            image_list = os.listdir(marker_path[0])
            for image_name in image_list:
                if image_name.startswith('.'):
                    continue
                tasks.append(executor.submit(process_image, slide_name, image_name, marker_path, marker, save_root))

if __name__ == "__main__":
    main()