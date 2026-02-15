import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def resize_image(image_path, output_path, size=(512, 512)):
    """处理单张图片的函数"""
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize(size, Image.LANCZOS)  # LANCZOS 是更优质的插值方法
            resized_img.save(output_path)
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")


def resize_images_in_directory(directory, size=(512, 512), output_directory=None, num_workers=4):
    if output_directory is None:
        output_directory = directory  # 默认覆盖原图

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 支持的文件扩展名
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    # 获取所有支持的文件路径
    image_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(supported_extensions)
    ]

    print(f"Found {len(image_files)} images in directory {directory}.")

    # 多进程并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename in image_files:
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)
            futures.append(executor.submit(resize_image, image_path, output_path, size))

        # 等待任务完成
        for step, future in enumerate(futures):
            if step % 100 == 0:
                print(f"Processed {step}/{len(image_files)} images.")


# /data/ceiling/workspace/HCC/CUT/datasets/HE2mIF_1024/CD19_val/

resize_images_in_directory(
    '/data/ceiling/workspace/HCC/CUT/datasets/HE2mIF_1024/Foxp3_train/',
    size=(512, 512),
    num_workers=8  # 设置并行进程数量
)