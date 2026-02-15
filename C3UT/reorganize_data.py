import os
import shutil
from collections import defaultdict

# 设置原始数据路径和目标输出路径
root_dir = '/data/ceiling/workspace/HCC/CUT/datasets/HE2mIF_1024'
output_root = '/data/ceiling/workspace/HCC/CUT/datasets/reorganized_val'

image_names = ['201647678.4', 
               '201638904.1', 
               '201638887.3']
# 创建输出根目录
os.makedirs(output_root, exist_ok=True)

# 获取所有 *_val 文件夹
val_folders = [f for f in os.listdir(root_dir) if f.endswith('_train') and os.path.isdir(os.path.join(root_dir, f))]

# 遍历每个 marker_val 文件夹
for marker_folder in val_folders:
    marker_path = os.path.join(root_dir, marker_folder)
    for filename in os.listdir(marker_path):
        if not filename.endswith('.png'):
            continue
        try:
            prefix = filename.split('_')[0]  # 提取前缀，比如 '201473864-5'
            if prefix in image_names:
                target_dir = os.path.join(output_root, prefix, marker_folder)
                os.makedirs(target_dir, exist_ok=True)
                # 拷贝图像
                src = os.path.join(marker_path, filename)
                dst = os.path.join(target_dir, filename)
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")

print("✅ 图像重新组织完成！")