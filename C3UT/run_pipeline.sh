#!/bin/bash

# 1️⃣ 用户定义路径和名称
wsi_root="/data/ceiling/workspace/HCC/Registration/HE/"       # 替换为你的 WSI 根目录路径
h5_root="/data/jsh/ZJU/data_processed/Registration/HE_patch1024_step1024/patches"         # 替换为你的 H5 根目录路径
save_root="/data/ceiling/workspace/HCC/CUT/datasets/Generated_HE2mIF_1024_CycleGAN"
names=(
"201638887.3.ome"
"201638904.1.ome"
"201647678.4.ome"
"201547450.3.ome"
"201419249.5.ome"
"201472230-9.ome"
"201560526.5.ome"
"201473864-5.ome"
"201404940.2.ome"
"201433764.3.ome"
"201429858.3.ome"
"201418604.2.ome"
"201615614.4.ome"
) # 替换为你的 WSI 文件名列表，不带扩展名
save_names=(
"201638887.3.ome"
"201638904.1.ome"
"201647678.4.ome"
"201547450.3.ome"
"201419249.5.ome"
"201472230-9.ome"
"201560526.5.ome"
"201473864-5.ome"
"201404940.2.ome"
"201433764.3.ome"
"201429858.3.ome"
"201418604.2.ome"
"201615614.4.ome"
) # 替换为保存的目标文件名列表，不带扩展名

# 检查 names 和 save_names 的长度是否一致
if [ ${#names[@]} -ne ${#save_names[@]} ]; then
    echo "❌ names 和 save_names 长度不一致，请检查输入！"
    exit 1
fi

# 2️⃣ 遍历所有 WSI 图像名称
for ((i = 0; i < ${#names[@]}; i++)); do

    name="${names[i]}"
    save_name="${save_names[i]}"
    
    echo "🚀 开始处理图像: $name ($((i+1))/${#names[@]})"

    h5_path="${h5_root}/${name}.h5"
    wsi_path="${wsi_root}/${name}.tiff"
    save_he_path="${save_root}/${name}/HE"
    save_path="${save_root}/${name}"

    # Step 2: 运行 he2patches.py
    echo "▶️ 运行 he2patches.py..."
    python he2patches.py \
        --h5_path "$h5_path" \
        --wsi_path "$wsi_path" \
        --save_path "$save_he_path" \
        --patch_size 1024 1024

    if [ $? -ne 0 ]; then
        echo "❌ he2patches.py 运行失败，跳过此图像！"
        continue
    fi
    echo "✅ 完成 he2patches.py 的运行."

    # Step 3: 启动 modified_test.py 的六个任务
    echo "▶️ 启动 modified_test.py 的六个任务..."
    
    # 第一批任务
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2CD3_1024 --model cycle_gan >s0_temp_HE2CD3.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2CD4_1024 --model cycle_gan >s0_temp_HE2CD4.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2CD8_1024 --model cycle_gan >s0_temp_HE2CD8.log 2>&1 &
    
    # 等待第一批任务完成
    wait
    echo "✅ 第一批任务已完成."
    
    # 第二批任务
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2CD19_1024 --model cycle_gan >s0_temp_HE2CD19.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2CD68_1024 --model cycle_gan >s0_temp_HE2CD68.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 nohup python -u modified_test.py --eval --num_test 1000000 --epoch 7 --results_dir "$save_path" --dataroot "$save_he_path" --dataset_mode aligned --name CycleGAN_HE2Foxp3_1024 --model cycle_gan >s0_temp_HE2Foxp3.log 2>&1 &
    
    # 等待第二批任务完成
    wait
    echo "✅ 第二批任务已完成."
done

echo "🎉 所有图像处理完成！"