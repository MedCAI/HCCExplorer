import pyvips
import argparse

def convert_multichannel_to_pyramidal_tiff(input_path, output_path):
    """
    将多通道 TIFF 图像转换为金字塔式 BigTIFF 格式。

    Args:
        input_path (str): 输入多通道 TIFF 文件路径。
        output_path (str): 输出金字塔式 TIFF 文件路径。
    """
    # Step 1: 加载图像
    image = pyvips.Image.new_from_file(input_path, access='sequential')
    
    print(f"🔍 图像信息: {image.width}x{image.height}, 通道数 (bands) = {image.bands}")
    if image.bands < 4:
        print("⚠️ 警告：图像通道数小于 4，可能是 RGB 图像")
    elif image.bands > 4:
        print("✅ 检测到多通道图像，准备保存为金字塔格式...")

    # Step 2: 保存为金字塔式 BigTIFF，使用无损压缩
    image.tiffsave(
        output_path,
        tile=True,
        pyramid=True,
        bigtiff=True,
        compression='deflate',  # lzw/deflate 支持多通道压缩
        tile_width=512,
        tile_height=512
    )

    print(f"✅ 已保存金字塔 TIFF：{output_path}")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert a multi-channel TIFF to pyramidal BigTIFF format.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input multi-channel TIFF file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output pyramidal TIFF file.")

    args = parser.parse_args()

    # 调用函数进行转换
    convert_multichannel_to_pyramidal_tiff(args.input_path, args.output_path)