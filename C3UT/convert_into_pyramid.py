import tifffile
import numpy as np
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom


def create_ome_metadata(size_x, size_y, size_c, size_z, size_t, channel_names):
    """
    创建符合 OME XML 格式的 ImageDescription 元数据
    """
    # 创建根元素 <OME>
    ome = Element('OME', {
        'xmlns': 'http://www.openmicroscopy.org/Schemas/OME/2016-06',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd',
    })

    # 创建 <Image> 元素
    image = SubElement(ome, 'Image', {'ID': 'Image:0', 'Name': 'Example'})

    # 创建 <Pixels> 元素
    pixels = SubElement(image, 'Pixels', {
        'DimensionOrder': 'XYZCT',
        'Type': 'uint8',
        'BigEndian': 'false',
        'SizeX': str(size_x),
        'SizeY': str(size_y),
        'SizeZ': str(size_z),
        'SizeC': str(size_c),
        'SizeT': str(size_t),
    })

    # 为每个通道创建 <Channel> 元素
    for i, channel_name in enumerate(channel_names):
        SubElement(pixels, 'Channel', {
            'ID': f'Channel:0:{i}',
            'Name': channel_name,
            'SamplesPerPixel': '1',
        })

    # 格式化 XML 输出
    xml_str = xml.dom.minidom.parseString(tostring(ome)).toprettyxml(indent="    ")
    return xml_str


def save_pyramid_tiff_for_qupath(input_tiff_path, output_tiff_path, channel_names, tile_size=512):
    """
    转换 TIFF 为 QuPath 兼容的金字塔格式，写入 OME 元数据
    """
    # 打开原始 OME-TIFF 图像
    with tifffile.TiffFile(input_tiff_path) as tif:
        # 获取图像维度信息
        image_data = tif.asarray()

    # 检查维度顺序
    if image_data.ndim == 3:  # (size_y, size_x, size_c) 或 (size_c, size_y, size_x)
        if image_data.shape[-1] == len(channel_names):  # 最后一维是通道数
            # 输入格式为 (size_y, size_x, size_c)，需要调整为 (size_c, size_y, size_x)
            size_y, size_x, size_c = image_data.shape
            image_data = np.transpose(image_data, (2, 0, 1))  # 转换为 (size_c, size_y, size_x)
        elif image_data.shape[0] == len(channel_names):  # 第一维是通道数
            # 输入格式为 (size_c, size_y, size_x)，不需要调整
            size_c, size_y, size_x = image_data.shape
        else:
            raise ValueError("输入图像的维度顺序无法识别，请检查输入文件")
    else:
        raise ValueError("输入图像必须是 3 维数组")

    dtype = image_data.dtype
    size_z = 1  # 假设 Z 轴深度为 1
    size_t = 1  # 假设时间点为 1

    # 创建 OME 元数据
    ome_metadata = create_ome_metadata(size_x, size_y, size_c, size_z, size_t, channel_names)

    # 创建金字塔层级
    pyramid_shapes = []
    current_shape = (size_c, size_y, size_x)
    while min(current_shape[1:]) > 256:  # 持续缩小，直到最小边小于 256 像素
        new_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)
        pyramid_shapes.append(new_shape)
        current_shape = new_shape

    # 写入金字塔 TIFF 文件
    with tifffile.TiffWriter(output_tiff_path, bigtiff=True) as tiff_writer:
        for level, shape in enumerate([(size_c, size_y, size_x)] + pyramid_shapes):
            print(f"处理金字塔层级 {level}, 大小: {shape}")
            level_data = np.zeros(shape, dtype=dtype)  # 创建当前层级的空数组

            # 分块处理
            for y in range(0, size_y, tile_size):
                for x in range(0, size_x, tile_size):
                    # 计算块大小
                    y_end = min(y + tile_size, size_y)
                    x_end = min(x + tile_size, size_x)
                    block_data = image_data[:, y:y_end, x:x_end]  # 手动切块

                    # 如果不是第 0 层，则缩放图像块
                    if level > 0:
                        block_data = np.stack([
                            np.array(Image.fromarray(block_data[c]).resize(
                                (block_data.shape[2] // 2, block_data.shape[1] // 2),
                                Image.Resampling.BICUBIC  # 使用更高质量的缩放
                            ))
                            for c in range(size_c)
                        ])

                    # 将块数据写入当前层级
                    level_data[:, y // (2 ** level):y_end // (2 ** level),
                               x // (2 ** level):x_end // (2 ** level)] = block_data

            # 写入当前层级
            tiff_writer.write(level_data,
                              shape=level_data.shape,
                              dtype=level_data.dtype,
                              subfiletype=1 if level > 0 else 0,  # 子文件类型（0为主文件，1为子文件）
                              description=ome_metadata if level == 0 else None,  # 仅第 0 层添加元数据
                              photometric='rgb' if size_c == 3 else 'minisblack',  # 如果是 RGB 图像，设置为 RGB
                              compression='jpeg')  # 使用 JPEG 压缩以确保 QuPath 兼容

    print(f"QuPath 兼容的金字塔 TIFF 已保存到: {output_tiff_path}")


if __name__ == "__main__":
    # 提供输入和输出 TIFF 文件路径
    input_tiff_path = "/data/ceiling/workspace/HCC/CUT/convert_images/temp_save/201429858.3.ome.tiff"  # 替换为你的输入文件路径
    output_tiff_path = "/data/ceiling/workspace/HCC/CUT/convert_images/temp_save/pyramid_201429858.3.ome.tiff"  # 替换为你的输出文件路径

    # 通道名称列表（根据实际情况调整）
    channel_names = [
        "DAPI",
        "Foxp3",
        "CD19",
        "CD68",
        "CD4",
        "CD3",
        "CD8",
        "SampleAF"
    ]

    # 转换为 QuPath 兼容的金字塔 TIFF 并保存
    save_pyramid_tiff_for_qupath(input_tiff_path, output_tiff_path, channel_names, tile_size=512)