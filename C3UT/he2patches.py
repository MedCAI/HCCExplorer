import openslide
import os
from PIL import Image
import h5py
import argparse

def convert_he_to_patches(wsi_path, save_path, coords, size=(512, 512)):
    os.makedirs(save_path, exist_ok=True)
    slide = openslide.OpenSlide(wsi_path)

    patch_width, patch_height = size
    wsi_width, wsi_height = slide.dimensions
    
    for step, (x, y) in enumerate(coords):
        if step % 1000 == 0:
            print(step)
        if x + patch_width > wsi_width or y + patch_height > wsi_height:
            print(f"Skipping patch at ({x}, {y}) — out of bounds.")
            continue
        patch = slide.read_region((int(x), int(y)), 0, size).convert("RGB")
        patch = patch.resize((512, 512), Image.LANCZOS)  # or Image.LANCZOS for better quality
        filename = f"{int(x)}_{int(y)}.png"
        patch.save(os.path.join(save_path, filename))
    print(f"Saved {len(coords)} patches to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract patches from WSI based on coordinates in an HDF5 file.")
    parser.add_argument('--h5_path', type=str, required=True, help='Path to the .h5 file containing coordinates.')
    parser.add_argument('--wsi_path', type=str, required=True, help='Path to the WSI (.svs, .tiff, etc.) file.')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[1024, 1024], help='Patch size as two integers: width height')
    parser.add_argument('--save_path', type=str, required=True, help='Patch to save patches')
    args = parser.parse_args()
    save_path = args.save_path

    with h5py.File(args.h5_path, 'r') as file:
        coords = file['coords'][:]

    convert_he_to_patches(args.wsi_path, save_path, coords, size=tuple(args.patch_size))

if __name__ == "__main__":
    main()