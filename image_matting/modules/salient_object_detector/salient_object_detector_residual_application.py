import argparse
import os
from pathlib import Path

from PIL import Image
from skimage import io
import numpy as np

from saliency_output_utils import save_saliency_output
from spectral_residual_saliency_detection.saliency import get_saliency_map

MODEL_DIR = './u2net/saved_models/u2net/u2net.pth'


def main():
    args = parse_args()

    image_path = args.image
    output_dir = args.output

    saliency_map = get_saliency_map(image_path)
    mask = saliency_map > np.mean(saliency_map) * 3
    saliency_map[mask] = 1
    saliency_map[~mask] = 0

    save_saliency_output(image_path, saliency_map, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Salient Object Detector Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


if __name__ == "__main__":
    main()
