import argparse
import os
from pathlib import Path

from PIL import Image
from skimage import io

from u2net import U2NetSalientObjectDetector

MODEL_DIR = './u2net/saved_models/u2net/u2net.pth'


def main():
    args = parse_args()

    image_path = args.image
    output_dir = args.output

    u2_salient_object_detector = U2NetSalientObjectDetector(MODEL_DIR)

    saliency_map = u2_salient_object_detector.detect(image_path)

    save_output(image_path, saliency_map, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Salient Object Detector Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


def save_output(input_image_path, saliency_map, output_dir):
    predict = saliency_map
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(input_image_path)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_filename = Path(input_image_path).stem
    imo.save(f'{output_dir}/{image_filename}_sal.png')


if __name__ == "__main__":
    main()
