import argparse

from saliency_output_utils import save_saliency_output
from u2net import U2NetSalientObjectDetector

MODEL_DIR = './u2net/saved_models/u2net/u2net.pth'


def main():
    args = parse_args()

    image_path = args.image
    output_dir = args.output

    u2_salient_object_detector = U2NetSalientObjectDetector(MODEL_DIR)

    saliency_image = u2_salient_object_detector.detect(image_path)

    save_saliency_output(saliency_image, image_path, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Salient Object Detector Application')
    parser.add_argument('-i', '--image', required=True, type=str, help='path to input image')
    parser.add_argument('-o', '--output', required=False, default='.', type=str, help='path to output directory')
    return parser.parse_args()


if __name__ == "__main__":
    main()
