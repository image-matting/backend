import os
from pathlib import Path
from PIL import Image

from skimage import io

SUFFIX = "_saliency.png"


def save_saliency_output(saliency_image, filename, output_dir):
    predict = saliency_image
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(filename)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_filename = Path(filename).stem
    saliency_output_path = f'{output_dir}/{image_filename}{SUFFIX}'
    imo.save(saliency_output_path)

    return saliency_output_path
