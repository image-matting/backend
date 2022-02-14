import os
import uuid
from pathlib import Path

from flask import Flask, request, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from alpha_mask_output_utils import save_alpha_mask_output
from deep_image_matting import AlphaMaskGenerator
from saliency_output_utils import save_saliency_output
from trimap import generate_trimap
from trimap_output_utils import save_trimap_output
from u2net import U2NetSalientObjectDetector

UPLOAD_DIR = './uploads'

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = uuid.uuid4().hex

INPUT_IMAGE_NAME = "input_image"
BACKGROUND_IMAGE_NAME = "background_image"

SALIENCY_MODEL_DIR = '../image_matting/modules/salient_object_detector/u2net/saved_models/u2net/u2net.pth'
ALPHA_MODEL_DIR = '../image_matting/modules/alpha_mask_generator/deep_image_matting/model/stage1_skip_sad_52.9.pth'

saliency_object_detector = U2NetSalientObjectDetector(SALIENCY_MODEL_DIR)
alpha_mask_generator = AlphaMaskGenerator(ALPHA_MODEL_DIR)


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        unique_dir = f'{UPLOAD_DIR}/{uuid.uuid4().hex}'
        if not os.path.exists(unique_dir):
            os.makedirs(unique_dir)

        input_image_file = request.files[INPUT_IMAGE_NAME]
        saved_input_image_path = save_file(input_image_file, unique_dir, INPUT_IMAGE_NAME)

        background_image_file = request.files[BACKGROUND_IMAGE_NAME]
        saved_background_image_path = save_file(background_image_file, unique_dir, BACKGROUND_IMAGE_NAME)

        saliency_image = saliency_object_detector.detect(saved_input_image_path)
        saliency_image_path = save_saliency_output(saliency_image, saved_input_image_path, unique_dir)

        trimap_image = generate_trimap(saliency_image_path)
        trimap_image_path = save_trimap_output(trimap_image, INPUT_IMAGE_NAME, unique_dir)

        alpha_mask_image = alpha_mask_generator.generate_alpha_mask(saved_input_image_path, trimap_image_path)
        alpha_image_path = save_alpha_mask_output(alpha_mask_image, INPUT_IMAGE_NAME, unique_dir)

    return ""


def save_file(file, directory, filename):
    file_extension = Path(secure_filename(file.filename)).suffix
    filename_to_save = f'{filename}{file_extension}'
    filepath = f'{directory}/{filename_to_save}'
    file.save(f'{directory}/{filename_to_save}')

    return filepath


if __name__ == '__main__':
    app.run(debug=True)
