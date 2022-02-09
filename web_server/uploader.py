import os
import uuid
from pathlib import Path

from flask import Flask, request, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

UPLOAD_DIR = './uploads'

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = uuid.uuid4().hex


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        unique_dir = f'{UPLOAD_DIR}/{uuid.uuid4().hex}'
        if not os.path.exists(unique_dir):
            os.makedirs(unique_dir)

        if 'input_image' in request.files:
            file = request.files['input_image']
            save_file(file, unique_dir, "input_image")

        if 'background_image' in request.files:
            file = request.files['background_image']
            save_file(file, unique_dir, "background_image")
    return ""


def save_file(file, directory, filename):
    file_extension = Path(secure_filename(file.filename)).suffix
    filename_to_save = f'{filename}.{file_extension}'
    file.save(f'{directory}/{filename_to_save}')


if __name__ == '__main__':
    app.run(debug=True)
