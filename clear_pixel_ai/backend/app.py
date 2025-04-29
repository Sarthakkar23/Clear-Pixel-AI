from flask import Flask, request, send_file, jsonify, render_template_string
from flask_cors import CORS
import os
from clear_pixel_ai.backend.processing import process_image_bytes, process_image
import io
from flask import send_file
from flask import send_file, make_response
import io

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = os.path.join("backend", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import logging
from flask import got_request_exception

logging.basicConfig(level=logging.DEBUG)

def log_exception(sender, exception, **extra):
    sender.logger.error(f"Exception: {exception}", exc_info=exception)

got_request_exception.connect(log_exception, app)

@app.route('/')
def home():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "static", "upload.html")
    with open(html_path, "r") as f:
        html = f.read()
    return render_template_string(html)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            print("No image part in request")
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        image_bytes = file.read()
        processed_bytes = process_image_bytes(image_bytes, operation='denoise')

        return send_file(io.BytesIO(processed_bytes.read()), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /upload: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        operation = request.form.get('operation')
        filter_type = request.form.get('filter_type')

        image_bytes = file.read()
        processed_bytes = process_image_bytes(image_bytes, operation=operation, filter_type=filter_type)

        return send_file(io.BytesIO(processed_bytes.read()), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /process: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/super_resolve', methods=['POST'])
def super_resolve():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        abs_upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
        file.save(abs_upload_path)

        output_path = process_image(abs_upload_path, operation='super_resolve')
        abs_output_path = os.path.abspath(output_path)

        return send_file(abs_output_path, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /super_resolve: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/inpaint', methods=['POST'])
def inpaint():
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Image and mask required'}), 400
        file = request.files['image']
        mask_file = request.files['mask']
        if file.filename == '' or mask_file.filename == '':
            return jsonify({'error': 'No selected file or mask'}), 400

        abs_upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
        abs_mask_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, mask_file.filename))
        file.save(abs_upload_path)
        mask_file.save(abs_mask_path)

        output_path = process_image(abs_upload_path, operation='inpaint', mask_path=abs_mask_path)
        abs_output_path = os.path.abspath(output_path)

        return send_file(abs_output_path, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /inpaint: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        operation = request.form.get('operation')
        filter_type = request.form.get('filter_type')

        abs_upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
        file.save(abs_upload_path)

        output_path = process_image(abs_upload_path, operation=operation, filter_type=filter_type)
        abs_output_path = os.path.abspath(output_path)

        return send_file(abs_output_path, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /process: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/super_resolve', methods=['POST'])
def super_resolve_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        abs_upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
        file.save(abs_upload_path)

        output_path = process_image(abs_upload_path, operation='super_resolve')
        abs_output_path = os.path.abspath(output_path)

        return send_file(abs_output_path, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /super_resolve: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/inpaint', methods=['POST'])
def inpaint_route():
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Image and mask required'}), 400
        file = request.files['image']
        mask_file = request.files['mask']
        if file.filename == '' or mask_file.filename == '':
            return jsonify({'error': 'No selected file or mask'}), 400

        abs_upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
        abs_mask_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, mask_file.filename))
        file.save(abs_upload_path)
        mask_file.save(abs_mask_path)

        output_path = process_image(abs_upload_path, operation='inpaint', mask_path=abs_mask_path)
        abs_output_path = os.path.abspath(output_path)

        return send_file(abs_output_path, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /inpaint: {e}", exc_info=e)
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
