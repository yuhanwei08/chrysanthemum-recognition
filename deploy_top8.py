"""Local Flask application for the eight-class EfficientNet-B3 checkpoint."""
import base64
import binascii
import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSES = ['乒乓菊', '勋章菊', '墨菊', '波斯菊', '金丝皇菊', '金鸡菊', '雏菊', '黑眼菊']


def create_model(num_classes=8):
    model = models.efficientnet_b3(weights=None)
    features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(features, 512), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256, num_classes),
    )
    return model


def load_predictor(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {path}. See README.md.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint.get('class_names', DEFAULT_CLASSES)
    if len(classes) != 8 or len(set(classes)) != 8:
        raise ValueError('The checkpoint must contain eight distinct class labels.')
    model = create_model(len(classes))
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device).eval()
    recorded_accuracy = checkpoint.get('best_val_acc')
    if recorded_accuracy is not None:
        recorded_accuracy = float(recorded_accuracy)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def predict_image(image):
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            scores, indices = torch.softmax(model(tensor), dim=1).topk(3)
        return [{'class': classes[int(index)], 'confidence': round(float(score), 4)}
                for score, index in zip(scores[0], indices[0])]

    return predict_image, list(classes), recorded_accuracy


def create_app(predictor=None, classes=None, validation_accuracy=None):
    """An optional predictor supports API checks without loading model weights."""
    app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))
    app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
    if predictor is None:
        predictor, classes, validation_accuracy = load_predictor(
            os.environ.get('MODEL_PATH', str(BASE_DIR / 'best_top8.pth')))
    if classes is None:
        classes = list(DEFAULT_CLASSES)

    @app.get('/')
    def index():
        return render_template('index_top8.html')

    @app.get('/api/info')
    def api_info():
        return jsonify(name='Chrysanthemum Recognition System', model='EfficientNet-B3',
                       classes=classes, num_classes=len(classes),
                       validation_accuracy=validation_accuracy)

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='Upload exceeds the 12 MB request limit.'), 413

    @app.post('/api/predict')
    def predict():
        try:
            if 'image' in request.files:
                image_bytes = request.files['image'].read()
            elif request.is_json:
                payload = request.get_json(silent=True) or {}
                encoded = payload.get('image')
                if not isinstance(encoded, str) or not encoded:
                    return jsonify(error='Please upload an image.'), 400
                image_bytes = base64.b64decode(encoded.split(',', 1)[-1], validate=True)
            else:
                return jsonify(error='Please upload an image.'), 400
            with Image.open(io.BytesIO(image_bytes)) as uploaded:
                image = uploaded.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, binascii.Error, Image.DecompressionBombError):
            return jsonify(error='The uploaded file could not be read as an image.'), 400
        try:
            results = predictor(image)
            return jsonify(success=True, predictions=results, top1=results[0])
        except Exception:
            app.logger.exception('Image prediction failed')
            return jsonify(error='Prediction failed. Please check the server logs.'), 500

    return app


if __name__ == '__main__':
    application = create_app()
    print('Open http://127.0.0.1:5000 in your browser.')
    application.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
