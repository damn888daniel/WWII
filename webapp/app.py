import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import joblib
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Создать папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загрузка модели при старте
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'clip_fake_detector.pkl'
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
clip_model = None
clip_processor = None
classifier = None


def load_models():
    global clip_model, clip_processor, classifier
    print("Loading models...")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=dtype).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    classifier = joblib.load(MODEL_PATH)
    clip_model.eval()
    print("Models loaded successfully!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path, caption):
    """Предсказание для изображения"""
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    inputs = clip_processor(
        text=[caption],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        img_emb = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        txt_emb = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)
        combo = torch.cat([img_emb, txt_emb], dim=-1)

    embedding = combo.cpu().numpy()
    prob = float(classifier.predict_proba(embedding)[0][1])
    pred = int(prob >= 0.5)

    return {
        'prediction': 'synthetic' if pred == 1 else 'real',
        'prob_fake': prob,
        'prob_real': 1 - prob,
        'confidence': max(prob, 1 - prob) * 100
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    caption = request.form.get('caption', '')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

    if not caption:
        caption = "WWII historical photograph"

    # Сохранение файла
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Предсказание
        result = predict_image(filepath, caption)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Удаление файла после обработки
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5001)
