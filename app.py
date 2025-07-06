from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Ganti sesuai label model Anda
LABELS = ["Fadel", "Jabra"]

app = Flask(__name__)
app.secret_key = 'absensi_cnn_secret'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model CNN .h5 (pastikan file model_absen.h5 ada di folder project)
MODEL_PATH = 'model.h5'
if os.path.exists(MODEL_PATH):
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
else:
    cnn_model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'photo' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['photo']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Di sini nanti akan dipanggil fungsi prediksi CNN
            result = predict_absen(filepath)
            return render_template('result.html', result=result, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_absen(image_path):
    if cnn_model is None:
        return 'Model CNN tidak ditemukan.'
    try:
        # Cek input shape model
        input_shape = cnn_model.input_shape
        # input_shape biasanya (None, H, W, C) atau (None, N)
        if len(input_shape) == 4:
            _, h, w, c = input_shape
            img = image.load_img(image_path, target_size=(h, w))
            img_array = image.img_to_array(img)
            if c == 1:
                # convert to grayscale
                img_array = np.mean(img_array, axis=2, keepdims=True)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
        elif len(input_shape) == 2:
            # Model expects flat vector, misal (None, 33856)
            n = input_shape[1]
            # Coba reshape gambar ke persegi, atau flatten dari 224x224x3
            img = image.load_img(image_path)
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            flat = img_array.flatten()
            if flat.shape[0] != n:
                return f"Error: Model expects input shape {n}, gambar setelah flatten {flat.shape[0]}. Ubah ukuran gambar atau model!" 
            img_array = np.expand_dims(flat, axis=0)
        else:
            return f"Error: Model input shape tidak dikenali: {input_shape}"
        pred = cnn_model.predict(img_array)
        pred_value = pred.tolist()
        # Output model: one-hot/probabilitas per kelas atau sigmoid 1 neuron
        if len(pred.shape) == 2 and pred.shape[1] == len(LABELS):
            idx = int(np.argmax(pred[0]))
            label = LABELS[idx] if idx < len(LABELS) else f"Unknown({idx})"
            confidence = float(np.max(pred[0]))
            return f'Nama: {label} | Confidence: {confidence:.2f} | pred: {pred_value}'
        elif len(pred.shape) == 2 and pred.shape[1] == 1 and len(LABELS) == 2:
            # Binary sigmoid output
            val = pred[0][0]
            label = LABELS[1] if val > 0.5 else LABELS[0]
            confidence = float(val) if val > 0.5 else 1-float(val)
            return f'Nama: {label} | Confidence: {confidence:.2f} | pred: {pred_value}'
        else:
            return f'Output model ({pred.shape[1]}) tidak sesuai jumlah label ({len(LABELS)}). pred: {pred_value}'
    except Exception as e:
        input_shape_info = getattr(cnn_model, "input_shape", "unknown")
        return f"Error prediksi: {str(e)} | Model input shape: {input_shape_info}"

if __name__ == '__main__':
    app.run(debug=True)
