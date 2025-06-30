from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('poultry_model.h5')
classes = ['Coccidiosis', 'Salmonella', 'Healthy']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            pred_class = classes[np.argmax(preds[0])]
            confidence = round(np.max(preds[0]) * 100, 2)

            prediction = f"{pred_class} ({confidence}%)"
            image_url = filepath

    return render_template('index.html', prediction=prediction, image_url=image_url)
