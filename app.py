import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file:
        filename = file.filename
        filepath = os.path.join('static', filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)

        if prediction[0][0] > 0.5:
            result = "Healthy"
        else:
            result = "Diseased"

        return render_template(
            'index.html',
            prediction=result,
            img_path=filename   
        )

if __name__ == "__main__":
    app.run(debug=True)