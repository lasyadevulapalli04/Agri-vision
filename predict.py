import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model.h5')

img_name = sys.argv[1]

img_path = os.path.join("dataset", "test", img_name)

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array, verbose=0)

print("Prediction value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction: Healthy")
else:
    print("Prediction: Diseased")