from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model_path = r"C:/Users/yuvraj/OneDrive/Desktop/jupyter notebook/har_model.h5"
model = load_model(model_path)

class_labels = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating',
       'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
       'sitting', 'sleeping', 'texting', 'using_laptop'] 

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(126, 126)) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)
    if predicted_class[0] < len(class_labels) and predicted_class[0] >= 0:
        predicted_label = class_labels[predicted_class[0]]
        return predicted_label
    else:
        return "Unknown"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', file.filename)
        file.save(file_path)

        result = model_predict(file_path, model)
        return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)