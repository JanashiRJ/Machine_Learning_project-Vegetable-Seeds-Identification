
# imports the necessary libraries
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('D:/All Research Papers/Final Project WEBSITE/Implementation/Vegetable Seeds Identification/Model/Seeds Identify Model.h5')

# Mapping of class labels to seed names
CLASSES = [
    'Beans seeds',
    'Beet Seeds',
    'Bitter Gourd Seeds',
    'Brinjal seeds',
    'Cucumber seeds',
    'Pumpkin Seed',
    'Radish Seeds',
    'Tomato Seeds',
]


@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index', methods=['GET', 'POST'])
def ai_engine_page():
    
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']
    
    # Read the image file and resize it
    image = Image.open(io.BytesIO(image_file.read()))
    image = image.resize((256, 256))  # Resize the image to (256, 256)

    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0

    # Make predictions using the loaded model
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    
    # Get the corresponding seed name
    predicted_seed = CLASSES[predicted_class]

    return render_template('predict.html', predicted_seed=predicted_seed)

@app.route('/market')
def market():
    return render_template('market.html')


if __name__ == '__main__':
    app.run()
