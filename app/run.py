from flask import Flask, render_template, request, redirect, url_for

from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

import glob
import numpy as np
import tensorflow as tf
import os

from extract_bottleneck_features import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
Xception_model = None
ResNet50_model = None
graph = None

dog_names = ["Affenpinscher",
    "Afghan hound",
    "Airedale terrier",
    "Akita",
    "Alaskan malamute",
    "American eskimo dog",
    "American foxhound",
    "American staffordshire terrier",
    "American water spaniel",
    "Anatolian shepherd dog",
    "Australian cattle dog",
    "Australian shepherd",
    "Australian terrier",
    "Basenji",
    "Basset hound",
    "Beagle",
    "Bearded collie",
    "Beauceron",
    "Bedlington terrier",
    "Belgian malinois",
    "Belgian sheepdog",
    "Belgian tervuren",
    "Bernese mountain dog",
    "Bichon frise",
    "Black and tan coonhound",
    "Black russian terrier",
    "Bloodhound",
    "Bluetick coonhound",
    "Border collie",
    "Border terrier",
    "Borzoi",
    "Boston terrier",
    "Bouvier des flandres",
    "Boxer",
    "Boykin spaniel",
    "Briard",
    "Brittany",
    "Brussels griffon",
    "Bull terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn terrier",
    "Canaan dog",
    "Cane corso",
    "Cardigan welsh corgi",
    "Cavalier king charles spaniel",
    "Chesapeake bay retriever",
    "Chihuahua",
    "Chinese crested",
    "Chinese shar-pei",
    "Chow chow",
    "Clumber spaniel",
    "Cocker spaniel",
    "Collie",
    "Curly-coated retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie dinmont terrier",
    "Doberman pinscher",
    "Dogue de bordeaux",
    "English cocker spaniel",
    "English setter",
    "English springer spaniel",
    "English toy spaniel",
    "Entlebucher mountain dog",
    "Field spaniel",
    "Finnish spitz",
    "Flat-coated retriever",
    "French bulldog",
    "German pinscher",
    "German shepherd dog",
    "German shorthaired pointer",
    "German wirehaired pointer",
    "Giant schnauzer",
    "Glen of imaal terrier",
    "Golden retriever",
    "Gordon setter",
    "Great dane",
    "Great pyrenees",
    "Greater swiss mountain dog",
    "Greyhound",
    "Havanese",
    "Ibizan hound",
    "Icelandic sheepdog",
    "Irish red and white setter",
    "Irish setter",
    "Irish terrier",
    "Irish water spaniel",
    "Irish wolfhound",
    "Italian greyhound",
    "Japanese chin",
    "Keeshond",
    "Kerry blue terrier",
    "Komondor",
    "Kuvasz",
    "Labrador retriever",
    "Lakeland terrier",
    "Leonberger",
    "Lhasa apso",
    "Lowchen",
    "Maltese",
    "Manchester terrier",
    "Mastiff",
    "Miniature schnauzer",
    "Neapolitan mastiff",
    "Newfoundland",
    "Norfolk terrier",
    "Norwegian buhund",
    "Norwegian elkhound",
    "Norwegian lundehund",
    "Norwich terrier",
    "Nova scotia duck tolling retriever",
    "Old english sheepdog",
    "Otterhound",
    "Papillon",
    "Parson russell terrier",
    "Pekingese",
    "Pembroke welsh corgi",
    "Petit basset griffon vendeen",
    "Pharaoh hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese water dog",
    "Saint bernard",
    "Silky terrier",
    "Smooth fox terrier",
    "Tibetan mastiff",
    "Welsh springer spaniel",
    "Wirehaired pointing griffon",
    "Xoloitzcuintli",
    "Yorkshire terrier"]

def init():
    global Xception_model
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.load_weights('../saved_models/weights.best.Xception.hdf5')
    global ResNet50_model
    ResNet50_model = ResNet50(weights='imagenet')
    global graph
    graph = tf.get_default_graph()

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def Xception_predict_breed(img_path):
    with graph.as_default():
        bottleneck_feature = extract_Xception(path_to_tensor(img_path))
        predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def dog_detector(img_path):
    with graph.as_default():
        img = preprocess_input(path_to_tensor(img_path))
        prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'jpg'

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            breed = "Sorry, it doesn't seem link you uploaded a dog image"
            if dog_detector(img_path):
                breed = "It is predicted to be: " + Xception_predict_breed(img_path)
            return render_template('index.html', breed=breed, img_path=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=3001, debug=True)