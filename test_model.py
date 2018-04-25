# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# IMPORTANT:: Select one of them ->  GoogLeNet,  AlexNet , CaffeNet , VGG19

selected_algorithm = 'GoogLeNet'

test_image_name = "fire.5428.jpg"

# load json and create model
json_file = open('./quick_models/' + selected_algorithm + '/model/' + selected_algorithm + '_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("./trained_models/fire" + selected_algorithm + "Model_weights.h5")
 
# evaluate loaded model on test data
classifier.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 3 - Making new predictions

test_image = image.load_img('./test_images/' + test_image_name, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'not fire'
else:
    prediction = 'fire'