# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import model_from_json


# IMPORTANT:: Select one of them ->  GoogLeNet,  AlexNet , CaffeNet , VGG19

selected_algorithm = 'GoogLeNet'
selected_epoch_size = 25
selected_batch_size = 32
total_number_of_training_images = 30006 # total number of files in "dataset/training_set" folder
total_number_of_testing_images  = 11073 # total number of files in "dataset/test_set" folder


json_file = open('./quick_models/' + selected_algorithm + '/model/' + selected_algorithm + '_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# IMPORTANT::: whole algorithm's input image sizes are 224

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = selected_batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = selected_batch_size,
                                            class_mode = 'binary')

# IMPORTANT::: This is the original one
classifier.fit_generator(training_set,
                         steps_per_epoch = total_number_of_training_images,
                         epochs = selected_epoch_size,
                         validation_data = test_set,
                         validation_steps = total_number_of_testing_images)


#model_json = classifier.to_json()
#with open("fireGoogLeNetModel_json.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("./trained_models/fire" + selected_algorithm + "Model_weights.h5")
print("Saved model to disk in two different ways")
