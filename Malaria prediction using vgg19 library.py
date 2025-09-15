# link for dataset-'https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria'
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]
# Folder 'cell_images' must be in same directory otherwise have to give full location of the folder
train_path = 'cell_images/Train'
valid_path = 'cell_images/Test'

# add preprocessing layer to the front of VGG
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False


folders = glob('cell_images/Train/*')

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# structure of model
model.summary()

# setting cost and optimization method
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('cell_images/Train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('cell_images/Test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# fitting the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
# Save the model
model.save('malaria_prediction_model.h5')
