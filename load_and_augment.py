import tensorflow.keras as k
from preprocess import preprocess
"""This script loads datasets and augments them using keras imagedatagenerator class, and returns their objects as output
   it takes 2 arguments which are paths of the training and testing data 
"""
def load_and_augment_data(path,tpath):
   image_gen=k.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7,1.2],
    shear_range=0.0,
    zoom_range=[0.4,1.6],
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=preprocess_input,
    data_format=None,
    validation_split=0.2,
    dtype=None,
    )
    image_gen_testing=k.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    brightness_range=[1,1],
    shear_range=0.0,
    zoom_range=0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=preprocess_input,
    data_format=None,
    validation_split=None,
    )

    training_gen=image_gen.flow_from_directory(path,batch_size=32,target_size=(224,224),subset='training')
    validation_gen=image_gen.flow_from_directory(path,batch_size=32,target_size=(224,224),subset='validation')
    testing_gen=image_gen_testing.flow_from_directory(tpath,batch_size=32,target_size=(224,224))

    return training_gen,validation_gen,testing_gen
  	