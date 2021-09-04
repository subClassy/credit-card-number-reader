import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (32, 32, 3)
img_width = 32
img_height = 32
num_classes = 10
nb_train_samples = 20000
nb_validation_samples = 4000
batch_size = 16
epochs = 10

train_data_dir = './credit_card/train'
validation_data_dir = './credit_card/test'

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 10,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip = False,
    fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = False)   

