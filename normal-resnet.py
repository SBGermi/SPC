import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from absl import app
from absl import flags
import h5py
import numpy as np
import tensorflow.compat.v2 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import random
from skimage.util import random_noise

IMAGE_SHAPE = [224, 224, 3]
NUM_CLASSES = 4

flags.DEFINE_float('learning_rate', default = 0.001, help = 'Learning rate')
flags.DEFINE_integer('num_epochs', default = 100, help = 'Number of epochs')
flags.DEFINE_integer('batch_size', default = 32, help = 'Batch size')
FLAGS = flags.FLAGS

def motion_blur(image):
  mask = np.zeros((7, 7))
  mask[3, :] = np.ones(7)
  mask = mask / 7
  out = cv2.filter2D(image, -1, mask)

  return out

def frostedglass_blur(image):
  mask_size = 4
  H, W, C = image.shape
  out = np.zeros((H, W, C))
  for i in range(0, H - 4):
    for j in range(0, W - 4):
      rand_index = int(random.random() * 4)
      out[i, j] = image[i + rand_index, j + rand_index]

  return out

def gaussian_blur(image):
  out = cv2.GaussianBlur(image, (7, 7), 0)

  return out

def gaussian_noise(image):
  out = random_noise(image, mode = 'gaussian', var = 0.1**2)
  out = (255 * out).astype(np.float64)

  return out

def sp_noise(image):
  out = random_noise(image, mode = 's&p')
  out = (255 * out).astype(np.float64)

  return out

def gamma_lower(image):
  table = np.array([((i / 255.0) ** 4) * 255 for i in np.arange(0, 256)]).astype("uint8")
  out = (cv2.LUT((255 * image).astype(np.uint8), table)).astype(np.float) / 255

  return out

def gamma_higher(image):
  table = np.array([((i / 255.0) ** 0.25) * 255 for i in np.arange(0, 256)]).astype("uint8")
  out = (cv2.LUT((255 * image).astype(np.uint8), table)).astype(np.float) / 255
  return out

def occlusion(image):
  mask = np.zeros((224, 224, 3), dtype = np.uint8)
  mask = cv2.circle(mask, (112, 112), 75, (255, 255, 255), -1)
  black = np.zeros((224, 224, 3), dtype = np.uint8)
  out = np.where(mask == np.array([255, 255, 255]), black, image)

  return out

def resnet():
  image = tf.keras.layers.Input(shape = IMAGE_SHAPE, dtype = 'float32')
  x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(image)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides = 2, padding = 'same')(x)
  x = basic_block(x, 64, 1)
  x = basic_block(x, 64, 1)
  x = basic_block(x, 128, 2)
  x = basic_block(x, 128, 1)
  x = basic_block(x, 256, 2)
  x = basic_block(x, 256, 1)
  x = basic_block(x, 512, 2)
  x = basic_block(x, 512, 1)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.nn.softmax)(x)

  model = tf.keras.Model(inputs = image, outputs = x)
  optimizer = tf.keras.optimizers.Adam(lr = FLAGS.learning_rate)
  model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

  return model

def basic_block(input_tensor, filters, strides = 1):
  x = tf.keras.layers.Conv2D(filters, 3, strides, padding = 'same')(input_tensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Conv2D(filters, 3, strides = 1, padding = 'same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  if strides != 1:
    shortcut = tf.keras.layers.Conv2D(filters, 1, strides = strides)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
  else:
    shortcut = input_tensor
  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)

  return x

def main(argv):
  del argv

  tf.io.gfile.makedirs('Results/ResNet_Normal') 

  print('Creating the model ...')
  model = resnet()

  print('Loading the train data ...')
  Train_Data_Gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.20)
  Train_data = Train_Data_Gen.flow_from_directory('Data/CIFAR/train/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, subset = 'training') 
  Valid_data = Train_Data_Gen.flow_from_directory('Data/CIFAR/train/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, subset = 'validation') 
    
  print('Training the model ...')
  for epoch in range(FLAGS.num_epochs):
    print('Epoch {} out of {}'.format(epoch + 1, FLAGS.num_epochs))
    train = model.fit_generator(Train_data, steps_per_epoch = Train_data.samples // FLAGS.batch_size, validation_data = Valid_data, validation_steps = Valid_data.samples // FLAGS.batch_size - 1, epochs = 1)
    
  print('Saving the model ...')
  model.save_weights('Results/ResNet_Norm/weight.h5', save_format = 'h5')

  print('Loading the test data ...')
  Test_Org_Data_Gen = ImageDataGenerator(rescale = 1./255)
  Test_Org_data = Test_Org_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_motion_blur_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = motion_blur)
  Test_motion_blur_data = Test_motion_blur_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_frostedglass_blur_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = frostedglass_blur)
  Test_frostedglass_blur_data = Test_frostedglass_blur_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_gaussian_blur_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = gaussian_blur)
  Test_gaussian_blur_data = Test_gaussian_blur_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_gaussian_noise_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = gaussian_noise)
  Test_gaussian_noise_data = Test_gaussian_noise_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_sp_noise_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = sp_noise)
  Test_sp_noise_data = Test_sp_noise_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_gamma_lower_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = gamma_lower)
  Test_gamma_lower_data = Test_gamma_lower_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_gamma_higher_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = gamma_higher)
  Test_gamma_higher_data = Test_gamma_higher_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)
  Test_occlusion_Data_Gen = ImageDataGenerator(rescale = 1./255, preprocessing_function = occlusion)
  Test_occlusion_data = Test_occlusion_Data_Gen.flow_from_directory('Data/CIFAR/test/', target_size = (224, 224), class_mode = 'categorical', batch_size = FLAGS.batch_size, shuffle = False)

  print('Testing the model ...')
  print('Original data ...')
  test_1 = model.predict_generator(Test_Org_data, steps = Test_Org_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_org.csv', test_1, delimiter = ',')

  print('Motion blur data ...')
  test_2 = model.predict_generator(Test_motion_blur_data, steps = Test_motion_blur_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_mb.csv', test_2, delimiter = ',')

  print('Frosted glass blur data ...')
  test_3 = model.predict_generator(Test_frostedglass_blur_data, steps = Test_frostedglass_blur_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_feb.csv', test_3, delimiter = ',')
    
  print('Gaussian blur data ...')
  test_4 = model.predict_generator(Test_gaussian_blur_data, steps = Test_gaussian_blur_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_gb.csv', test_4, delimiter = ',')

  print('Gaussian noise data ...')
  test_5 = model.predict_generator(Test_gaussian_noise_data, steps = Test_gaussian_noise_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_gn.csv', test_5, delimiter = ',')

  print('Salt-and-Pepper noise data ...')
  test_6 = model.predict_generator(Test_sp_noise_data, steps = Test_sp_noise_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_spn.csv', test_6, delimiter = ',')

  print('Gamma lower data ...')
  test_7 = model.predict_generator(Test_gamma_lower_data, steps = Test_gamma_lower_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_gl.csv', test_7, delimiter = ',')

  print('Gamma higher data ...')
  test_8 = model.predict_generator(Test_gamma_higher_data, steps = Test_gamma_higher_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_gh.csv', test_8, delimiter = ',')

  print('Occlusion data ...')
  test_9 = model.predict_generator(Test_occlusion_data, steps = Test_occlusion_data.samples // FLAGS.batch_size, verbose = 1)
  np.savetxt('Results/ResNet_Norm/score_o.csv', test_9, delimiter = ',')

  np.savetxt('Results/ResNet_Norm/labels.csv', Test_Org_data.labels + 1, delimiter = ',')

if __name__ == '__main__':
  app.run(main)
