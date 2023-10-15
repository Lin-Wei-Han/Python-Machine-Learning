from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

# =================================================================
# Part 1 - Data Preprocessing
# =================================================================
train_data_generator = ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.2,
                                          zoom_range = 0.2,
                                          horizontal_flip = True)
traning_set = train_data_generator.flow_from_directory('data/training_set',
                                                       target_size = (64, 64),
                                                       batch_size = 32,
                                                       class_mode = 'binary')
test_data_generator = ImageDataGenerator(rescale = 1./255)
test_set = test_data_generator.flow_from_directory('data/test_set',
                                                       target_size = (64, 64),
                                                       batch_size = 32,
                                                       class_mode = 'binary')

# =================================================================
# Part 2 - Building the CNN
# =================================================================
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# =================================================================
# Part 3 - Training the CNN
# =================================================================
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = traning_set, validation_data = test_set, epochs = 25)

# =================================================================
# Part 4 - Making a single prediction
# =================================================================
test_image = image.load_img('data/single_prediction/dog.jpg',target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)
print(traning_set.class_indices)

print('result:{}'.format(result))

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)