import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K


def pre_process(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs,((0,0),(2,2),(2,2)), constant_values = 0.0) # 0 because that is the number of elements
    # np.expand_dims() is used to add a new axis 
    imgs = np.expand_dims(imgs,-1)
    return imgs



(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()

x_train = pre_process(x_train)
x_test = pre_process(x_test)




embedding_size = 10

encoder_input = layers.Input(shape=(32,32,1), name="encoder_input")
x = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2, activation='relu',padding='same')(encoder_input)
x = layers.Conv2D(filters=64,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2D(filters=128,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
shape_before_flatten = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(embedding_size,name='enocder_output')(x)

encoder = models.Model(encoder_input,encoder_output)


decoder_input = layers.Input(shape=(embedding_size,), name='decoder_input')
x = layers.Dense(np.prod(shape_before_flatten))(decoder_input)
x = layers.Reshape(shape_before_flatten)(x)
x = layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
decoder_output = layers.Conv2D(filters=1, kernel_size=(3,3), strides=1,activation='sigmoid', padding='same',name='decoder_output')(x)

decoder = models.Model(decoder_input,decoder_output)

autoencoder = models.Model(encoder_input, decoder(encoder_output)) # input_shape, output of final layer
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(x_train,x_train,epochs=2,batch_size=100,shuffle=True,validation_data=(x_test,x_test))

autoencoder.save('autoencoder_model.keras')



analysis_images = x_test[:5000]
random_indices = np.random.choice(len(analysis_images), 10, replace=False)

# prediction
predictions = autoencoder.predict(analysis_images)

random_samples = predictions[random_indices]
random_actual = analysis_images[random_indices]


plt.figure(figsize=(20, 4))

# Plot actual images
for i, image in enumerate(random_actual):
    plt.subplot(2, 10, i + 1)
    plt.xlabel('Actual')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

# Plot sampled images
for i, image in enumerate(random_samples):
    plt.subplot(2, 10, i + 11)  # Offset by 10 to start from the second row
    plt.xlabel('Sampled')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
