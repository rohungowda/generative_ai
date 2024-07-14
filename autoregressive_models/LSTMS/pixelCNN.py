import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras import (datasets, layers, models, callbacks, utils, metrics, losses, optimizers)


IMAGE_SIZE = 32
PIXEL_LEVELS = 8
RESIDUAL_BLOCKS = 3
FILTERS = 128

def preprocess(imgs_int):
    imgs_int = tf.image.resize(imgs_int, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    imgs_int = (imgs_int / (256 / PIXEL_LEVELS)).astype(int) # sets all the pixwl values between 0 - PIXEL_LEVELS [labels]
    imgs = imgs_int.astype("float32")
    imgs = imgs / PIXEL_LEVELS # scaled between 0 and 1 [input to llm]
    return imgs, imgs_int


class MaskedConvLayer(layers.Layer):
    def __init__(self, mask_type,**kwargs):
        super(MaskedConvLayer,self).__init__()
        self.mask_type = mask_type
        self.conv= layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)

        kernel_shape = self.conv.kernel.shape

        kernel_x_size, kernel_y_size, input_filters, output_filters = kernel_shape

        self.mask = np.zeros(kernel_shape)
        self.mask = np.reshape(self.mask, (output_filters,input_filters,kernel_x_size, kernel_y_size))
        mask_width, mask_height = kernel_x_size // 2, kernel_y_size // 2

        # make sure mask is correct size (not exactly sure what is mask in this scenario)
        self.mask[:,:,:mask_width,:] = 1.0
        self.mask[:,:,mask_width,:mask_height] = 1.0

        def bmask(i_out, i_in):
            cout_idx = np.expand_dims(np.arange(output_filters) % 3 == i_out, 1)
            cin_idx = np.expand_dims(np.arange(input_filters) % 3 == i_in, 0)
            a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            return a1 * a2


        # what color is outputted and based on what is inputted R: 0, G : 1, B : 2
        self.mask[bmask(1, 0), mask_width, mask_height] = 1.0
        self.mask[bmask(2, 0), mask_width, mask_height] = 1.0
        self.mask[bmask(2, 1), mask_width, mask_height] = 1.0

        if self.mask_type == "B":
            for i in range(3):
                self.mask[bmask(i, i), mask_width, mask_height] = 1.0

        self.mask = np.reshape(self.mask, (kernel_x_size, kernel_y_size, input_filters, output_filters))


    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

class ResidualBlocks(layers.Layer):
    def __init__(self, filters):
        super(ResidualBlocks, self).__init__()
        self.conv1 = layers.Conv2D(filters // 2, kernel_size=1, padding='valid', activation='relu')
        self.masked_conv = MaskedConvLayer(mask_type='B',filters=(filters//2),kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, kernel_size=1, padding='valid', activation='relu')

    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.masked_conv(x)
        x = self.conv2(x)
        return layers.add([inputs,x])


def create_PixelCNN():
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = MaskedConvLayer(
        mask_type="A",
        filters=FILTERS,
        kernel_size=7,
        activation="relu",
        padding="same",
    )(inputs)

    for _ in range(RESIDUAL_BLOCKS):
        x = ResidualBlocks(filters=FILTERS)(x)

    for _ in range(2):
        x = MaskedConvLayer(
            mask_type="B",
            filters=FILTERS,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)



    out = layers.Conv2D(
        filters=PIXEL_LEVELS * 3,
        kernel_size=1,
        strides=1,
        activation="softmax",
        padding="valid",
    )(x)

    out = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3, PIXEL_LEVELS))(out)

    pixel_cnn = models.Model(inputs, out)
    return pixel_cnn

def Display(images, save=False, name="generic_name"):

    num_images = len(images)
    rows = min(num_images, 8)  # Ensure maximum of 8 rows
    cols = (num_images + rows - 1) // rows  # Calculate number of columns

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    for ax in axes.flat:
        ax.axis('off')

    for i, image in enumerate(images):
        if i < rows * cols:
            ax = axes[i // cols, i % cols]
            ax.imshow(image)
            ax.axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.25)

    if save:
        plt.savefig(name)
    else:
        plt.show()

    plt.close(fig)

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def sample_from(self, probs, temperature):  # <2>
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        output = np.empty((3), dtype=int)

        for c in range(3): 
            output[c] = np.random.choice(PIXEL_LEVELS, p=probs[c])
        
        return output

    def generate(self, temperature):
        generated_images = np.zeros(
            shape=(self.num_img,) + (pixel_cnn.input_shape)[1:]
        )
        batch, rows, cols, channels = generated_images.shape

        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    probs = self.model.predict(generated_images, verbose=0)[
                        :, row, col, :
                    ]
                    generated_images[:, row, col, channel] = [
                        self.sample_from(x, temperature)[channel] for x in probs
                    ]
                    generated_images[:, row, col, channel] /= PIXEL_LEVELS

        return generated_images

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate(temperature=1.0)
        Display(generated_images,save=True,name=f"{epoch}_image")




pixel_cnn = create_PixelCNN()
pixel_cnn.summary()

img_generator_callback = ImageGenerator(num_img=10)



(x_train, _), (_, _) = datasets.cifar10.load_data()

input_data, output_data = preprocess(x_train)

print(input_data.shape)
print(output_data.shape)





# Assuming img_tensor is your 32x32x3 TensorFlow array
# Convert to numpy array if necessary



Display(input_data[:36], save=True)



adam = optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss="sparse_categorical_crossentropy") # given softmax values check if it falls under categorical values like [0,1,2,3,4,5,6...]

pixel_cnn.fit(
    input_data,
    output_data,
    batch_size=128,
    epochs=150,
    callbacks=[img_generator_callback],
)