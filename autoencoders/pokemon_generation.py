import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (layers, models, callbacks, utils, metrics, losses, optimizers)

from scipy.stats import norm
import pandas as pd



Batch_Size = 32
Image_Size = 256
Seed = 433524
Verbose = False
Directory = "pokemon"
Num_Features = 50
Embedding_Dim = 500
Beta = 4500
Learning_Rate = 0.0005
Load_Model = False
Epochs = 10



class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="end_train_total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="end_train_reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="end_train_kl_loss")

        self.val_total_loss_tracker = metrics.Mean(name="end_val_total_loss")
        self.val_reconstruction_loss_tracker = metrics.Mean(name="end_val_reconstruction_loss")
        self.val_kl_loss_tracker = metrics.Mean(name="end_val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @property
    def val_metrics(self):
        return[
        self.val_total_loss_tracker,
        self.val_reconstruction_loss_tracker,
        self.val_kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data, training=True)
            reconstruction_loss = tf.reduce_mean( Beta * losses.mean_squared_error(data, reconstruction))
            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return { "train_total_loss": self.total_loss_tracker.result(), "train_reconstruction_loss": self.reconstruction_loss_tracker.result(),"train_kl_loss": self.kl_loss_tracker.result()}


    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(Beta * losses.mean_squared_error(data, reconstruction))
        kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return { "val_total_loss": self.val_total_loss_tracker.result(), "val_reconstruction_loss": self.val_reconstruction_loss_tracker.result(),"val_kl_loss": self.val_kl_loss_tracker.result()}

class PokemonGen(callbacks.Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated = self.model.decoder(random_latent_vectors)
        Display(generated,True,f'./pokemon_output/generated_img_{epoch}.png')


class SaveModel(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"./models/pokemon_diff_vae_{epoch}.keras")
        self.model.evaluate(validation_dataset)

def Image_Preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


def Display(images, save, name="generic_name", labels=None):
    num_images = len(images)
    rows = min(num_images, 8)  # Ensure maximum of 8 rows
    cols = (num_images + rows - 1) // rows  # Calculate number of columns

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), gridspec_kw={'hspace': 0.5, 'wspace': 0})

    for ax in axes.flat:
        ax.axis('off')

    for i, image in enumerate(images):
        if i < rows * cols:
            ax = axes[i // cols, i % cols]
            ax.imshow(image)
            ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0.5)

    if save:
        plt.savefig(name)
    else:
        plt.show()


# --------------------------------------------------------------------------------------------------------------


dataset = utils.image_dataset_from_directory(
    directory= "../datasets/pokemon",
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=Batch_Size,
    image_size=(Image_Size, Image_Size),
    shuffle=True,
    seed=Seed,
    validation_split=0.20,
    subset="both",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=False
)

train_dataset = dataset[0]
validation_dataset = dataset[1]

train_no_labels_dataset = train_dataset.map(lambda x,y: Image_Preprocess(x))
validation_no_labels_dataset = validation_dataset.map(lambda x,y: Image_Preprocess(x))


if Verbose:
    test_feature = next(iter(train_dataset.take(1)))
    imgs = test_feature[0].numpy()
    labels = test_feature[1].numpy()
    #Display(imgs,False,labels=labels)




# Encoder
encoder_input = layers.Input(shape=(Image_Size, Image_Size, 3), name="encoder_input")


x = layers.Conv2D(Num_Features, kernel_size=4, strides=2, padding="same")(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

Num_Features *= 2
x = layers.Conv2D(Num_Features, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

Num_Features *= 2
x = layers.Conv2D(Num_Features, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

Num_Features
x = layers.Conv2D(Num_Features, kernel_size=2, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
z_mean = layers.Dense(Embedding_Dim, name="z_mean")(x)
z_log_var = layers.Dense(Embedding_Dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

if Verbose:
    encoder.summary()



# Decoder
decoder_input = layers.Input(shape=(Embedding_Dim,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Reshape(shape_before_flattening)(x)

x = layers.Conv2DTranspose(Num_Features, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

Num_Features // 2
x = layers.Conv2DTranspose(Num_Features, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

Num_Features // 2
x = layers.Conv2DTranspose(Num_Features, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)


x = layers.Conv2DTranspose(Num_Features, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

decoder_output = layers.Conv2DTranspose(3, kernel_size=3, strides=1, activation="sigmoid", padding="same")(x)
decoder = models.Model(decoder_input, decoder_output)

if Verbose:
    decoder.summary()


vae = VAE(encoder, decoder)


optimizer = optimizers.Adam(learning_rate=Learning_Rate)
vae.compile(optimizer=optimizer)

if Load_Model:
    vae.load_weights("./models/pokemon_vae_0.keras")
else:
    vae.fit(
    train_no_labels_dataset,
    epochs=Epochs,
    callbacks=[
        PokemonGen(num_img=36, latent_dim=Embedding_Dim),
        SaveModel(),
    ],
    )

