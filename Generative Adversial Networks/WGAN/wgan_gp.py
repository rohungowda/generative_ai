import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (layers, models, callbacks, utils, metrics, losses, optimizers)

IMAGE_SIZE = 64
CHANNELS = 3
EMBEDDING_DIM = 350
BATCH_SIZE = 64
EPOCH = 100




def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

class Generator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, EMBEDDING_DIM))
        generated = self.model.generator(random_latent_vectors)
        generated = tf.cast((tf.cast(generated,'float32') * 127.5) + 127.5, tf.uint8).numpy()
        if not (epoch % 10):
            Display(generated,True,f'./generated_output_4_1/generated_img_1_{epoch}.png')

class SaveModel(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if not (epoch % 10):
            self.model.generator.save(f"./generated_models_4_1/gen_1_{epoch}.keras")
            self.model.critic.save(f"./generated_models_4_1/disc_1_{epoch}.keras")

class GraphLosses(callbacks.Callback):
    def __init__(self):
        self.CRITIC_LOSSES = []
        self.GENERATOR_LOSSES = []
    def on_epoch_end(self, epoch, logs=None):
        self.GENERATOR_LOSSES.append(self.model.metrics[0].result().numpy())
        self.CRITIC_LOSSES.append(self.model.metrics[1].result().numpy())

    def on_train_end(self, logs=None):
        x = list(range(1,len(self.GENERATOR_LOSSES) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(x, self.GENERATOR_LOSSES, label='Generator_loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Generator Losses')
        axes[0].legend()
        axes[0].grid(True)

        # Plot the second loss on the second subplot
        axes[1].plot(x, self.CRITIC_LOSSES, label='Critic_loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Critic Losses')
        axes[1].legend()
        axes[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig("Losses_Plot")

        # Display the plots
        plt.show()
        plt.clf()
        plt.close(fig)


def Display(images, save=False, name="generic_name"):

    num_images = len(images)
    rows = min(num_images, 8)  # Ensure maximum of 8 rows
    cols = (num_images + rows - 1) // rows  # Calculate number of columns

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for ax in axes.flat:
        ax.axis('off')

    for i, image in enumerate(images):
        if i < rows * cols:
            ax = axes[i // cols, i % cols]
            ax.imshow(image)
            ax.axis('off')

    #plt.subplots_adjust(wspace=0, hspace=0.5)

    if save:
        plt.savefig(name)
    else:
        plt.show()

    plt.close(fig)




def create_critic(filters=[32,64,128,256]):

    critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    x = layers.Conv2D(filters[0], kernel_size=4, strides=2, padding="same")( critic_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(filters[1], kernel_size=4, strides=2, padding="same")( critic_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(filters[1], kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(filters[2], kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(filters[3], kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(1, kernel_size=4, strides=2, padding="valid")(x)
    critic_output = layers.Flatten()(x)

    critic = models.Model(critic_input, critic_output)

    return critic


def create_generator(filters=[512, 256, 128, 64]):

    generator_input = layers.Input(shape=(EMBEDDING_DIM,))
    x = layers.Reshape((1, 1, EMBEDDING_DIM))(generator_input)

    x = layers.Conv2DTranspose(filters[0], kernel_size=4, strides=2, padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.25)(x)

    x = layers.Conv2DTranspose(filters[1], kernel_size=4, strides=2, padding="same", use_bias=False )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.25)(x)

    x = layers.Conv2DTranspose(filters[2], kernel_size=4, strides=2, padding="same", use_bias=False )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.25)(x)

    x = layers.Conv2DTranspose(filters[3], kernel_size=3, strides=2, padding="same", use_bias=False )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.25)(x)


    generator_output = layers.Conv2DTranspose(CHANNELS, kernel_size=2, strides=2, padding="same", activation="tanh",)(x)
    generator = models.Model(generator_input, generator_output)
    
    return generator


def critic_loss_fn(real_predictions, generated_predictions):
    critic_real_loss = tf.reduce_mean(real_predictions)
    critic_generated_loss = tf.reduce_mean(generated_predictions)
    return critic_generated_loss - critic_real_loss



def generator_loss_fn(generated_predictions):
    return -tf.reduce_mean(generated_predictions)

class WGAN_GP(models.Model):
    def __init__(self, critic, generator, number_critic_steps=3, gp_coefficient=10.0):
        super(WGAN_GP,self).__init__()
        self.critic = critic
        self.generator = generator
        self.gp_coefficient = gp_coefficient
        self.number_critic_steps = number_critic_steps
    
    def compile(self, critic_optimizer, generator_optimizer, critic_loss_fn, generator_loss_fn):
        super(WGAN_GP,self).compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.critc_loss_fn = critic_loss_fn
        self.generator_loss_fn = generator_loss_fn
        self.generator_loss_metric = metrics.Mean(name="generator_loss")
        self.critic_loss_metric = metrics.Mean(name="critic_loss")

    @property
    def metrics(self):
        return [self.generator_loss_metric, self.critic_loss_metric]

    def gradient_penalty(self, batch_size, real_data, generated_images):
        epsilon =  tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0) # comes from a uniform distribution not normal distribution
        diff = generated_images - real_data
        interpolated_images = real_data + epsilon * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)

            interpolated_loss = self.critic(interpolated_images, training=True)

        interpolated_gradients = gp_tape.gradient(interpolated_loss, [interpolated_images])[0] # with respect to interpolated_images

        norm = tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp


    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        gen_data = tf.random.normal(shape=(batch_size, EMBEDDING_DIM))

        for t in range(self.number_critic_steps):
            with tf.GradientTape() as critic_tape:


                generated_images = self.generator(gen_data, training=True)


                real_predictions = self.critic(real_images, training=True)
                gen_predictions = self.critic(generated_images, training=True)

                critic_loss = self.critc_loss_fn(real_predictions, gen_predictions)
                gp = self.gradient_penalty(batch_size, real_images, generated_images)

                total_critic_loss = critic_loss + gp * self.gp_coefficient

            critic_gradients = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients,self.critic.trainable_variables))

        
        generated_data = tf.random.normal(shape=(batch_size, EMBEDDING_DIM)) 
      
        with tf.GradientTape() as gen_tape:

            generated_images = self.generator(generated_data, training=True)

            generated_predictions = self.critic(generated_images, training=True)
            total_gen_loss = self.generator_loss_fn(generated_predictions)

        generator_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))

        self.generator_loss_metric.update_state(total_gen_loss)
        self.critic_loss_metric.update_state(total_critic_loss)

        return {m.name: m.result() for m in self.metrics}

critic = create_critic()

generator = create_generator()


train_data = utils.image_dataset_from_directory(
    "../../datasets/artwork/artwork",
    labels=None,
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)

verbose = False

train = train_data.map(lambda x: preprocess(x))

if verbose:
    test_feature = next(iter(train.take(1)))
    imgs = tf.cast((tf.cast(test_feature,'float32') * 127.5) + 127.5, tf.uint8).numpy()
    Display(imgs)



wgan= None

LOAD_MODEL = True

if LOAD_MODEL:
    print("loading model")
    generator.load_weights("./generated_models_4_1/gen_110.keras")
    critic.load_weights("./generated_models_4_1/disc_110.keras")
    wgan = WGAN_GP(critic, generator)
else:
    print("creating new model")
    wgan = WGAN_GP(critic, generator)
    # original paper used RMSProp
generator_optimizer = optimizers.Adam( learning_rate=0.00007, beta_1=0, beta_2=0.9)
critic_optimizer = optimizers.Adam( learning_rate=0.00007, beta_1=0, beta_2=0.9)

wgan.compile(critic_optimizer,generator_optimizer,critic_loss_fn,generator_loss_fn)




wgan.fit(train, batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[Generator(num_img=36), SaveModel(), GraphLosses()])