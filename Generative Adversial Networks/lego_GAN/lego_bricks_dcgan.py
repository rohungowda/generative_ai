import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import (layers, models, callbacks, utils, metrics, losses, optimizers)

IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 128
Z_DIM = 100
EPOCHS = 300
LOAD_MODEL = False
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
LEARNING_RATE = 0.0002
NOISE_PARAM = 0.1


# scales 0-255 btwn -1,1 for the generator tanh function bc it provides better gradients because of stronger slopes - helps the model be more stable
def preprocess(imgs):
    imgs = (tf.cast(imgs,'float32') - 127.5)/ 127.5
    return imgs

# the use of batch normalization directly after the conv2D layer allows for us not to need th bias as it mean = 0, var = 1 the incoming values from the activation

# don't want dropout in generator bc each and every node is critical

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
        self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
        self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_real_acc_metric,
            self.d_fake_acc_metric,
            self.d_acc_metric,
            self.g_loss_metric,
            self.g_acc_metric,
        ]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        # Train the discriminator on fake images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(
                random_latent_vectors, training=True
            )
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(
                generated_images, training=True
            )

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + NOISE_PARAM * tf.random.uniform(
                tf.shape(real_predictions)
            )
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - NOISE_PARAM * tf.random.uniform(
                tf.shape(fake_predictions)
            )

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, fake_predictions)

        # gradient goes outside the with statement??????

        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.d_real_acc_metric.update_state(real_labels, real_predictions)
        self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
        self.d_acc_metric.update_state(
            [real_labels, fake_labels], [real_predictions, fake_predictions]
        )
        self.g_loss_metric.update_state(g_loss)
        self.g_acc_metric.update_state(real_labels, fake_predictions)

        return {m.name: m.result() for m in self.metrics}

#In essence, a slope of -0.3 means that negative inputs result in a larger "bounce" 
#(output value) than a slope of -0.2, leading to potentially faster learning but also more variability in the training process.



    # when padding is same and strides =1, output size is same as input size
    # momentum, the more it is the more importance givern to old mean and variance
    # moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
    # moving_var = moving_var * momentum + var(batch) * (1 - momentum)

generator_input = layers.Input(shape=(Z_DIM,))
x = layers.Reshape((1, 1, Z_DIM))(generator_input)
x = layers.Conv2DTranspose(
    512, kernel_size=4, strides=1, padding="valid", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    64, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
    CHANNELS,
    kernel_size=4,
    strides=2,
    padding="same",
    use_bias=False,
    activation="tanh",
)(x)
generator = models.Model(generator_input, generator_output)
generator.summary()


discriminator_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False)(
    discriminator_input
)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    128, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    256, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    512, kernel_size=4, strides=2, padding="same", use_bias=False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    1,
    kernel_size=4,
    strides=1,
    padding="valid",
    use_bias=False,
    activation="sigmoid",
)(x)
discriminator_output = layers.Flatten()(x)

discriminator = models.Model(discriminator_input, discriminator_output)
discriminator.summary()

# smaller discriminator may lead the model to focus more on finer details allowing the model to overfit to the training data
# the key is to learn enough 


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

    #plt.subplots_adjust(wspace=0, hspace=0.5)

    if save:
        plt.savefig(name)
    else:
        plt.show()

class Generator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, Z_DIM))
        generated = self.model.generator(random_latent_vectors)
        generated = tf.cast((tf.cast(generated,'float32') * 127.5) + 127.5, tf.uint8).numpy()
        if not (epoch % 10):
            Display(generated,True,f'./pokemon_output/generated_img_{epoch}.png')

class SaveModel(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if not (epoch % 25):
            #self.model.evaluate(validation_dataset)
            self.model.generator.save(f"./gen_{epoch}.keras")
            self.model.discriminator.save(f"./disc_{epoch}.keras")
            self.model.save(f"./gan_{epoch}.keras")

def plot_losses(d_loss_metric, d_real_acc_metric, d_fake_acc_metric, g_loss_metric, g_acc_metric):
    epochs = range(1, len(d_loss_metric) + 1)

    fig, ax = plt.subplots()  # Create new figure and axes
    ax.plot(epochs, d_loss_metric, label='Discriminator Loss')
    ax.plot(epochs, g_loss_metric, label='Generator Loss')
    ax.plot(epochs, d_real_acc_metric, label='Real Accuracy')
    ax.plot(epochs, d_fake_acc_metric, label='Fake Accuracy')
    ax.plot(epochs, g_acc_metric, label='Generator total accuracy')
    
    ax.set_title('Discriminator and Generator Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.savefig("Training_losses.png")  # Save the plot before displaying it
    #plt.show()

class TrainingPlot(callbacks.Callback):
    def __init__(self):
        self.discr_loss = []
        self.gen_loss = []
        self.discr_real =[]
        self.discr_fake= []
        self.gen_acc=[]

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        self.discr_loss.append(self.model.metrics[0].result().numpy())
        self.gen_loss.append(self.model.metrics[4].result().numpy())
        self.discr_real.append(self.model.metrics[1].result().numpy())
        self.discr_fake.append(self.model.metrics[2].result().numpy())
        self.gen_acc.append(self.model.metrics[5].result().numpy())
        


    def on_train_end(self, logs=None):
        plot_losses(self.discr_loss, self.discr_real, self.discr_fake, self.gen_loss, self.gen_acc)

dataset = utils.image_dataset_from_directory(
    directory="../datasets/lego_dataset",
    labels=None,
    label_mode = None,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    seed=42,
    interpolation='bilinear'
)

train= dataset

train_dataset = train.map(lambda x: preprocess(x))
#validation_dataset = validation.map(lambda x: preprocess(x))

test_feature = next(iter(train_dataset.take(1)))
imgs = tf.cast((tf.cast(test_feature,'float32') * 127.5) + 127.5, tf.uint8).numpy()
#Display(imgs,False)
#plot_losses([3.3, 3.4],[3.2, 3.1])


train_dataset = train.map(lambda x: preprocess(x))
#validation_dataset = validation.map(lambda x: preprocess(x))




# Create a DCGAN
dcgan = DCGAN(
    discriminator=discriminator, generator=generator, latent_dim=Z_DIM
)

#dcgan.build((None,IMAGE_SIZE,IMAGE_SIZE,CHANNELS))

# gives the model beta_1 a short timer memory when it comes to gradients, only priortizing recent ones 
# makes process more violatile and less stable
dcgan.compile(optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
        optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
)

dcgan.fit(train_dataset, epochs=EPOCHS, callbacks=[
Generator(num_img=36),
SaveModel(),
TrainingPlot(),
])