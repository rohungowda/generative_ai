import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers,models,utils,metrics,losses,optimizers
from scipy.stats import norm
from keras.models import load_model, save_model
from keras.saving import register_keras_serializable


image_size = 32
batch_size = 128
embedding_dim = 200
l_r = 0.0005

num_filters = 128

load = False

train_data = utils.image_dataset_from_directory(
    "img_align_celeba/img_align_celeba",
    labels=None,
    color_mode="rgb",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)



def pre_process(img):
    img = tf.cast(img, dtype="float32") / 255.0
    return img

#  lazy loading or lazy evaluation. It means that data is loaded and processed only when it's required
train = train_data.map(lambda x: pre_process(x))

@register_keras_serializable()
class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch,dim)) # randomized vector with gaussian normal dbn
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon





encoder_input = layers.Input(shape=(image_size, image_size, 3), name="encoder_input")
x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="same")(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

shape_before_flattening = K.int_shape(x)[1:]  # the decoder will need this!

x = layers.Flatten()(x)
z_mean = layers.Dense(embedding_dim, name="z_mean")(x)
z_log_var = layers.Dense(embedding_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


decoder_input = layers.Input(shape=(embedding_dim,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

decoder_output = layers.Conv2DTranspose(3, kernel_size=3, strides=1, activation="sigmoid", padding="same")(x)

decoder = models.Model(decoder_input, decoder_output)
decoder.summary()


@register_keras_serializable()
class B_VAE(models.Model):
    def __init__(self,encoder,decoder,beta,**kwargs):
        super(B_VAE,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_track = metrics.Mean(name="total_loss")
        self.reconstruction_loss_track = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_track = metrics.Mean(name="kl_loss")

    
    # allows you to access it like a variable
    @property
    def metrics(self):
        return [self.total_loss_track, self.reconstruction_loss_track, self.kl_loss_track]
    
    
    
    def call(self,inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed_image = self.decoder(z)
        return z_mean, z_log_var, reconstructed_image

    def train_step(self,data):
        with tf.GradientTape() as tape:
            #forward pass and records information that will be usefull to use outside
            z_mean, z_log_var, reconstructed = self(data)

            # calculates losses for each pixel in height width and channel for all the batches, so every pixel and sums it which gives us a loss per batch, 
            # and then takes the mean loss across all the batches
            reconstruction_loss = tf.reduce_mean( self.beta * losses.mean_squared_error(data, reconstructed)) # think like you want a mean loss so heavily focus on reconstruction  
            # calculates the values first then adds all the vectors together to get[B,d] which it then averages all over
            kl_loss =  tf.reduce_mean(tf.reduce_sum(-0.5 *(1 + z_log_var - tf.square(z_mean)- tf.exp(z_log_var)), axis = 1))
            total_loss = reconstruction_loss + kl_loss
        # d Loss / d X where X is the weights -> this is the back propogation step
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights)) # sets the gradient for each layer of the weights

        self.total_loss_track.update_state(total_loss)
        self.reconstruction_loss_track.update_state(reconstruction_loss)
        self.kl_loss_track.update_state(kl_loss)


        return {m.name: m.result() for m in self.metrics}




if load:
    encoder = load_model("encoder.keras",custom_objects={'Sampling': Sampling})
    decoder = load_model("decoder.keras")
vae = B_VAE(encoder, decoder, 2000)

optimizer = optimizers.Adam(learning_rate=l_r)
vae.compile(optimizer=optimizer)

if not load:
    vae.fit(train_data, epochs=1)
    vae.encoder.save('encoder.keras')
    vae.decoder.save('decoder.keras')


# take is batch number
sample_images = np.array(list(train.take(1).get_single_element()))

_, _, z = vae.encoder.predict(sample_images)

x = np.linspace(-3, 3, 100)

#20 inches and the height will be 5 inches.
fig = plt.figure(figsize=(20, 5))

# control the spacing between subplots 
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(50):
    #number of rows, the number of columns, and the index of the subplot you want to add, note 5 * 10 = 50
    ax = fig.add_subplot(5, 10, i + 1)
    ax.hist(z[:, i], density=True, bins=20)
    ax.axis("off")
    ax.text(0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes)
    ax.plot(x, norm.pdf(x))

plt.show()

grid_width, grid_height = (10, 3)
z_sample = np.random.normal(size=(grid_width * grid_height, embedding_dim))

# Decode the sampled points
reconstructions = vae.decoder(z_sample)

print(reconstructions.shape)

# Draw a plot of decoded images
fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Output the grid of faces
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i])

reconstructions *= 255
reconstructions.numpy()
for i in range(reconstructions.shape[0]):
    img = utils.array_to_img(reconstructions[i])
    img.save(f'./output_my/generated_img_{i}.png')

plt.show()

# I understand how the vae works and to some extent the math behind it. I just have some questions that deal with why the original faces was producing blank images I believe this has to
# do with how it wasn't a problem of understanding but of how the image wwas being generated.