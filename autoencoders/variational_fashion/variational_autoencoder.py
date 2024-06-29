import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turns off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supresses tensorflow info messages


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models, metrics, losses, optimizers
from tensorflow.keras import backend as K
from scipy.stats import norm

def pre_process(imgs):
    imgs = imgs.astype('float32') / 255.0
    imgs = np.pad(imgs,((0,0),(2,2),(2,2)), constant_values = 0.0) # 0 because that is the number of elements
    # np.expand_dims() is used to add a new axis 
    imgs = np.expand_dims(imgs,-1)
    return imgs

class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch,dim)) # randomized vector with gaussian normal dbn
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



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
    
    @property
    def metrics_val(self):
        return [self.total_loss_track_val, self.reconstruction_loss_track_val, self.kl_loss_track_val]
    
    
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
            reconstruction_loss = tf.reduce_mean( self.beta * losses.binary_crossentropy(data,reconstructed,axis=(1,2,3))) # think like you want a mean loss so heavily focus on reconstruction  
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
    
    

def plot_losses(total_loss, reconstruction_loss, ak_l_loss):
    

    plt.figure(figsize=(15, 5))  # Adjust the figure size as per your preference

    # Plot Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(total_loss, label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Reconstruction Loss
    plt.subplot(1, 3, 2)
    plt.plot(reconstruction_loss, label='Reconstruction Loss', color='orange')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot AK_L Loss
    plt.subplot(1, 3, 3)
    plt.plot(ak_l_loss, label='AK_L Loss', color='green')
    plt.title('AK_L Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()

x_train = pre_process(x_train)
x_test = pre_process(x_test)

embedding_size = 2




encoder_input = layers.Input(shape=(32,32,1), name="encoder_input")
x = layers.Conv2D(filters=32,kernel_size=(3,3),strides=2, activation='relu',padding='same')(encoder_input)
x = layers.Conv2D(filters=64,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2D(filters=128,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
shape_before_flatten = K.int_shape(x)[1:]
x = layers.Flatten()(x)
z_mean = layers.Dense(embedding_size, name="z_mean")(x)
z_log_var = layers.Dense(embedding_size, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input,[z_mean, z_log_var, z], name="encoder")


decoder_input = layers.Input(shape=(embedding_size,), name='decoder_input')
x = layers.Dense(np.prod(shape_before_flatten))(decoder_input)
x = layers.Reshape(shape_before_flatten)(x)
x = layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
x = layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=2, activation='relu',padding='same')(x)
decoder_output = layers.Conv2D(filters=1, kernel_size=(3,3), strides=1,activation='sigmoid', padding='same',name='decoder_output')(x)

decoder = models.Model(decoder_input,decoder_output, name="decoder")

b_vae = B_VAE(encoder, decoder, beta=500)

b_vae.summary()

optimizer = optimizers.Adam(learning_rate=0.0005)
b_vae.compile(optimizer=optimizer)
b_vae.fit(x_train,epochs=5,batch_size=100, shuffle=True)



analysis_images = x_test[:5000]
random_indices = np.random.choice(len(analysis_images), 10, replace=False)

# prediction
predicted_mean, predicted_log_var, predictions = b_vae.predict(analysis_images)

print(f"average mean: {tf.reduce_mean(predicted_mean)}") # want to be close to 0
print(f"average log variance: {tf.reduce_mean(predicted_log_var)}") # want to be close to 0

random_samples = predictions[random_indices]
random_actual = analysis_images[random_indices]


z_mean, z_log_var, z = b_vae.encoder(x_test)

x = np.linspace(-5, 5, 100)

fig = plt.figure(figsize=(12, 5))
fig.subplots_adjust(hspace=0.6, wspace=0.4)




for i in range(2):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.hist(z[:, i], density=True, bins=20)
    ax.axis("off")
    ax.text(0.5, -0.2, f'Dimension {i + 1}', fontsize=10, ha="center", transform=ax.transAxes)
    ax.plot(x, norm.pdf(x))

plt.show()


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