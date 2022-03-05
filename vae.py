#%%
from typing import Sequence
from jax import numpy as jnp
from jax import vmap, lax, jit
import jax
import numpy as np
import flax.linen as nn
from functools import partial
import optax

class Encoder(nn.Module):
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        for ft in self.features:
            x = nn.Conv(ft, (7,7), padding='VALID')(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=tuple(x.shape[-3:-1]), strides=tuple(x.shape[-3:-1]))
        x = jnp.squeeze(x, axis=[-2,-3])
        x = nn.Dense(self.features[-1]*2)(x)
        z_mu, z_logsigma = jnp.split(x, 2, axis=-1)
        return z_mu, jnp.exp(z_logsigma)

class Decoder(nn.Module):
    features: Sequence[int]
    sigma: float

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, tuple(x.shape[:-1]) + (1, 1) + tuple(x.shape[-1:]))
        x = jnp.tile(x, (1, 16, 16, 1))
        for ft in self.features:
            x = nn.ConvTranspose(ft, (7,7), padding='VALID')(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(1, (1,1), padding='VALID')(x)
        return x, self.sigma * jnp.ones_like(x)


#%%
# define losses for VAE
def vae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model):
    z_mu, z_sigma = enc_model.apply(variable_tuple[0], data)
    noise_z = jax.random.normal(jkey, shape=z_mu.shape)
    z = z_mu + z_sigma * noise_z
    xp_mu, xp_sigma = dec_model.apply(variable_tuple[1], z)

    _, jkey = jax.random.split(jkey)
    noise_xp = jax.random.normal(jkey, shape=xp_mu.shape)
    xp = xp_mu + xp_sigma * noise_xp
    return xp, (z_mu, z_sigma, z)

def vae_loss_func(variable_tuple, jkey, data, enc_model, dec_model):
    xp, (z_mu, z_sigma, z) = vae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model)
    z_var = jnp.square(z_sigma)
    kld = -0.5 * jnp.sum(1 + jnp.log(z_var) - jnp.square(z_mu) - z_var, axis=-1)
    return 2000*jnp.mean(jnp.square(data - xp)) + jnp.mean(kld)

# %%
nb = 8
img_size = [28,28]
test_img_input = jnp.ones([nb, *img_size], dtype=jnp.float32)
test_img_input = test_img_input[...,None]
jkey = jax.random.PRNGKey(0)

enc_model = Encoder([32,32])
enc_variables = enc_model.init(jkey, test_img_input)
test_enc_out = jit(enc_model.apply)(enc_variables, test_img_input)

#%%
_, jkey = jax.random.split(jkey)
dec_model = Decoder([32,32], sigma=0.05)
dec_variables = dec_model.init(jkey, test_enc_out[0])
test_dec_out = jit(dec_model.apply)(dec_variables, test_enc_out[0])

#%%
_, jkey = jax.random.split(jkey)
vae_loss_func_partial = partial(vae_loss_func, enc_model=enc_model, dec_model=dec_model)
# vqvae_loss_func_jit = jit(vqvae_loss_func_partial)
# vae_loss_func_jit = jit(vae_loss_func_partial)


#%%
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
ds_train = ds_train.batch(nb).prefetch(1)
ds_test = ds_test.batch(nb)
img_size = ds_info.features['image'].shape

#%%
import matplotlib.pyplot as plt

def display_predictions(variable_tuple, jkey, data):
    data = np.array(data).astype(np.float32)/255.0
    xp, z = vae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model)

    noise_z = jax.random.normal(jkey, shape=z[-1].shape)
    xp_gen, _ = dec_model.apply(variable_tuple[1], noise_z)

    plt.figure()
    for i in range(nb):
        plt.subplot(3,8,2*i+1)
        plt.imshow(data[i])
        plt.axis('off')
        plt.subplot(3,8,2*i+2)
        plt.imshow(np.clip(xp[i], 0, 1))
        plt.axis('off')
    for i in range(8):
        plt.subplot(3,8,i+17)
        plt.imshow(np.clip(xp_gen[i], 0, 1))
        plt.axis('off')
    plt.show()

for x,y in ds_train.take(1):
    display_predictions((enc_variables, dec_variables), jkey, x)

#%%
# test grad func
value_and_grad_func = jax.value_and_grad(vae_loss_func_partial)
params = (enc_variables, dec_variables)
# res = vqvae_loss_func_partial(params, jkey, np.array(x).astype(np.float32)/255.0, codebook=codebook)
for x,y in ds_train.take(1):
    value, grad = value_and_grad_func(params, jkey, np.array(x).astype(np.float32)/255.0)

# %%
# from flax.training import train_state
value_and_grad_func = jax.value_and_grad(vae_loss_func_partial)
optimizer = optax.adam(learning_rate=1e-3)
params = (enc_variables, dec_variables)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, jkey, x):
    values, grads = value_and_grad_func(params, jkey, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    xp, _ = vae_reconstruction_func(params, jkey, x, enc_model, dec_model)
    return opt_state, params

#%%
epoch_no = 100
for en in range(epoch_no):
    for x, y in tfds.as_numpy(ds_train):
        opt_state, params = train_step(params, opt_state, jkey, np.array(x).astype(np.float32)/255.0)
        _, jkey = jax.random.split(jkey)
    for x, y in ds_test.take(1):
        display_predictions(params, jkey, x)

# %%
