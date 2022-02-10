#%%
from typing import Sequence
from jax import numpy as jnp
from jax import vmap, lax, jit
import jax
import numpy as np
import flax.linen as nn
from functools import partial
import optax

VAE_TYPE = 'VQVAE'

class Encoder(nn.Module):
    features: Sequence[int]
    vae_type: str
    
    @nn.compact
    def __call__(self, x):
        for ft in self.features:
            x = nn.Conv(ft, (7,7), padding='VALID')(x)
            x = nn.relu(x)
        x = nn.max_pool(x, window_shape=tuple(x.shape[-3:-1]), strides=tuple(x.shape[-3:-1]))
        x = jnp.squeeze(x, axis=[-2,-3])
        if self.vae_type == 'VAE':
            x = nn.Dense(self.features[-1]*2)(x)
            z_mu, z_logsigma = jnp.split(x, 2, ais=-1)
            return z_mu, jnp.exp(z_logsigma)
        elif self.vae_type == 'VQVAE':
            x = nn.Dense(self.features[-1])(x)
            return x, None
        else:
            raise ValueError('VQE RTPE DEF')

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
# define losses for VQ-VAE
def vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model, code_book):
    zb, _ = enc_model.apply(variable_tuple[0], data)

    cb_diff = jnp.sum(jnp.square(jnp.expand_dims(zb, axis=-1)- code_book), axis=-2) # (nb, nc)
    cb_idx = jnp.argmin(cb_diff, axis=-1) # (nb)
    cb_onehot = jax.nn.one_hot(cb_idx, code_book.shape[-1]) # (nb, nc)
    z = jnp.sum(jnp.expand_dims(cb_onehot, axis=-2) * code_book, axis=-1) + zb - jax.lax.stop_gradient(zb) # (nb, nz)

    xp_mu, xp_sigma = dec_model.apply(variable_tuple[1], z)
    _, jkey = jax.random.split(jkey)
    noise_xp = jax.random.normal(jkey, shape=xp_mu.shape)
    xp = xp_mu + xp_sigma * noise_xp
    return xp, (zb, cb_onehot, z)

def vqvae_loss_func(variable_tuple, jkey, data, enc_model, dec_model, codebook):
    xp, (zb, cb_onehot, z) = vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model, codebook)
    # new_codebook = jnp.mean(jnp.expand_dims(cb_onehot, axis=-2) * jnp.expand_dims(zb, axis=-1), axis=0) # (nz, nc)
    return jnp.mean(data - xp)

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
    kld = -0.5 * jnp.sum(1 + jnp.log(z_sigma) - jnp.square(z_mu) - z_sigma, axis=-1)
    return 2000*jnp.mean(jnp.square(data - xp)) + jnp.mean(kld)

# %%
nb = 8
img_size = [28,28]
test_img_input = jnp.ones([nb, *img_size], dtype=jnp.float32)
test_img_input = test_img_input[...,None]
jkey = jax.random.PRNGKey(0)

enc_model = Encoder([32,32], VAE_TYPE)
enc_variables = enc_model.init(jkey, test_img_input)
test_enc_out = jit(enc_model.apply)(enc_variables, test_img_input)

#%%
_, jkey = jax.random.split(jkey)
dec_model = Decoder([32,32], sigma=0.05)
dec_variables = dec_model.init(jkey, test_enc_out[0])
test_dec_out = jit(dec_model.apply)(dec_variables, test_enc_out[0])

#%%
_, jkey = jax.random.split(jkey)
codebook = jax.random.normal(jkey, shape=(test_enc_out[0].shape[-1], 30))
vqvae_loss_func_partial = partial(vqvae_loss_func, enc_model=enc_model, dec_model=dec_model)
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
img_size = ds_info.features['image'].shape

#%%
import matplotlib.pyplot as plt

def display_predictions(variable_tuple, jkey, data):
    data = np.array(data).astype(np.float32)/255.0
    # xp, z = vae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model)
    xp, z = vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model, codebook)

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
value_and_grad_func = jax.value_and_grad(vqvae_loss_func_partial)
params = (enc_variables, dec_variables)
# res = vqvae_loss_func_partial(params, jkey, np.array(x).astype(np.float32)/255.0, codebook=codebook)
for x,y in ds_train.take(1):
    value, grad = value_and_grad_func(params, jkey, np.array(x).astype(np.float32)/255.0, codebook=codebook)

# %%
# from flax.training import train_state
value_and_grad_func = jax.value_and_grad(vqvae_loss_func_partial)
optimizer = optax.adam(learning_rate=1e-3)
params = (enc_variables, dec_variables)
opt_state = optimizer.init(params)

# @jax.jit
def train_step(params, opt_state, jkey, x, codebook):
    values, grads = value_and_grad_func(params, jkey, x, codebook=codebook)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    xp, (zb, cb_onehot, z) = vqvae_reconstruction_func(params, jkey, x, enc_model, dec_model, codebook)
    new_codebook = jnp.mean(jnp.expand_dims(cb_onehot, axis=-2) * jnp.expand_dims(zb, axis=-1), axis=0)
    return opt_state, params, new_codebook

# def one_epoch(opt_state, params, jkey):
#     for x, y in tfds.as_numpy(ds_train):
#         opt_state, params, new_codebook = train_step(params, opt_state, jkey, np.array(x).astype(np.float32)/255.0, new_codebook)
#         _, jkey = jax.random.split(jkey)
#     return opt_state, params, jkey

#%%
epoch_no = 100
for en in range(epoch_no):
    for x, y in tfds.as_numpy(ds_train):
        opt_state, params, codebook = train_step(params, opt_state, jkey, np.array(x).astype(np.float32)/255.0, codebook)
        _, jkey = jax.random.split(jkey)
    # opt_state, params, jkey = one_epoch(opt_state, params, jkey)
    for x, y in ds_train.take(1):
        display_predictions(params, jkey, x)

# %%
