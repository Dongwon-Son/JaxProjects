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
        for ft in self.features[:-1]:
            x = nn.Conv(ft, (7,7), strides=(2,2), padding='VALID')(x)
            x = nn.relu(x)
        x = nn.Conv(self.features[-1], (7,7), strides=(2,2), padding='VALID')(x)
        # x = nn.avg_pool(x, window_shape=(3,3), strides=(2,2), padding='SAME')
        return x

class Decoder(nn.Module):
    features: Sequence[int]
    sigma: float

    @nn.compact
    def __call__(self, x):
        for ft in self.features:
            x = nn.ConvTranspose(ft, (7,7), strides=(2,2), padding='VALID')(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(1, (2,2), padding='VALID')(x)
        return x, self.sigma * jnp.ones_like(x)


#%%
# define losses for VQ-VAE
def vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model):
    ze = enc_model.apply(variable_tuple[0], data)
    codebook = variable_tuple[-1]
    ze_diff_square = jnp.sum(jnp.square(jnp.expand_dims(ze, axis=-1)- codebook), axis=-2) # (nb, nc)
    e = jnp.argmin(ze_diff_square, axis=-1) # (nb)
    e_onehot = jax.nn.one_hot(e, codebook.shape[-1]) # (nb, nc)
    zq = jnp.sum(jnp.expand_dims(e_onehot, axis=-2) * codebook, axis=-1) # (nb, nz)

    xp_mu, xp_sigma = dec_model.apply(variable_tuple[1], jax.lax.stop_gradient(zq) + ze - jax.lax.stop_gradient(ze))
    _, jkey = jax.random.split(jkey)
    noise_xp = jax.random.normal(jkey, shape=xp_mu.shape)
    xp = xp_mu + xp_sigma * noise_xp
    return xp, (ze, zq)

def vqvae_loss_func(variable_tuple, jkey, data, enc_model, dec_model):
    xp, (ze, zq) = vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model)
    return jnp.mean(jnp.square(data - xp)) + 0.1*jnp.mean(jnp.square(jax.lax.stop_gradient(ze)-zq)) + 0.1*jnp.mean(jnp.square(ze-jax.lax.stop_gradient(zq)))

# %%
nb = 8
img_size = [28,28]
test_img_input = jnp.ones([nb, *img_size], dtype=jnp.float32)
test_img_input = test_img_input[...,None]
jkey = jax.random.PRNGKey(0)

enc_model = Encoder([8,16])
enc_variables = enc_model.init(jkey, test_img_input)
test_enc_out = jit(enc_model.apply)(enc_variables, test_img_input)

#%%
_, jkey = jax.random.split(jkey)
dec_model = Decoder([16,8], sigma=0.05)
dec_variables = dec_model.init(jkey, test_enc_out[0])
test_dec_out = jit(dec_model.apply)(dec_variables, test_enc_out[0])

#%%
_, jkey = jax.random.split(jkey)
codebook_dim = 20
codebook = jax.random.normal(jkey, shape=(test_enc_out.shape[-1], codebook_dim))
vqvae_loss_func_partial = partial(vqvae_loss_func, enc_model=enc_model, dec_model=dec_model)


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
    xp, z = vqvae_reconstruction_func(variable_tuple, jkey, data, enc_model, dec_model)

    _, jkey = jax.random.split(jkey)
    cb = variable_tuple[-1]
    e = jax.random.categorical(jkey, logits=np.ones(cb.shape[-1:]), shape=z[-1].shape[:-1])
    e_onehot = jax.nn.one_hot(e, cb.shape[-1]) # (nb, nc)
    zq = jnp.sum(jnp.expand_dims(e_onehot, axis=-2) * cb, axis=-1) # (nb, nz)
    xp_gen, _ = dec_model.apply(variable_tuple[1], jax.lax.stop_gradient(zq))

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

params = (enc_variables, dec_variables, codebook)
for x,y in ds_train.take(1):
    display_predictions(params, jkey, x)

#%%
# test grad func
value_and_grad_func = jax.value_and_grad(vqvae_loss_func_partial)
for x,y in ds_train.take(1):
    value, grad = value_and_grad_func(params, jkey, np.array(x).astype(np.float32)/255.0)

# %%
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, jkey, x):
    values, grads = value_and_grad_func(params, jkey, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
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
