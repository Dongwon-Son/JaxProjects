# %%
# import libraries
import jax.numpy as jnp
import numpy as np
import jax
import flax.linen as nn
from typing import Sequence
from functools import partial
import optax
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

# %%
# hyper parameters
NB = 16
NZD = 16

# %%
# define networks
class Critic(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(8, kernel_size=(7,7))(x)
        x = nn.relu(x)
        x = nn.Conv(8, kernel_size=(5,5))(x)
        x = nn.relu(x)
        
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Generator(nn.Module):
    image_size: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(np.prod(np.array(self.image_size)))(x)
        x = nn.sigmoid(x)
        x = jnp.reshape(x, (x.shape[0], *self.image_size))
        return x

# %%
# define WGAN-GP loss
def wgan_loss(params, jkey, critic_net, gen_net, x_real):
    critic_params = params[0]
    gen_params = params[1]

    z = jax.random.normal(jkey, shape=(NB, NZD))
    _, jkey = jax.random.split(jkey)
    x_gen = gen_net.apply(gen_params, z)

    eval_x_real = critic_net.apply(critic_params, x_real)
    eval_x_gen = critic_net.apply(critic_params, jax.lax.stop_gradient(x_gen))

    critic_x_func = partial(lambda x,y : jnp.sum(critic_net.apply(x,y)), critic_params)
    t = np.random.uniform(0, 1, size=(x_real.shape[0], 1, 1, 1))
    x_tilda = x_real* t + (1-t)* jax.lax.stop_gradient(x_gen)
    grad_D_x_tilda = jax.grad(critic_x_func)(x_tilda)
    grad_D_x_tilda_norm = jnp.linalg.norm(jnp.reshape(grad_D_x_tilda, (grad_D_x_tilda.shape[0], -1)), axis=-1)

    critic_loss = (jnp.mean(eval_x_real) - jnp.mean(eval_x_gen)
                    + 10*jnp.mean(jnp.square(grad_D_x_tilda_norm-1)))
    # critic_loss = jnp.mean(eval_x_real) - jnp.mean(eval_x_gen)
    # critic_loss = jnp.mean(jnp.square(grad_D_x_tilda_norm-1))
    generator_loss = critic_net.apply(jax.lax.stop_gradient(critic_params), x_gen)
    generator_loss = jnp.mean(generator_loss)

    return critic_loss + generator_loss



# %%
# define train function

# %%
# import dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
ds_train = ds_train.batch(NB).prefetch(1)
ds_test = ds_test.batch(NB)
img_size = ds_info.features['image'].shape

for bd in ds_test.take(1):
    test_data = bd


# %%
# init networks
critic_net = Critic()
gen_net = Generator(img_size)
jkey = jax.random.PRNGKey(0)
critic_net_params = critic_net.init(jkey, jnp.array(test_data[0]))
_, jkey = jax.random.split(jkey)
test_z = jax.random.normal(jkey, shape=[NB, NZD])
_, jkey = jax.random.split(jkey)
gen_net_params = gen_net.init(jkey, test_z)
params = (critic_net_params, gen_net_params)

# %%
# jit test
# critic_net_jit = jax.jit(critic_net.apply)
# gen_net_jit = jax.jit(gen_net.apply)

# %timeit critic_net_jit(critic_net_params, jnp.array(test_data[0]))
# %timeit gen_net_jit(gen_net_params, test_z)

# _, jkey = jax.random.split(jkey)
# loss_test = wgan_loss(params, jkey, critic_net, gen_net, jnp.array(test_data[0]))
# grad_loss_jit = jax.jit(jax.grad(wgan_loss), static_argnums=(2,3))
# grad_test = grad_loss_jit(params, jkey, critic_net, gen_net, jnp.array(test_data[0]))

# %%
# init optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
value_and_grad_func = jax.value_and_grad(wgan_loss)

# %%
# def train func
@jax.jit
def train_step(params, opt_state, jkey, x):
    values, grads = value_and_grad_func(params, jkey, critic_net, gen_net, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# def display func
def display_pred(x, params):
    z = jax.random.normal(jax.random.PRNGKey(0), (NB, NZD))
    gen_imgs = gen_net.apply(params[1], z)
    plt.figure()
    for i in range(NB):
        plt.subplot(int(NB/4),4,i+1)
        plt.imshow(gen_imgs[i])
        plt.axis('off')
    plt.show()

    # critic test
    eval_gen = critic_net.apply(params[0], gen_imgs)
    eval_real = critic_net.apply(params[0], np.array(x).astype(np.float32)/255.0)
    print(np.mean(eval_gen), np.mean(eval_real))


# %%
# start train
epoch_no = 100
for en in range(epoch_no):
    for x, y in tfds.as_numpy(ds_train):
        _, jkey = jax.random.split(jkey)
        params, opt_state = train_step(params, opt_state, jkey, np.array(x).astype(np.float32)/255.0)
    for x,y in ds_test.take(1):
        display_pred(x, params)
# %%
