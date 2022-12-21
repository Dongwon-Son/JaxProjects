import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jax
import optax
import os, shutil, datetime
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import argparse

parser = argparse.ArgumentParser(description='Argparse')

parser.add_argument('--fixx', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reg_coef', type=float, default=0.1)

args    = parser.parse_args()

NB = 512
NX = 1
NZ = 4
jkey = jax.random.PRNGKey(args.seed)

def gen_data(jkey):
    return jax.random.normal(jkey, shape=(NB,1), dtype=jnp.float32)*2.0+3.0

class Enc(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.Dense(64)(x)
            x = nn.relu(x)
        x = nn.Dense(NZ*2)(x)
        x_mu, x_log_scale = jnp.split(x, 2, -1)
        return x_mu, jnp.exp(x_log_scale)

class Dec(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.Dense(64)(x)
            x = nn.relu(x)
        x = nn.Dense(NX)(x)
        return x

x_test = gen_data(jkey)
enc = Enc()
dec = Dec()
encded, enc_params = enc.init_with_output(jkey, x_test)
dec_params = dec.init(jkey, encded)
params = (enc_params, dec_params)

def loss_func(params, x, jkey):
    if args.fixx:
        enc_inputs = jnp.zeros_like(x)
    else:
        enc_inputs = x
    z_mu, z_scale = enc.apply(params[0], enc_inputs)
    z = jax.random.normal(jkey, shape=z_mu.shape, dtype=jnp.float32) * z_scale + z_mu
    _, jkey = jax.random.split(jkey)
    x_rec = dec.apply(params[1], z)
    rec_loss = jnp.sum((x_rec - x)**2, -1)
    z_var = jnp.square(z_scale)
    kld = -0.5 * jnp.sum(1 + jnp.log(z_var) - jnp.square(z_mu) - z_var, axis=-1)
    assert rec_loss.shape == kld.shape
    return jnp.mean(args.reg_coef * kld + rec_loss)

loss_value_and_grad = jax.value_and_grad(loss_func)

optimizer = optax.adam(4e-3)
opt_state = optimizer.init(params)

def train_step(params, opt_state, jkey):
    for _ in range(8):
        x = gen_data(jkey)
        _, jkey = jax.random.split(jkey)
        loss, grad = loss_value_and_grad(params, x, jkey)
        _, jkey = jax.random.split(jkey)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    train_metric = {'loss': loss}
    return params, opt_state, train_metric, jkey

train_step_jit = jax.jit(train_step)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs_cvae/' + current_time + "_" + str(args.seed) + "_" + str(args.fixx) + '_' + str(args.reg_coef)
summary_writer = tf.summary.create_file_writer(log_dir)
shutil.copy(os.path.abspath(__file__), os.path.join(log_dir, os.path.basename(os.path.abspath(__file__))))

for itr in range(90_000):
    params, opt_state, train_metric, jkey = train_step_jit(params, opt_state, jkey)

    if itr != 0 and itr % 2500 == 0:
        # evals
        x = gen_data(jkey)
        _, jkey = jax.random.split(jkey)
        z_mu, z_scale = enc.apply(params[0], x)
        z = jax.random.normal(jkey, shape=z_mu.shape, dtype=jnp.float32) * z_scale + z_mu
        x_rec = dec.apply(params[1], z)

        _, jkey = jax.random.split(jkey)
        z = jax.random.normal(jkey, shape=z_mu.shape, dtype=jnp.float32)
        x_gen = dec.apply(params[1], z)

        with summary_writer.as_default():
            with tf.name_scope("eval"):
                tf.summary.histogram('x_input', x, step=itr)
                tf.summary.histogram('x_rec', x_rec, step=itr)
                tf.summary.histogram('x_gen', x_gen, step=itr)
            with tf.name_scope("eval"):
                tf.summary.scalar('mean_x_input', jnp.mean(x), step=itr)
                tf.summary.scalar('std_x_input', jnp.std(x), step=itr)
                tf.summary.scalar('mean_x_rec', jnp.mean(x_rec), step=itr)
                tf.summary.scalar('std_x_rec', jnp.std(x_rec), step=itr)
                tf.summary.scalar('mean_x_gen', jnp.mean(x_gen), step=itr)
                tf.summary.scalar('std_x_gen', jnp.std(x_gen), step=itr)
        print(f"itr: {itr}, loss: {train_metric['loss']:.3f}")
