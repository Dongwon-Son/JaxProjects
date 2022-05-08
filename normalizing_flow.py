# %%
# import libraries
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

jkey = jax.random.PRNGKey(0)
# %%
# define normalizing flow network
dim = 2
mlp1 = nn.Sequential([nn.Dense(128),
                    nn.relu,
                    nn.Dense(128),
                    nn.relu,
                    nn.Dense(dim)])
mlp2 = nn.Sequential([nn.Dense(128),
                    nn.relu,
                    nn.Dense(128),
                    nn.relu,
                    nn.Dense(dim)])

def forward(param, z):
    z1, z2 = jnp.split(z, 2, axis=-1)
    log_scale, shift = jnp.split(mlp1.apply(param[0],z1), 2, axis=-1)
    log_scale = nn.tanh(log_scale)
    z2 = z2 * jnp.exp(log_scale) + shift

    log_scale, shift = jnp.split(mlp2.apply(param[1],z2), 2, axis=-1)
    log_scale = nn.tanh(log_scale)
    x = jnp.concatenate([z1* jnp.exp(log_scale) + shift, z2], axis=-1)

    return x

def backward(param, x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    log_scale2, shift = jnp.split(mlp2.apply(param[1],x2), 2, axis=-1)
    log_scale2 = nn.tanh(log_scale2)
    x1 = (x1 - shift) * jnp.exp(-log_scale2)

    log_scale1, shift = jnp.split(mlp1.apply(param[0],x1), 2, axis=-1)
    log_scale1 = nn.tanh(log_scale1)
    z = jnp.concatenate([x1, (x2 - shift)* jnp.exp(-log_scale1)], axis=-1)

    log_det = jnp.concatenate((-log_scale1, -log_scale2), axis=-1)
    log_det = jnp.sum(log_det, axis=-1)
    return z, log_det

def logprob(param, x):
    z, log_det = backward(param, x)
    base_log_prob = z.shape[-1]*(-0.5) * jnp.log((2*np.pi)) -0.5*jnp.sum(z*z, axis=-1)

    return base_log_prob+log_det

def sample(param, jkey, n):
    z = jax.random.normal(jkey, shape=(n,dim))
    x = forward(param, z)
    return x

z = jax.random.normal(jkey, shape=(2,2))
z1, z2 = jnp.split(z, 2, axis=-1)
param1 = mlp1.init(jkey, z1)
param2 = mlp2.init(jkey, z2)
param = (param1, param2)

x = forward(param, z)
zn = backward(param, x)
lp = logprob(param, x)

# %%
# loss func
def loss_sample(param, x):
    lp = logprob(param, x)
    return -jnp.mean(lp)

loss_value_and_grad = jax.value_and_grad(loss_sample)

# %%
# train
optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(param)

dist = tfd.MixtureSameFamily(
    tfd.Categorical(probs=[0.3, 0.7]),
    tfd.MultivariateNormalDiag(loc=[[1.,2.],[-1.,-2.]], scale_diag=[[0.3,1.0], [1.,1.]])
    )


def train_step(param, opt_state, jkey, itr=1):
    for i in range(itr):
        x = dist.sample(2000, seed=jkey)
        value, grad = loss_value_and_grad(param, x)
        updates, opt_state = optimizer.update(grad, opt_state)
        param = optax.apply_updates(param, updates)
    return param, opt_state, value

train_step_jit = jax.jit(train_step, static_argnames=['itr'])

x_gt = dist.sample(2000, seed=jkey)
def sample_plot(param, jkey):
    x_sp = sample(param, jkey, 2000)
    plt.figure()
    plt.scatter(x_gt[...,0],x_gt[...,1])
    plt.scatter(x_sp[...,0],x_sp[...,1], c='r')
    plt.show()
    plt.close()
    


# %%
# train step
for i in range(100000):
    _, jkey = jax.random.split(jkey)
    param, opt_state, loss = train_step_jit(param, opt_state, jkey, itr=10)
    # param, opt_state, loss = train_step(param, opt_state, jkey, itr=1)
    if i % 1000 == 0:
        print(i, loss)
        sample_plot(param, jkey)

# %%
