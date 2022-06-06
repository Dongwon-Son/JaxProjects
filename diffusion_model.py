# %%
# import libraries
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import numpy as np
import einops
import matplotlib.pyplot as plt
import copy

jkey = jax.random.PRNGKey(0)

# hyper parameters
NB = 64
DS_SCALE = 1.0

# %%
# dataset
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
ds_train = ds_train.batch(NB).prefetch(1)
ds_test = ds_test.batch(NB)
img_size = ds_info.features['image'].shape

def input_preprocessing(x):
    res = np.array(x[...,0]).astype(np.float32)/255.0*2.0 - 1.0
    return res * DS_SCALE

# %%
# hyper parameters
T_limit = 100
SUB_T = 2
# beta_range = [1e-4, 2e-2]
# beta_range = [1e-4, 1e-3]
# beta_range = [1e-4, 5e-3]
beta_range = [1e-1, 5e-1]
beta = jnp.arange(T_limit).astype(jnp.float32)
beta = beta / (T_limit-1) * (beta_range[1] - beta_range[0]) + beta_range[0]
alpha = 1.-beta
alphabar = jnp.cumprod(alpha)

# diffusion model
class DiffusionModel(nn.Module):

    @nn.compact
    def __call__(self, x, t):
        # temb = self.get_timestep_embedding(t, 512)
        temb = (t[...,None] == jnp.arange(512).astype(t.dtype)).astype(jnp.float32)
        temb = nn.Dense(512)(temb)
        temb = nn.Dense(512)(nn.relu(temb))
        original_shape = x.shape
        x = einops.rearrange(x, '... i j -> ... (i j)')
        x = jnp.concatenate([x, temb], axis=-1)
        for i in range(3):
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            if i==1:
                skip = x
        x += skip
        for i in range(2):
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            if i==0:
                skip = x
        x += skip
        for i in range(2):
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            if i==0:
                skip = x
        x += skip
        x = nn.Dense(original_shape[-1] * original_shape[-2])(x)
        x = jnp.reshape(x, original_shape)
        return x
    
    def get_timestep_embedding(self, timesteps, embedding_dim: int):
        timesteps = jnp.array(timesteps)
        half_dim = embedding_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = timesteps.astype(dtype=jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb

for x, y in ds_test.take(1):
    test_input = input_preprocessing(x)
model = DiffusionModel()
params = model.init(jkey, test_input, jnp.arange(1,test_input.shape[0]+1))
pixel_size = test_input.shape[-2:]

# %%
# noizing visualization
def noise_prcess(x0, jkey, timestep):
    epsilon = jax.random.normal(jkey, shape=x0.shape, dtype=jnp.float32)
    alphabar_t = alphabar[timestep-1][...,None,None]
    return  x0 * jnp.sqrt(alphabar_t) + jnp.sqrt(1-alphabar_t) * epsilon

x_list = [test_input]
x_t = test_input
for t in range(1, T_limit+1):
    _, jkey = jax.random.split(jkey)
    x_t = noise_prcess(test_input, jkey, t)
    x_list.append(x_t)
indices = [0, SUB_T, 40, 60, 80, 100]
# indices = np.array(indices) / 100 * T_limit
# indices = indices.astype(np.int32)
plt.figure(figsize=[7,7])
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_list[indices[i]][0] + 1.0)
plt.show()
plt.close()

# %%
# define loss
def loss_func(params, jkey, x):
    xshape = x.shape
    t_rand = jax.random.randint(jkey, minval=1, maxval=SUB_T+1, shape=(xshape[0], ), dtype=jnp.int32)
    _, jkey = jax.random.split(jkey)
    epsilon = jax.random.normal(jkey, shape=xshape, dtype=jnp.float32)

    alphabar_t = alphabar[t_rand-1][...,None,None]
    x_input = jnp.sqrt(alphabar_t) * x + jnp.sqrt(1-alphabar_t)*epsilon
    epsilon_pred = model.apply(params, x_input, t_rand)

    return jnp.mean((epsilon_pred - epsilon)**2)

loss_value_and_grad = jax.value_and_grad(loss_func)

def gen_one_itr(params, x, t_arr, jkey):
    epsilon_pred = model.apply(params, x, t_arr)
    zero_mask = (t_arr > 1).astype(jnp.float32)[...,None,None]
    z = jax.random.normal(jkey, shape=x.shape)*zero_mask  + jnp.zeros(shape=x.shape)*(1-zero_mask)
    sigma = jnp.sqrt(beta[t-1])
    x = 1./jnp.sqrt(alpha[t-1]) * (x - (1.-alpha[t-1])/jnp.sqrt(1.-alphabar[t-1])*epsilon_pred) + sigma*z
    return x

gen_one_itr_jit = jax.jit(gen_one_itr)

def generative(params, jkey, nb):
    
    for x,y in ds_test.take(1):
        x = noise_prcess(input_preprocessing(x[:nb]), jkey, SUB_T)
        origin_input = copy.deepcopy(x)

    # x = jax.random.normal(jkey, shape=(nb, pixel_size[0], pixel_size[1]))
    
    for t in np.arange(1,SUB_T+1)[::-1]:
        _, jkey = jax.random.split(jkey)
        t_arr = jnp.array([t]*nb)
        x = gen_one_itr_jit(params, x, t_arr, jkey)
    return x + 1.0, origin_input + 1.0

# %%
# train init
optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(params)

def train_step(params, opt_state, jkey, x):
    loss, grads = loss_value_and_grad(params, jkey, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

train_step_jit = jax.jit(train_step)
    

#%%
# start train
epoch_no = 100000
for en in range(epoch_no):
    _, jkey = jax.random.split(jkey)
    for x, y in tfds.as_numpy(ds_train):
        x_input = input_preprocessing(x)
        params, opt_state, loss = train_step_jit(params, opt_state, jkey, x_input)
        _, jkey = jax.random.split(jkey)
    
    print(en, loss)
    if en%50 == 0:
        imgs, origin_imgs = generative(params, jkey, nb=6)
        plt.figure(figsize=[18,6])
        for i in range(6):
            plt.subplot(2,6,2*i+1)
            plt.imshow(imgs[i])
            plt.axis('off')
            plt.subplot(2,6,2*i+2)
            plt.imshow(origin_imgs[i])
            plt.axis('off')
        plt.show()
        plt.close()

# %%
