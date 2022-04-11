# %%
import jax
import flax.linen as nn
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

# %%
# define ViT networks
GRID_SIZE = [4,4]

class ViT(nn.Module):
    base_dim : int = 64

    def setup(self):
        self.k_nn = nn.Dense(self.base_dim)
        self.q_nn = nn.Dense(self.base_dim)
        self.v_nn = nn.Dense(self.base_dim)
        self.in_mlp = nn.Dense(self.base_dim)
        self.final_mlp = nn.Dense(10)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()

    def __call__(self, x):
        x = self.input_vectorize(x)
        for _ in range(2):
            x = self.attention(x)
        logit = self.final_mlp(jnp.reshape(x, (x.shape[0], -1)))
        return logit

    def attention(self, x):
        nx = self.layernorm1(x)
        k = self.k_nn(nx)
        q = self.q_nn(nx)
        v = self.v_nn(nx)
        # v = nn.relu(v) # (v, n)
        alpha = jnp.sum(k[...,None,:] * q[...,None,:,:], axis=-1, keepdims=True) # (v, v, 1)
        alpha = nn.softmax(alpha/jnp.sqrt(self.base_dim), axis=-3)
        x = jnp.sum(v[...,None,:] * alpha, axis=-3) + x
        x = self.in_mlp(self.layernorm2(x)) + x
        return x

    @nn.compact
    def input_vectorize(self, x):
        patch_shape = (int(x.shape[-3]/GRID_SIZE[0]), int(x.shape[-2]/GRID_SIZE[1]))
        x = jnp.reshape(x, x.shape[:-3] + (GRID_SIZE[0], patch_shape[0], GRID_SIZE[1], patch_shape[1]))
        x = jnp.transpose(x, (0,1,3,2,4))
        x = jnp.reshape(x, x.shape[:-4] + (GRID_SIZE[0]*GRID_SIZE[1], patch_shape[0]*patch_shape[1]))
        
        # positional encoding
        pe = []
        seq = jnp.arange(GRID_SIZE[0] * GRID_SIZE[1]) # (NB, NV(16), 1)
        for i in range(4):
            pe.append(jnp.cos(seq*1/10000**(i/4))[None,...,None])
            pe.append(jnp.sin(seq*1/10000**(i/4))[None,...,None])
        pe = jnp.concatenate(pe, axis=-1)
        pe = jnp.tile(pe, (x.shape[0], 1, 1)) # (NB, NV(16), 2*4)
        x = jnp.concatenate([x, pe], axis=-1)
        
        x = nn.Dense(self.base_dim)(x)
        x = nn.relu(x)

        return x

# simple CNN model
class MLP(nn.Module):
    base_dim : int = 64
    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1))
        for _ in range(3):
            x = nn.Dense(self.base_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(8, (7,7), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(8, (5,5), padding='VALID')(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x


# %%
# define loss
def classification_loss(param, model, x, y_real):
    y_pred_logit = model.apply(param, x)
    y_real_onehot = jax.nn.one_hot(y_real, num_classes=10)
    y_real_onehot = jnp.array(y_real_onehot, jnp.float32)

    cross_entropy = -y_real_onehot*jax.nn.log_softmax(y_pred_logit)
    cross_entropy = jnp.sum(cross_entropy, axis=-1)

    return jnp.mean(cross_entropy)

loss_value_and_grad_func = jax.value_and_grad(classification_loss)

# %%
NB = 16

# load dataset
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
# init model
vit_model = ViT()
# vit_model = MLP()
# vit_model = CNN()
jkey = jax.random.PRNGKey(0)
test_input = jnp.array(test_data[0], dtype=jnp.float32)/255.0
vit_param = vit_model.init(jkey, test_input)
vit_pred_jit = jax.jit(vit_model.apply)

# %%
# model test
# parameter count
flatten_param = jax.tree_util.tree_flatten(vit_param)
total_shape = jnp.array([jnp.prod(jnp.array(fp.shape)) for fp in flatten_param[0]])
total_param_cnt = jnp.sum(total_shape)
print("parameter cnt : {}".format(total_param_cnt))

vit_pred_jit(vit_param, test_input)
%timeit vit_pred_jit(vit_param, test_input)

# %%
# train func
optimizer = optax.adam(5e-4)
opt_state = optimizer.init(vit_param)

@jax.jit
def train_step(param, opt_state, data):
    value, grad = loss_value_and_grad_func(param, vit_model, jnp.array(data[0], dtype=jnp.float32)/255.0, jnp.array(data[1], jnp.int32))
    update, opt_state = optimizer.update(grad, opt_state)
    param = optax.apply_updates(param, update)

    return value, param, opt_state

# %%
# define visualization
def evaluation(param, data):
    preds = vit_model.apply(param, jnp.array(data[0], dtype=jnp.float32)/255.0)
    plt.figure()
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(data[0][i])
        plt.axis('off')
    plt.show()
    print("pred label : {}".format(jnp.argmax(preds, axis=-1)))


# %%
# start iteration

for _ in range(10000):
    for data in tfds.as_numpy(ds_train):
        value, vit_param, opt_state = train_step(vit_param, opt_state, data)
    
    print("loss : {}".format(value))
    for data in ds_test.take(1):
        evaluation(vit_param, data)
    
    false_cnt = 0
    total_cnt = 0
    for data in tfds.as_numpy(ds_test):
        preds = vit_pred_jit(vit_param, jnp.array(data[0], dtype=jnp.float32)/255.0)
        false_cnt += jnp.sum(jnp.minimum(jnp.abs(jnp.argmax(preds, axis=-1) - data[1]), 1))
        total_cnt += preds.shape[0]
    print("acc : {}".format(1 - false_cnt / total_cnt))
# %%
