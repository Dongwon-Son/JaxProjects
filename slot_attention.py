# %%
# import libraries
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import numpy as np
import einops
import matplotlib.pyplot as plt
import pybullet as p
import itertools
import pickle

import util.transform_util as tutil

jkey = jax.random.PRNGKey(0)

# hyper parameters
OBJ_NO=4
BATCH_SIZE=64
# %%
# dataset - circle square // set prediction

def get_rgb():
    cam_pos = [0,0,12.0]
    PIXEL_SIZE = [48,48]
    fov = 2
    near = 5.0
    far = 15.0
    pm = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=near, farVal=far)
    vm = p.computeViewMatrix(cam_pos,[0,0,0],[0,1,0])

    img_out = p.getCameraImage(*PIXEL_SIZE, viewMatrix=vm, projectionMatrix=pm, shadow=1)
    rgb = np.array(img_out[2]).reshape([*PIXEL_SIZE, 4])[...,:3]
    return rgb


def data_generation(obj_id_list, type_list):
    pos_list = []
    for oid in obj_id_list:
        pos = np.random.uniform(-0.18, 0.18, size=3)
        pos[-1] = 0
        p.resetBasePositionAndOrientation(oid, pos, [0,0,0,1])
        pos_list.append(pos[:2])

    return get_rgb(), np.concatenate([np.expand_dims(type_list,axis=-1), pos_list], axis=-1)

def batch_data(batch_size=32):
    p.resetSimulation()
    obj_id_list = []
    type_list = []

    hext = 0.03
    gen_obj_no = 15
    for _ in range(gen_obj_no):
        if np.random.uniform(0, 1) > 0.5:
            obj_id_list.append(p.createMultiBody(baseMass=0,
                            basePosition=[100,0,0],
                            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[hext,hext,hext],
                                                                rgbaColor=np.random.uniform(0,1,size=4)),
                            ))
            type_list.append(0)
        else:
            obj_id_list.append(p.createMultiBody(baseMass=0,
                            basePosition=[100,0,0],
                            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=hext,
                                                                rgbaColor=np.random.uniform(0,1,size=4)),
                            ))
            type_list.append(1)
    obj_id_list = np.array(obj_id_list)
    type_list = np.array(type_list)
    rgb_list = []
    label_list = []
    for _ in range(batch_size):
        idx = np.random.choice(np.arange(gen_obj_no), size=OBJ_NO, replace=False)
        rgb, label = data_generation(obj_id_list[idx], type_list[idx])
        rgb_list.append(rgb)
        label_list.append(label)
        for oid in obj_id_list[idx]:
            p.resetBasePositionAndOrientation(oid, [100,0,0], [0,0,0,1])
    return np.stack(rgb_list, axis=0).astype(np.float32)/255, np.stack(label_list, axis=0).astype(np.float32)


p.connect(p.DIRECT)
# %timeit batch_data()

# %%
# define model
class SlotAttention(nn.Module):
    sa_nf:int=64
    nslot:int=OBJ_NO

    def setup(self):
        self.input_ln = nn.LayerNorm()
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()
        self.k = nn.Dense(self.sa_nf, use_bias=False)
        self.q = nn.Dense(self.sa_nf, use_bias=False)
        self.v = nn.Dense(self.sa_nf, use_bias=False)
        self.gru = nn.GRUCell()
        self.input_mlp = nn.Sequential([nn.Dense(128), 
                            nn.relu, 
                            nn.Dense(128)])
        self.mlps = nn.Sequential([nn.Dense(128), 
                            nn.relu, 
                            nn.Dense(self.sa_nf)])

    @nn.compact
    def __call__(self, rgb, jkey):
        '''
        rgb, intrinsic : (NB, NV, ...)
        vm_cg : (NB, NV, 4, 4)
        '''
        rgb
        rgb_origin = rgb
        pixel_size = rgb.shape[-3:-1]

        # rgb features
        rgb = rgb.reshape([-1,*pixel_size,3])
        rgb = nn.Conv(8, [7,7], padding='SAME')(rgb)
        rgb = nn.relu(rgb)
        rgb = nn.Conv(8, [5,5], padding='SAME')(rgb)
        rgb = nn.relu(rgb)
        rgb = einops.rearrange(rgb, '... (t1 i) (t2 j) k -> ... t1 t2 (i j k)', t1=8, t2=8)
        rgb = nn.Conv(rgb.shape[-1], [1,1], padding='SAME')(rgb)
        rgb = nn.relu(rgb)
        patch_size = rgb.shape[-3:-1]

        ## soft position embedding
        ranges = [jnp.linspace(0., 1., num=res) for res in patch_size]
        grid = jnp.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = jnp.stack(grid, axis=-1)
        grid = jnp.reshape(grid, [patch_size[0], patch_size[1], -1])
        grid = jnp.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        grid = jnp.concatenate([grid, 1.0 - grid], axis=-1)
        rgb += nn.Dense(rgb.shape[-1])(grid)

        inputs = rgb.reshape((rgb.shape[0], -1, rgb.shape[-1]))
        inputs = self.input_mlp(self.input_ln(inputs))

        # slot attention
        outer_shape = inputs.shape[:-2]
        in_nf = float(inputs.shape[-1])
        inputs = self.ln1(inputs)
        k, v = self.k(inputs), self.v(inputs)
        init_mu = self.param('init_mu', nn.initializers.glorot_uniform(), (1, self.sa_nf), jnp.float32)
        init_log_sigma = self.param('init_sigma', nn.initializers.glorot_uniform(), (1, self.sa_nf), jnp.float32)
        init_sigma = jnp.exp(init_log_sigma)
        slots = init_mu + jax.random.normal(jkey, shape=outer_shape+(self.nslot, self.sa_nf), dtype=jnp.float32) * init_sigma
        for _ in range(3):
            prev_slots = slots
            slots = self.ln2(slots)
            attn = nn.softmax((in_nf**(-0.5)) * jnp.einsum('...ij,...sj->...is', k, self.q(slots)), axis=-1)
            attn += 1e-7
            attn /= jnp.sum(attn, axis=-2, keepdims=True)
            updates = jnp.einsum('...is,...ij->...sj', attn, v)
            slots, _ = self.gru(prev_slots, updates)
            slots += self.mlps(self.ln3(slots))

        out = slots
        for _ in range(2):
            out = nn.Dense(64)(out)
            out = nn.relu(out)
        out = nn.Dense(3)(out)
        out = nn.tanh(out) + jnp.array([1.0, 0., 0.])
        out *= jnp.array([0.5, 0.2, 0.2])

        return out

init_rgb, init_label = batch_data(BATCH_SIZE)

model = SlotAttention()
param = model.init(jkey, init_rgb, jkey)
out = model.apply(param, init_rgb, jkey)
# %%
# define loss
def loss_func(param, jkey, x, y):
    yp = model.apply(param, x, jkey)

    l2loss = []
    for idx in itertools.permutations(np.arange(OBJ_NO), OBJ_NO):
        l2loss.append(jnp.sum(jnp.array([1.,25.,25.])*(yp - y[...,idx,:])**2, axis=(-1,-2)))
    l2loss = jnp.min(jnp.stack(l2loss, axis=-1), axis=-1)

    return jnp.mean(l2loss)

loss_value_and_grad = jax.value_and_grad(loss_func)

loss_func(param, jkey, init_rgb, init_label)

# %%
# train step

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(param)

def train_step(param, opt_state, jkey, x, y):
    loss, grads = loss_value_and_grad(param, jkey, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state, loss

train_step_jit = jax.jit(train_step)

def visualize_preds(param, x, jkey):
    yp = model.apply(param, x, jkey)

    plt.figure(figsize=[10,10])
    for i in range(8):
        p.resetSimulation()
        obj_id_list = []
        hext = 0.030
        for j in range(yp.shape[1]):
            pos = np.concatenate([yp[i,j,1:3], [0.]], axis=-1)
            if yp[i,j,0] < 0.5:
                obj_id_list.append(p.createMultiBody(baseMass=0,
                                basePosition=pos,
                                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[hext,hext,hext],
                                                                    rgbaColor=np.random.uniform(0,1,size=4)),
                                ))
            else:
                obj_id_list.append(p.createMultiBody(baseMass=0,
                                basePosition=pos,
                                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=hext,
                                                                    rgbaColor=np.random.uniform(0,1,size=4)),
                                ))

        plt.subplot(4,4,2*i+1)
        plt.imshow(get_rgb())
        plt.axis('off')
        plt.subplot(4,4,2*i+2)
        plt.imshow(x[i])
        plt.axis('off')
    plt.show()

# %%
# load param and visualization
with open('param.pickle','rb') as f:
    param = pickle.load(f)

visualize_preds(param, init_rgb, jkey)

# %%
# start train
for i in range(10000000):
    _, jkey = jax.random.split(jkey)
    x, y = batch_data(BATCH_SIZE)
    param, opt_state, loss = train_step_jit(param, opt_state, jkey, x, y)

    if i%4000 == 0:
        print(loss)
        visualize_preds(param, x, jkey)
        with open('param.pickle', 'wb') as f:
            pickle.dump(param, f)

# %%
