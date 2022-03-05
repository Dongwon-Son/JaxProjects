# %%
# import libraries
import jax.numpy as jnp
import jax
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import flax.linen as nn
from open3d.web_visualizer import draw
import random
import glob
import optax
import functools

import util.sdf_util as sdfu

# %% hyper parameters
mesh_list = glob.glob('obj_mesh/*/meshes/*.obj')
NB = 4
NZ = 16
EPOCH = 2000
# pick_mesh_dir = random.choice(mesh_list)
pick_mesh_dir = mesh_list[5]
mesh_legacy = o3d.io.read_triangle_mesh(pick_mesh_dir)
mesh_legacy.compute_vertex_normals()
aabb = mesh_legacy.get_axis_aligned_bounding_box()
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)
UNI_PNTS_N = 5000
SUR_PNTS_N = 30000
SUR_NOISE_S = 0.02
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0,0,0])
draw([mesh_legacy, mesh_frame])

# %%
def gen_one_data():
    # pnts samples
    uniform_pnts = np.random.uniform(aabb.get_min_bound()-0.050, aabb.get_max_bound()+0.050, size=[UNI_PNTS_N,3])
    surface_pnts = np.asarray(mesh_legacy.sample_points_uniformly(number_of_points=SUR_PNTS_N).points)
    surface_pnts += np.random.normal(size=surface_pnts.shape) * np.min(aabb.get_extent()) * SUR_NOISE_S
    query_points = np.concatenate([uniform_pnts, surface_pnts], axis=0)
    occupancy = scene.compute_occupancy(query_points.astype(np.float32))
    occupancy = occupancy.numpy().astype(np.float32)

    ## test sphere
    # query_points = np.random.uniform(-0.25, 0.25, size=[SUR_PNTS_N,3])
    # def sphere_sdf(center, r, x):
    #     return jnp.array((jnp.sum((x - center)**2, axis=-1) < r**2), dtype=jnp.float32)
    # occupancy = sphere_sdf(np.array([0,0,0]), 0.1, query_points)

    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_points[np.where(occupancy==1)]))
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.1, origin=[0,0,0])
    # draw([pcd, mesh_frame])

    return query_points, occupancy
    
# %%
# network design
class DecSDF(nn.Module):
    @nn.compact
    def __call__(self, x):
        # frequency expand
        fx = []
        for i in range(10):
            fx.append(jnp.cos((2**i)*np.pi*x))
            fx.append(jnp.sin((2**i)*np.pi*x))
        x = jnp.concatenate(fx, axis=-1)
        for _ in range(3):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        x = jnp.squeeze(x, axis=-1)
        return x

# %%
# define SDFVAE loss
def sdfvae_loss(params, jkey, dec, data):
    dec_param = params
    
    occ_pred = dec.apply(dec_param, data[0])
    occ_real = data[1]
    
    occ_pred = jnp.clip(occ_pred, 1e-7, 1.0-1e-7)

    # binary cross entropy
    # rec_loss = occ_real * jnp.log(occ_pred) + (1-occ_real) * jnp.log(1-occ_pred)
    # rec_loss = -rec_loss
    # focal loss
    p = occ_real*occ_pred + (1-occ_real) * (1-occ_pred)
    alpha = 0.5
    gamma = 2
    rec_loss = -alpha*((1-p)**gamma)*jnp.log(p)
    rec_loss = np.mean(rec_loss, axis=-1)
    
    return jnp.mean(rec_loss)
loss_value_and_grad_func = jax.value_and_grad(sdfvae_loss)

# %%
# define train hyper parameters
NB = 4

# %%
# define batch data generation function
def batch_data_gen():
    res = []
    for i in range(NB):
        res.append(gen_one_data())
    batch_data = jax.tree_map(lambda *x: jnp.stack(x), *res)
    return batch_data

data = batch_data_gen()

# %%
# init networks
dec = DecSDF()

jkey =jax.random.PRNGKey(0)
dec_param = dec.init(jkey, data[0])
params = dec_param

# %%
# test
sdfvae_loss(params, jkey, dec, data)

# %%
# visualization
def visualize_sdf(params, jkey, data):
    def sdf_func(x):
        origin_shape = x.shape
        x_expanded = jnp.reshape(x, (-1, 3))
        x_expanded = x_expanded[None,...]
        occ = dec.apply(params,x_expanded)
        return jnp.reshape(occ, origin_shape[:-1])
    depth_gen = sdfu.visualize_sdf(sdf_func)
    plt.figure()
    plt.imshow(depth_gen)
    plt.axis('off')
    plt.show()


visualize_sdf(params, jkey, data)

# %% init optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# define train func
@jax.jit
def train_step(jkey, params, data, opt_state):
    _, jkey = jax.random.split(jkey)
    loss, grads = loss_value_and_grad_func(params, jkey, dec, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

for ep in range(EPOCH):
    for _ in range(50):
        data = batch_data_gen()
        loss, params, opt_state = train_step(jkey, params, data, opt_state)
    if ep%10 == 0 :
        visualize_sdf(params, jkey, data)
        print(loss)

# %%