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
import jaxlie as jl

import util.sdf_util as sdfu

# %% hyper parameters
mesh_list = glob.glob('obj_mesh/*/meshes/*.obj')
NB = 4
NZ = 16
EPOCH = 2000
pick_mesh_dir = random.choice(mesh_list)
# pick_mesh_dir = mesh_list[5]
mesh_legacy = o3d.io.read_triangle_mesh(pick_mesh_dir)
mesh_legacy.compute_vertex_normals()
aabb = mesh_legacy.get_axis_aligned_bounding_box()
mesh_legacy.translate(-aabb.get_center())
mesh_legacy.scale(1/np.max(aabb.get_half_extent())*0.85, center=np.zeros(3))
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
    unbound_pnts = np.random.uniform(np.array([-1.0, -1.0, -1.0]), np.array([1.0,1.0,1.0]), size=[int(UNI_PNTS_N/4),3])
    uniform_pnts = np.random.uniform(aabb.get_min_bound()-0.050, aabb.get_max_bound()+0.050, size=[UNI_PNTS_N,3])
    surface_pcd = mesh_legacy.sample_points_uniformly(number_of_points=SUR_PNTS_N)
    surface_pnts = np.asarray(surface_pcd.points)
    surface_pnts += np.random.normal(size=surface_pnts.shape) * np.min(aabb.get_extent()) * SUR_NOISE_S
    query_points = np.concatenate([unbound_pnts, uniform_pnts, surface_pnts], axis=0)
    signed_distance = scene.compute_signed_distance(query_points.astype(np.float32))
    signed_distance = signed_distance.numpy().astype(np.float32)

    ## test sphere
    # query_points = np.random.uniform(-1.5, 1.5, size=[SUR_PNTS_N,3])
    # sdf_func = sdfu.sphere_sdf(r=1.0)
    # signed_distance = sdf_func(query_points)

    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_points[np.where(occupancy==1)]))
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.1, origin=[0,0,0])
    # draw([pcd, mesh_frame])

    return query_points, signed_distance, np.asarray(surface_pcd.points), np.asarray(surface_pcd.normals)
    
# %%
# network design
class DecSDF(nn.Module):
    clip_value: float

    @nn.compact
    def __call__(self, x):
        # frequency expand
        out_bound_mask = jnp.array(jnp.concatenate([x >= 1.0, x <= -1.0], axis=-1), dtype=jnp.float32)
        out_bound_mask = jnp.max(out_bound_mask, axis=-1)
        fx = []
        for i in range(10):
            fx.append((jnp.cos((2**i)*jnp.pi*x)))
            fx.append((jnp.sin((2**i)*jnp.pi*x)))
        x = jnp.concatenate(fx, axis=-1)
        for _ in range(3):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x) * self.clip_value
        x = jnp.squeeze(x, axis=-1)
        return (1-out_bound_mask)*x + out_bound_mask

# %%
# define SDFVAE loss
def sdfvae_loss(params, jkey, dec, data):
    dec_param = params
    
    sdf_pred = dec.apply(dec_param, data[0])
    clip_value = dec.clip_value
    sdf_real = jnp.clip(data[1], -clip_value, clip_value)
    sq_diff = jnp.square(sdf_real - sdf_pred)
    rec_loss = jnp.mean(sq_diff, axis=-1)

    # grad_func = jax.grad(lambda x : dec.apply(dec_param, x))
    grad_func = jax.grad(functools.partial(dec.apply, dec_param))
    origin_shape = data[2].shape
    surface_grad = jnp.reshape(jax.vmap(grad_func)(jnp.reshape(data[2], (-1,3))), origin_shape)
    normal_loss = jnp.mean(jnp.square(surface_grad - data[3]), axis=(-1,-2))

    # surface_grad = surface_grad / jnp.maximum(jnp.linalg.norm(surface_grad, axis=-1, keepdims=True), 1e-6)
    # normal_loss = (1-jnp.sum(surface_grad * data[3], axis=-1))
    # normal_loss = jnp.mean(normal_loss, axis=-1)
    
    return jnp.mean(rec_loss + 0.001*normal_loss)

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
dec = DecSDF(1.0)

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
    view_SO3 = jl.SO3.from_x_radians(jnp.pi+jnp.pi/8)@jl.SO3.from_y_radians(jnp.pi/8)
    cam_SE3 = jl.SE3.from_rotation(rotation=view_SO3) @ \
        jl.SE3.from_rotation_and_translation(rotation=jl.SO3(jnp.array([1,0,0,0],dtype=jnp.float32)), translation=jnp.array([0,0,-2.0]))
    light_position = jnp.array([-1.5,1.0,2.0])

    depth_gen = sdfu.sdf_renderer_exact(sdf_func, cam_SE3=cam_SE3, far=3, light_position=light_position)
    plt.figure()
    plt.imshow(depth_gen)
    plt.axis('off')
    plt.show()


# visualize_sdf(params, jkey, data)

# %% init optimizer
optimizer = optax.adam(learning_rate=5e-4)
opt_state = optimizer.init(params)

# define train func
@jax.jit
def train_step(jkey, params, data, opt_state):
    loss, grads = loss_value_and_grad_func(params, jkey, dec, data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

for ep in range(EPOCH):
    for _ in range(50):
        _, jkey = jax.random.split(jkey)
        data = batch_data_gen()
        loss, params, opt_state = train_step(jkey, params, data, opt_state)
    if ep%10 == 0 :
        visualize_sdf(params, jkey, data)
        print(loss)

# %%