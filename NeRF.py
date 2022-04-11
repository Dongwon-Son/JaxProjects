# %%
# import libraries
import jax.numpy as jnp
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import jax
import util.sdf_util as sdfutil

import flax.linen as nn
import optax
import jaxlie as jl

# TYPE='object' #// 'scene'
TYPE='scene' #// 'scene'
timestep =0.001
PIXEL_SIZE = [64,64]
fov = 60
near = 0.1
if TYPE=='object':
    obj_no = 1
    far = 0.5
    camera_distance = 0.25
else:
    obj_no = 9
    far = 1.0
    camera_distance = 0.4
intrinsic = [PIXEL_SIZE[1], PIXEL_SIZE[0], 
            PIXEL_SIZE[1]*0.5/np.tan(fov*np.pi/180/2), PIXEL_SIZE[0]*0.5/np.tan(fov*np.pi/180/2), 
            PIXEL_SIZE[1]*0.5-0.5, PIXEL_SIZE[0]*0.5-0.5]
NS = 40
NB = 4
# %%
# create scene and objects
p.connect(p.DIRECT)
p.setTimeStep(timestep)
p.setGravity(0,0,-9.81)
if TYPE != 'object':
    p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,0.1]),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[1,1,0.1],
                                                rgbaColor=[0.9,0.2,0.2,1.0]),
                    basePosition=[0,0,-0.05])

# %%
# create meshes
obj_mesh_list = glob.glob('obj_mesh/*/meshes/model.obj')
obj_id_list = []
for i in range(obj_no):
    obj_file_name = random.choice(obj_mesh_list)
    if TYPE == 'object':
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=obj_file_name),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=obj_file_name),
        )
    else:
        quat_rand = np.random.normal(size=[4])
        quat_rand = quat_rand / np.linalg.norm(quat_rand)
        obj_id_list.append(p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=obj_file_name),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=obj_file_name),
            basePosition=np.random.uniform([-0.5,-0.5,0.2],[0.5,0.5,1.5]),
            baseOrientation=quat_rand
        ))

if TYPE != 'object':
    for i in range(int(5/timestep)):
        p.stepSimulation()
# %%
def get_rgb(cam_pos = [0,camera_distance,camera_distance]):

    pm = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=near, farVal=far)
    vm = p.computeViewMatrix(cam_pos,[0,0,0],[0,0,1])

    img_out = p.getCameraImage(*PIXEL_SIZE, viewMatrix=vm, projectionMatrix=pm)
    # img_out = p.getCameraImage(*pixel_size)
    rgb = np.array(img_out[2]).reshape([*PIXEL_SIZE, 4])[...,:3]

    return rgb.astype(np.float32)/255.0

plt.figure()
plt.imshow(get_rgb())
plt.axis('off')
plt.show()

# %%
# creatae NeRF model
class NeRF(nn.Module):

    @nn.compact
    def __call__(self, x, ray_vec):
        out_bound_mask = jnp.array(jnp.concatenate([x >= 1.0, x <= -1.0], axis=-1), dtype=jnp.float32)
        out_bound_mask = jnp.max(out_bound_mask, axis=-1, keepdims=True)
        x = self.positional_embedding(x, 10)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        alpha = jnp.arctan2(ray_vec[...,1], ray_vec[...,0])[...,None]
        beta = jnp.arctan2(ray_vec[...,2], jnp.linalg.norm(ray_vec[...,:2], axis=-1))[...,None]
        ab = self.positional_embedding(jnp.concatenate([alpha/np.pi, beta/(0.5*np.pi)], axis=-1), 4)
        ab = nn.Dense(128)(ab)
        ab = nn.relu(ab)
        
        # x = jnp.concatenate([x, ab], axis=-1)
        for i in range(3):
            x = nn.Dense(128)(x)
            if i != 2:
                # x = nn.relu(x)
                x = nn.sigmoid(x)
        x+=ab
        x = nn.Dense(64)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(4)(x)
        x = nn.sigmoid(x)
        # x = nn.tanh(x) * 0.5 + 0.5
        return (1-out_bound_mask)*x + out_bound_mask

    def positional_embedding(self, x, embedding_size=10):
        pe = []
        for i in range(embedding_size):
            pe.append(jnp.cos((2**i)*np.pi*x))
            pe.append(jnp.sin((2**i)*np.pi*x))
        x = jnp.concatenate(pe, axis=-1)
        return x

def test_nerf_sphere(x, ray_vec):
    sdf_func = sdfutil.sphere_sdf(r=0.2)
    sigma = sdf_func(x - jnp.array([0,0,0.2]))
    sigma = jnp.array(sigma < 0, jnp.float32)
    # sigma = sigma 
    color = jnp.ones_like(x) * jnp.array([0.1,0.9,0.1])
    return jnp.concatenate([color, sigma[...,None]], axis=-1)


# %%
# nerf rendering

def cal_ray_pnts(camera_pos):
    cam_zeta = intrinsic[2:]

    z_axis = -camera_pos
    x_axis = jnp.cross(jnp.array([0,0,1]), z_axis)
    xnorm = jnp.linalg.norm(x_axis, axis=-1, keepdims=True)
    x_axis = x_axis / (1e-7+xnorm)
    y_axis = jnp.cross(z_axis, x_axis)
    view_SO3 = jl.SO3.from_matrix(jnp.stack([x_axis,y_axis,z_axis], axis=-1))
    cam_SE3 = jl.SE3.from_rotation_and_translation(rotation=view_SO3, translation=camera_pos)

    K_mat = jnp.array([[cam_zeta[0], 0, cam_zeta[2]],
                    [0, cam_zeta[1], cam_zeta[3]],
                    [0,0,1]])

    x_grid_idx, y_grid_idx = jnp.meshgrid(jnp.arange(PIXEL_SIZE[1]), jnp.arange(PIXEL_SIZE[0]))
    pixel_pnts = jnp.concatenate([x_grid_idx[...,None], y_grid_idx[...,None], jnp.ones_like(y_grid_idx[...,None])], axis=-1)
    pixel_pnts = jnp.array(pixel_pnts, dtype=jnp.float32)
    K_mat_inv = jnp.linalg.inv(K_mat)
    pixel_pnts = jnp.matmul(K_mat_inv,pixel_pnts[...,None])[...,0]
    rays_s_canonical = pixel_pnts * near
    rays_e_canonical = pixel_pnts * far
    origin_shape = rays_s_canonical.shape

    # cam SE3 transformation
    rays_s = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_s_canonical, (-1,3))), origin_shape)
    rays_e = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_e_canonical, (-1,3))), origin_shape)
    ray_dir = rays_e - rays_s
    ray_len = jnp.linalg.norm(ray_dir, axis=-1)
    ray_dir_normalized = ray_dir/jnp.linalg.norm(ray_dir, axis=-1, keepdims=True)

    # batch ray marching
    ray_sample_pnts = rays_s[...,None,:] + ray_dir[...,None,:] * jnp.arange(NS)[...,None]/(NS-1)
    delta = (ray_len / NS)[...,None]
    ray_dir_tile = jnp.tile(ray_dir_normalized[...,None,:], [1,1,ray_sample_pnts.shape[-2],1])

    return ray_sample_pnts, ray_dir_tile, delta


def nerf_render(nerf_func, camera_pos):
    if len(camera_pos.shape) >= 2:
        ray_sample_pnts, ray_dir_tile, delta = jax.vmap(cal_ray_pnts)(camera_pos)
    else:
        ray_sample_pnts, ray_dir_tile, delta = cal_ray_pnts(camera_pos)
    sd_res = nerf_func(ray_sample_pnts, ray_dir_tile)
    colors, sigmas = sd_res[...,:3], sd_res[...,-1]
    
    sigmas_sums = jax.lax.associative_scan(lambda a,x: a+x, sigmas, axis=-1)
    T = jnp.exp(-sigmas_sums*delta)
    weights = (1-jnp.exp(-sigmas*delta))[...,1:] * T[...,:-1]
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    pixel_colors = jnp.sum(colors[...,1:,:] * weights[...,None], axis=-2)
    return pixel_colors

# %%
# test NeRF rendering
img = nerf_render(test_nerf_sphere, camera_pos=jnp.array([0,0.001,1]))

plt.figure()
plt.imshow(img)
plt.show()

# %%
# model and optimizer initialization
model = NeRF()
jkey = jax.random.PRNGKey(0)
input = jax.random.normal(jkey, shape=[*PIXEL_SIZE, 3])
param = model.init(jkey, input, input)

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(param)

# nerf_render(lambda x, y : model.apply(param, x, y), camera_pos=jnp.array([0,0.001,-1]))

# %%
def loss_func(param, model, cam_pos, img):
    # img_pred = nerf_render(lambda x, y : model.apply(param, x, y), cam_pos)

    ray_sample_pnts, ray_dir_tile, delta = jax.vmap(cal_ray_pnts)(cam_pos)
    sd_res = model.apply(param, ray_sample_pnts, ray_dir_tile)
    colors, sigmas = sd_res[...,:3], sd_res[...,-1]
    
    sigmas_sums = jax.lax.associative_scan(lambda a,x: a+x, sigmas, axis=-1)
    T = jnp.exp(-sigmas_sums*delta)
    weights = (1-jnp.exp(-sigmas*delta))[...,1:] * T[...,:-1]
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    img_pred = jnp.sum(colors[...,1:,:] * weights[...,None], axis=-2)

    return jnp.mean(jnp.sum(jnp.square(img - img_pred), axis=(-1,-2,-3)))

loss_func_value_and_grad = jax.value_and_grad(loss_func)
loss_func_value_and_grad_jit = jax.jit(loss_func_value_and_grad, static_argnums=(1,))

# @jax.jit
def update_step(param, opt_state, cam_pos, img):
    value, grad = loss_func_value_and_grad(param, model, cam_pos, img)
    updates, opt_state = optimizer.update(grad, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state

def update_steup_itr(param, opt_state, cam_pos_b, img_b):
    for _ in range(4):
        for j in range(int(cam_pos_b.shape[0]/NB)):
            cur_data = [dt[NB*j:NB*(j+1)] for dt in [cam_pos_b, img_b]]
            param, opt_state = update_step(param, opt_state, *cur_data)
    return param, opt_state

# update_step_jit = jax.jit(update_step)
update_steup_itr_jit = jax.jit(update_steup_itr)

# %%
# train steps
def data_generation(jkey):
    cam_pos_b = []
    img_b = []
    for _ in range(NB*10):
        jkey, _ = jax.random.split(jkey)
        cam_pos = jax.random.normal(jkey, shape=[3])
        xy_normalize = cam_pos[...,:2]/jnp.linalg.norm(cam_pos[...,:2], axis=-1, keepdims=True)
        cam_pos = jnp.concatenate([xy_normalize*camera_distance, camera_distance*jnp.ones_like(cam_pos[...,-1:])], axis=-1)
        cam_pos_b.append(cam_pos)
        img_b.append(get_rgb(cam_pos))
    cam_pos_b = jnp.stack(cam_pos_b, axis=0)
    img_b = jnp.stack(img_b, axis=0)

    return cam_pos_b, img_b

# %%
# test speed

cam_pos_b, img_b = data_generation(jkey)

# %timeit loss_func_value_and_grad(param, model, cam_pos_b, img_b)
# %timeit loss_func_value_and_grad_jit(param, model, cam_pos_b, img_b)
# update_step_jit(param, opt_state, cam_pos_b, img_b)
# train
# %timeit update_step(param, opt_state, cam_pos_b, img_b)
# %timeit update_step_jit(param, opt_state, cam_pos_b, img_b)

# %%
# star train
import time
for i in range(20000):
    # total_s_t = time.time()
    jkey, _ = jax.random.split(jkey)
    # data generation
    cam_pos_b, img_b = data_generation(jkey)

    dg_end_t = time.time()
    # train
    # s_t2 = time.time()
    param, opt_state = update_steup_itr_jit(param, opt_state, cam_pos_b, img_b)
    # print(dg_end_t - total_s_t, time.time()-s_t2,time.time()- total_s_t)
    # print(i)

    if i % 10 == 0:
        # evaluation
        img_pred = nerf_render(lambda x, y : model.apply(param, x, y), cam_pos_b[0])
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_pred)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img_b[0])
        plt.axis('off')
        plt.show()

# %%
