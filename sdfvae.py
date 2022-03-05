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
mesh_list = glob.glob('obj_mesh/*/meshes/*.obj')[:20]
pic_dict = {}
sdf_label_dict = {}
scene_dict = {}
NB = 4
NZ = 16
EPOCH = 10000

UNI_PNTS_N = 8000
SUR_PNTS_N = 30000
SUR_NOISE_S = 0.02

# %%
def gen_one_data():
    # mesh init
    pick_mesh_dir = random.choice(mesh_list)
    # pick_mesh_dir = mesh_list[5]
    # if pick_mesh_dir in pic_dict.keys():
    #     return pic_dict[pick_mesh_dir], sdf_label_dict[pick_mesh_dir]

    if pick_mesh_dir in pic_dict.keys():
        img_res = pic_dict[pick_mesh_dir]
    else:
        # init pybullet scene
        obj_id = p.createMultiBody(baseVisualShapeIndex=
                                    p.createVisualShape(p.GEOM_MESH, fileName=pick_mesh_dir))
        img_size = [64,64]
        img_res = []
        for _ in range(3):
            eyep = np.random.normal(size=[3])
            eyep = eyep / np.linalg.norm(eyep) * 0.4
            upv = np.random.normal(size=[3])
            upv = upv / np.linalg.norm(upv)
            vM = p.computeViewMatrix(cameraEyePosition=eyep, cameraTargetPosition=[0,0,0], cameraUpVector=upv)
            pM = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.1, farVal=3)
            imgres = p.getCameraImage(img_size[1],img_size[0], viewMatrix=vM, projectionMatrix=pM, flags=p.ER_NO_SEGMENTATION_MASK)

            img = imgres[2]
            img = np.asarray(img).reshape(*img_size, 4)
            img_res.append(img[None,...,:3])
        img_res = (np.concatenate(img_res, axis=0)/255.0).astype(np.float32)
        # plt.figure()
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        p.removeBody(obj_id)
        pic_dict[pick_mesh_dir] = img_res

    if pick_mesh_dir in scene_dict.keys():
        mesh_legacy, scene = scene_dict[pick_mesh_dir]
    else:
        # pnts samples
        mesh_legacy = o3d.io.read_triangle_mesh(pick_mesh_dir)
        mesh_legacy.compute_vertex_normals()
        # draw(mesh_legacy)

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        scene_dict[pick_mesh_dir] = (mesh_legacy, scene)
    aabb = mesh_legacy.get_axis_aligned_bounding_box()

    unbound_pnts = np.random.uniform(np.array([-0.4, -0.4, -0.4]), np.array([0.4,0.4,0.4]), size=[int(UNI_PNTS_N/4),3])
    uniform_pnts = np.random.uniform(aabb.get_min_bound()-0.05, aabb.get_max_bound()+0.05, size=[UNI_PNTS_N,3])
    surface_pnts = np.asarray(mesh_legacy.sample_points_uniformly(number_of_points=SUR_PNTS_N).points)
    surface_pnts += np.random.normal(size=surface_pnts.shape) * np.min(aabb.get_extent()) * SUR_NOISE_S
    query_points = np.concatenate([unbound_pnts, uniform_pnts, surface_pnts], axis=0)
    occupancy = scene.compute_occupancy(query_points.astype(np.float32))
    occupancy = occupancy.numpy().astype(np.float32)
    # random_order = np.arange(occupancy.shape[0])
    # np.random.shuffle(random_order)
    # occupancy = occupancy[random_order]
    # query_points = query_points[random_order]
    sdf_label_res = (query_points, occupancy)

    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_points[np.where(occupancy==1)]))
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.1, origin=[0,0,0])
    # draw([pcd, mesh_frame])
    # sdf_label_dict[pick_mesh_dir] = sdf_label_res
    return img_res, sdf_label_res, mesh_list.index(pick_mesh_dir)
    
# %%
# network design
class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        origin_shape = x.shape
        x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        x = nn.Conv(8, kernel_size=(7,7), strides=(2,2), padding='VALID')(x)
        x = nn.relu(x)
        # x = nn.Conv(16, kernel_size=(5,5), strides=(2,2), padding='VALID')(x)
        # x = nn.relu(x)
        x = jnp.reshape(x, (origin_shape[0], origin_shape[1], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = jnp.sum(x, axis=1)
        x = nn.Dense(2*NZ)(x)
        mean, logsigma = jnp.split(x, 2, axis=-1)
        return mean, logsigma

class DecSDF(nn.Module):
    @nn.compact
    def __call__(self, x, z):
        # frequency expand
        fx = []
        for i in range(10):
            fx.append(jnp.cos((2**i)*np.pi*x))
            fx.append(jnp.sin((2**i)*np.pi*x))
        x = jnp.concatenate(fx, axis=-1)
        z = jnp.expand_dims(z, axis=-2)
        z_tile = jnp.tile(z, (1, x.shape[1], 1))
        x = jnp.concatenate([x, z_tile], axis=-1)
        for _ in range(4):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        x = jnp.squeeze(x, axis=-1)
        return x

# %%
# define SDFVAE loss
def sdfvae_loss(params, jkey, enc, dec, data):
    enc_param = params[0]
    dec_param = params[1]
    z_hat_mean, z_hat_logsigma = enc.apply(enc_param, data[0])
    z_hat_sigma = jnp.exp(z_hat_logsigma)
    noise = jax.random.normal(jkey, shape=z_hat_mean.shape)
    z_hat = z_hat_mean + noise * z_hat_sigma

    occ_pred_dec = dec.apply(dec_param, data[1][0], jax.lax.stop_gradient(z_hat))
    occ_pred_dec = jnp.clip(occ_pred_dec, 1e-6, 1-1e-6)
    occ_pred_enc = dec.apply(jax.lax.stop_gradient(dec_param), data[1][0], z_hat)
    occ_pred_enc = jnp.clip(occ_pred_enc, 1e-6, 1-1e-6)
    occ_real = data[1][1]
    
    def cal_loss(y_pred, y_real):
        # binary cross entropy
        # rec_loss = occ_real * jnp.log(occ_pred) + (1-occ_real) * jnp.log(1-occ_pred)
        # rec_loss = -rec_loss
        # focal loss
        p = y_real*y_pred + (1-y_real) * (1-y_pred)
        alpha = 0.5
        gamma = 2
        rec_loss = -alpha*(1-p)**gamma*jnp.log(p)
        return  np.mean(rec_loss, axis=-1)
        

    z_hat_var = jnp.square(z_hat_sigma)
    reg_loss = 0.5*(jnp.square(z_hat_mean) + z_hat_var - jnp.log(z_hat_var) -1)
    reg_loss = np.mean(reg_loss, axis=-1)
    
    return jnp.mean(cal_loss(occ_pred_dec, occ_real) + 
                    cal_loss(occ_pred_enc, occ_real) + 
                            0.01*reg_loss)
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

p.connect(p.DIRECT)
data = batch_data_gen()

# %%
# init networks
enc = Encoder()
dec = DecSDF()

jkey =jax.random.PRNGKey(0)
enc_param = enc.init(jkey, data[0])
_, jkey =jax.random.split(jkey)
z = jax.random.normal(jkey,shape=(NB, NZ))
dec_param = dec.init(jkey, data[1][0], z)
params = (enc_param, dec_param)

# %%
# test
sdfvae_loss(params, jkey, enc, dec, data)

# %%
# visualization
def visualize_sdf(params, jkey, data):
    # reconstruction
    z_hat_mean, z_hat_logsigma = enc.apply(params[0], data[0])
    z_hat_sigma = jnp.exp(z_hat_logsigma)
    z_hat = z_hat_mean + z_hat_sigma*jax.random.normal(jkey,shape=z_hat_mean.shape)

    def sdf_func(x, z_hat_, idx):
        origin_shape = x.shape
        x_expanded = jnp.reshape(x, (-1, 3))
        x_expanded = x_expanded[None,...]
        occ = dec.apply(params[1], x_expanded, z_hat_[idx:idx+1])
        return jnp.reshape(occ, origin_shape[:-1])

    _, jkey = jax.random.split(jkey)
    z_noise = jax.random.normal(jkey, shape=z_hat_sigma.shape)

    plt.figure(figsize=[20,20])
    for i in range(4):
        plt.subplot(4,4,4*i+1)
        plt.imshow(data[0][i,0])
        plt.axis('off')
        plt.subplot(4,4,4*i+2)
        plt.imshow(data[0][i,1])
        plt.axis('off')
        plt.subplot(4,4,4*i+3)
        plt.imshow(sdfu.visualize_sdf(lambda x :sdf_func(x, z_hat, i)))
        plt.axis('off')
        plt.subplot(4,4,4*i+4)
        plt.imshow(sdfu.visualize_sdf(lambda x :sdf_func(x, z_noise, i)))
        plt.axis('off')
    plt.show()

visualize_sdf(params, jkey, data)

# %% 
# start train
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)

# define train func
@jax.jit
def train_step(jkey, params, data, opt_state):
    _, jkey = jax.random.split(jkey)
    loss, grads = loss_value_and_grad_func(params, jkey, enc, dec, data)
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