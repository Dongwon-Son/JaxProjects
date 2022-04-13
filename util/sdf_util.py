# %%
import jax.numpy as jnp
import numpy as np
import jax
import jaxlie as jl
from functools import partial
import time
import transform_util as tutil

def visualize_sdf(sdf_func):
    NS = 80
    img_size = (60,60)
    x_grid, y_grid = jnp.meshgrid(jnp.arange(img_size[1]),jnp.arange(img_size[0]))

    xy_grid = 0.4*jnp.stack([x_grid/(img_size[1]-1)-0.5 , -(y_grid/(img_size[0]-1)-0.5)], axis=-1)

    xy_grid_flat = jnp.reshape(xy_grid, (img_size[0]*img_size[1],2))
    rays_s = jnp.concatenate([xy_grid_flat, jnp.ones_like(xy_grid_flat[...,0:1])*0.5], axis=-1)
    rays_e = jnp.concatenate([xy_grid_flat, -jnp.ones_like(xy_grid_flat[...,0:1])*0.01], axis=-1)
    rays = jnp.concatenate([rays_s, rays_e], axis=-1)

    # ray to depth
    ray_dir = rays[...,3:] - rays[...,:3]
    ray_sample_pnts = rays[...,None,:3] + ray_dir[...,None,:] * jnp.arange(NS)[...,None]/NS
    sd_res = sdf_func(ray_sample_pnts)
    rev_sd_res = 1-sd_res
    rev_sd_scan = jax.lax.associative_scan(lambda a,x: a*x, rev_sd_res, axis=-1)
    rev_sd_scan = jnp.concatenate([jnp.ones_like(rev_sd_scan[...,-1:]), rev_sd_scan], axis=-1)
    depth_arr = jnp.linalg.norm(rays[...,3:] - rays[...,:3], axis=-1, keepdims=True) * jnp.arange(NS+1)/NS
    depth_dist = rev_sd_scan * jnp.concatenate([sd_res, jnp.ones_like(sd_res[...,-1:])], axis=-1)
    depth_dist = depth_dist / (jnp.sum(depth_dist, axis=-1, keepdims=True)+ 1e-8)
    depth = jnp.sum(depth_dist*depth_arr, axis=-1)
    depth = jnp.reshape(depth, img_size)
    return depth

@partial(jax.jit, static_argnums=(0,1,2))
def sdf_renderer(sdf_func, depth_out=False, cam_SE3=None, color=jnp.array([230/255, 9/255, 101/255]), background_color=jnp.array([1,1,1])):
    NS = 2000
    PIXEL_SIZE =  [80, 80]
    def vec_normalize(vec):
        return vec/(jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)

    if cam_SE3 is None:
        view_SO3 = jl.SO3.from_x_radians(jnp.pi/6)@jl.SO3.from_y_radians(jnp.pi/6)
        cam_SE3 = jl.SE3.from_rotation(rotation=view_SO3) @ \
                jl.SE3.from_rotation_and_translation(rotation=jl.SO3(jnp.array([1,0,0,0],dtype=jnp.float32)), translation=jnp.array([0,0,0.8]))

    # pixel sampling
    ray_distance = 2.0
    x_grid, y_grid = jnp.meshgrid(jnp.arange(PIXEL_SIZE[1]), jnp.arange(PIXEL_SIZE[0]))
    x_grid_centered = (x_grid - 0.5*(PIXEL_SIZE[1]-1))/(PIXEL_SIZE[1]-1)
    y_mesh_centered = -(y_grid - 0.5*(PIXEL_SIZE[0]-1))/(PIXEL_SIZE[0]-1)
    rays_s_canonical = jnp.stack([x_grid_centered, y_mesh_centered, jnp.zeros_like(y_mesh_centered)], axis=-1)
    rays_e_canonical = jnp.stack([x_grid_centered, y_mesh_centered, -ray_distance*jnp.ones_like(y_mesh_centered)], axis=-1)
    origin_shape = rays_s_canonical.shape

    # cam SE3 transformation
    rays_s = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_s_canonical, (-1,3))), origin_shape)
    rays_e = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_e_canonical, (-1,3))), origin_shape)

    # ray to depth
    ray_dir = rays_e - rays_s
    ray_dir_normalized = ray_dir/jnp.linalg.norm(ray_dir, axis=-1, keepdims=True)
    ray_sample_pnts = rays_s[...,None,:] + ray_dir[...,None,:] * jnp.arange(NS)[...,None]/NS
    sd_res = sdf_func(ray_sample_pnts)
    rev_sd_res = 1-sd_res
    rev_sd_scan = jax.lax.associative_scan(lambda a,x: a*x, rev_sd_res, axis=-1)
    rev_sd_scan = jnp.concatenate([jnp.ones_like(rev_sd_scan[...,-1:]), rev_sd_scan], axis=-1)
    depth_arr = jnp.linalg.norm(rays_e - rays_s, axis=-1, keepdims=True) * jnp.arange(NS+1)/NS
    depth_dist = rev_sd_scan * jnp.concatenate([sd_res, 5*jnp.ones_like(sd_res[...,-1:])], axis=-1)
    depth_dist = depth_dist / (jnp.sum(depth_dist, axis=-1, keepdims=True)+ 1e-8)
    depth = jnp.sum(depth_dist*depth_arr, axis=-1)
    depth = jnp.reshape(depth, PIXEL_SIZE)

    # depth to pnts
    depth_start_pnts = rays_s
    depth_align = 1
    depth_pnts = depth_start_pnts + ray_dir_normalized * depth[...,None] * depth_align

    # depth_pnts_sd_res = sdf_func(depth_pnts)
    # obj_mask = jnp.array(depth_pnts_sd_res[...,None] > 0.1, dtype=jnp.float32)
    obj_mask = 1-depth_dist[...,-1:]

    light_position = jnp.array([1,1,1])
    vec_to_light = vec_normalize(light_position - depth_pnts)
    grad_func = jax.grad(sdf_func)
    normal_vector = jnp.reshape(jax.vmap(grad_func)(jnp.reshape(depth_pnts, (-1,3))), depth_pnts.shape)
    normal_vector = -vec_normalize(normal_vector)

    to_view_start = vec_normalize(depth_start_pnts - depth_pnts)
    reflection_vec = 2*(jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True))*normal_vector - vec_to_light
    spec_alpha = 10
    img = obj_mask * (0.8*color + 
                    jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True)*color+
                    0.8*(jnp.maximum(jnp.sum(reflection_vec*to_view_start,axis=-1,keepdims=True), 0)**spec_alpha)*jnp.array([1,1,1])
                    )\
            + background_color * (1-obj_mask)
    img = jnp.clip(img, 0, 1)
    if depth_out:
        return img, depth
    else:
        return img


# @partial(jax.jit, static_argnums=(0,1))
def sdf_renderer_exact(sdf_func, cam_SE3=None, near=0.01, far=2.0, light_position=jnp.array([0.2,0.2,1]), color=jnp.array([230/255, 9/255, 101/255]), background_color=jnp.array([1,1,1])):
    # hyper parameters
    NS = 20
    PIXEL_SIZE =  [120, 120]
    cam_zeta = [PIXEL_SIZE[1]*0.8, PIXEL_SIZE[0]*0.8, 0.5*(PIXEL_SIZE[1]-1), 0.5*(PIXEL_SIZE[0]-1)]
    spec_alpha = 5
    LightPower = 5.0

    def vec_normalize(vec):
        return vec/(jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)

    if cam_SE3 is None:
        view_SO3 = jl.SO3.from_x_radians(jnp.pi+jnp.pi/6)@jl.SO3.from_y_radians(jnp.pi/6)
        cam_SE3 = jl.SE3.from_rotation(rotation=view_SO3) @ \
                jl.SE3.from_rotation_and_translation(rotation=jl.SO3(jnp.array([1,0,0,0],dtype=jnp.float32)), translation=jnp.array([0,0,-0.8]))

    K_mat = jnp.array([[cam_zeta[0], 0, cam_zeta[2]],
                    [0, cam_zeta[1], cam_zeta[3]],
                    [0,0,1]])

    # pixel= PVM (colomn-wise)
    # M : points
    # V : inv(cam_SE3)
    # P : Z projection and intrinsic matrix  
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
    ray_dir_normalized = ray_dir/jnp.linalg.norm(ray_dir, axis=-1, keepdims=True)

    # batch ray marching
    ray_sample_pnts = rays_s[...,None,:] + ray_dir[...,None,:] * jnp.arange(NS)[...,None]/NS
    sd_res = sdf_func(ray_sample_pnts)
    for _ in range(20):
        ray_sample_pnts += sd_res[...,None]*ray_dir_normalized[...,None,:]
        sd_res = sdf_func(ray_sample_pnts)
    boundary_mask = jnp.array(jnp.abs(sd_res) < 5e-5, jnp.float32)
    rev_bm = 1-boundary_mask
    rev_bm = jax.lax.associative_scan(lambda a,x: a*x, rev_bm, axis=-1)
    rev_bm = jnp.concatenate([jnp.ones_like(rev_bm[...,-1:]), rev_bm], axis=-1)
    depth_dist = rev_bm * jnp.concatenate([boundary_mask, jnp.ones_like(sd_res[...,-1:])], axis=-1)
    boundary_pnts = jnp.sum(jnp.concatenate([ray_sample_pnts,rays_e[...,None,:]], axis=-2)*depth_dist[...,None], axis=-2)

    # depth to pnts
    obj_mask = 1-depth_dist[...,-1:]

    vec_to_light = light_position - boundary_pnts
    len_to_light = jnp.linalg.norm(vec_to_light, axis=-1, keepdims=True)
    vec_to_light = vec_to_light / len_to_light
    grad_func = jax.grad(sdf_func)
    normal_vector = jnp.reshape(jax.vmap(grad_func)(jnp.reshape(boundary_pnts, (-1,3))), boundary_pnts.shape)
    normal_vector = vec_normalize(normal_vector)

    to_view_start = vec_normalize(rays_s - boundary_pnts)
    reflection_vec = 2*(jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True))*normal_vector - vec_to_light

    cosTheta = jnp.clip( jnp.sum(normal_vector*vec_to_light , axis=-1, keepdims=True), 0,1 )
    cosAlpha = jnp.clip( jnp.sum(reflection_vec*to_view_start,axis=-1,keepdims=True), 0,1 )
    img = obj_mask * (0.2*color + 
                    color * LightPower * cosTheta / (len_to_light*len_to_light) +
                    0.3*jnp.ones((3,)) * LightPower * pow(cosAlpha,spec_alpha) / (len_to_light*len_to_light)
                    )\
            + background_color * (1-obj_mask)
    img = jnp.clip(img, 0, 1)
    return img

# sdf_crop_scale = 0.10
def sphere_sdf(r):
    return lambda x : -r+jnp.linalg.norm(x, axis=-1)
    # return lambda x : jax.nn.tanh(-1/sdf_crop_scale*(r-jnp.linalg.norm(x, axis=-1)))*sdf_crop_scale

def box_sdf(h_ext):
    return lambda x : -jnp.min(h_ext-jnp.abs(x), axis=-1)
    # return lambda x : jax.nn.tanh(-1/sdf_crop_scale*jnp.min(h_ext-jnp.abs(x), axis=-1))*sdf_crop_scale

def transform_sdf(sdf_func, translate, rotate_quat):
    # SE3_inv = SE3.inverse()
    posquat_inv = tutil.pq_inv(translate, rotate_quat)
    return lambda x : sdf_func(tutil.pq_action(*posquat_inv, x))

def union_sdf(sdf_func1, sdf_func2):
    return lambda x : jnp.minimum(sdf_func1(x), sdf_func2(x))

def intersection_sdf(sdf_func1, sdf_func2):
    return lambda x : jnp.maximum(sdf_func1(x), sdf_func2(x))

def batch_operator(operator, batch_data):
    origin_shape = batch_data.shape
    return jnp.reshape(jax.vmap(operator)(jnp.reshape(batch_data, (-1, origin_shape[-1]))), origin_shape[:-1]+(-1,))

def normalize(vec):
    return vec/(jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # sdf_func = sphere_sdf(0.8)
    sdf_func = union_sdf(sphere_sdf(0.6), box_sdf(np.array([0.8,0.1,0.1])))

    view_SO3 = jl.SO3.from_x_radians(jnp.pi+jnp.pi/8)@jl.SO3.from_y_radians(jnp.pi/8)
    cam_SE3 = jl.SE3.from_rotation(rotation=view_SO3) @ \
        jl.SE3.from_rotation_and_translation(rotation=jl.SO3(jnp.array([1,0,0,0],dtype=jnp.float32)), translation=jnp.array([0,0,-2.0]))
    light_position = jnp.array([-1.5,1.0,2.0])

    img = sdf_renderer_exact(sdf_func, cam_SE3=cam_SE3, far=3, light_position=light_position)

    # time_s = time.time()
    # for _ in range(100):
    #     sdf_renderer_exact(sdf_func)
    # print(time.time() - time_s)

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
# %%
