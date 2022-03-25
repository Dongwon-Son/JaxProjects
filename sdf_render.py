# %%
import jax.numpy as jnp
import jax
import numpy as np
import jaxlie as jl
import matplotlib.pyplot as plt

# sphere sdf for test
def sphere_sdf(center, r, x):
    return jax.nn.sigmoid(100*(r-jnp.linalg.norm(x - center, axis=-1)))
    # return jnp.array((jnp.sum((x - center)**2, axis=-1) < r**2), dtype=jnp.float32)

# %%
# pixel sampling
PIXEL_SIZE =  [200, 200]
x_grid, y_grid = jnp.meshgrid(jnp.arange(PIXEL_SIZE[1]), jnp.arange(PIXEL_SIZE[0]))
x_grid_centered = (x_grid - 0.5*(PIXEL_SIZE[1]-1))/(PIXEL_SIZE[1]-1)
y_mesh_centered = -(y_grid - 0.5*(PIXEL_SIZE[0]-1))/(PIXEL_SIZE[0]-1)
rays_s_canonical = jnp.stack([x_grid_centered, y_mesh_centered, jnp.zeros_like(y_mesh_centered)], axis=-1)
rays_e_canonical = jnp.stack([x_grid_centered, y_mesh_centered, -jnp.ones_like(y_mesh_centered)], axis=-1)
origin_shape = rays_s_canonical.shape

# %%
# cam SE3 transformation
view_SO3 = jl.SO3.from_x_radians(0)
cam_SE3 = jl.SE3.from_rotation_and_translation(rotation=view_SO3, translation=jnp.array([0,0,1]))
rays_s = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_s_canonical, (-1,3))), origin_shape)
rays_e = jnp.reshape(jax.vmap(cam_SE3.apply)(jnp.reshape(rays_e_canonical, (-1,3))), origin_shape)

    

# %%
# ray to depth
NS = 200
ray_dir = rays_e - rays_s
ray_dir_normalized = ray_dir/jnp.linalg.norm(ray_dir, axis=-1, keepdims=True)
ray_sample_pnts = rays_s[...,None,:] + ray_dir[...,None,:] * jnp.arange(NS)[...,None]/NS
sd_res = (lambda x : sphere_sdf(np.array([0,0,0]), 0.4, x))(ray_sample_pnts)
rev_sd_res = 1-sd_res
rev_sd_scan = jax.lax.associative_scan(lambda a,x: a*x, rev_sd_res, axis=-1)
rev_sd_scan = jnp.concatenate([jnp.ones_like(rev_sd_scan[...,-1:]), rev_sd_scan], axis=-1)
depth_arr = jnp.linalg.norm(rays_e - rays_s, axis=-1, keepdims=True) * jnp.arange(NS+1)/NS
depth_dist = rev_sd_scan * jnp.concatenate([sd_res, jnp.ones_like(sd_res[...,-1:])], axis=-1)
depth_dist = depth_dist / (jnp.sum(depth_dist, axis=-1, keepdims=True)+ 1e-8)
depth = jnp.sum(depth_dist*depth_arr, axis=-1)
depth = jnp.reshape(depth, PIXEL_SIZE)

# %%
# plot depth
plt.figure()
plt.imshow(depth)
plt.axis('off')
plt.show()
# %%
# depth to pnts
depth_start_pnts = rays_s
depth_align = 1
depth_pnts = depth_start_pnts + ray_dir_normalized * depth[...,None] * depth_align

# %%
# ambient render
depth_pnts_sd_res = (lambda x : sphere_sdf(np.array([0,0,0]), 0.4, x))(depth_pnts)
ambient_color = jnp.array([0.3,0.1,0.1])
background_color = jnp.array([1,1,1])
obj_mask = jnp.array(depth_pnts_sd_res[...,None] > 0.1, dtype=jnp.float32)
img = obj_mask * ambient_color + background_color * (1-obj_mask)

plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

# %%
# diffusion color
def vec_normalize(vec):
    return vec/(jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)
light_position = jnp.array([1,1,1])
vec_to_light = vec_normalize(light_position - depth_pnts)
grad_func = jax.grad((lambda x : sphere_sdf(np.array([0,0,0]), 0.4, x)))
normal_vector = jnp.reshape(jax.vmap(grad_func)(jnp.reshape(depth_pnts, (-1,3))), depth_pnts.shape)
normal_vector = -vec_normalize(normal_vector)

img = obj_mask * (ambient_color + jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True)*ambient_color)\
        + background_color * (1-obj_mask)

normal_img = obj_mask * normal_vector\
        + background_color * (1-obj_mask)


plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

# %%
# speculation color
to_view_start = vec_normalize(depth_start_pnts - depth_pnts)
reflection_vec = 2*(jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True))*normal_vector - vec_to_light
spec_alpha = 10
img = obj_mask * (0.8*ambient_color + 
                jnp.sum(normal_vector*vec_to_light, axis=-1, keepdims=True)*ambient_color+
                0.8*(jnp.maximum(jnp.sum(reflection_vec*to_view_start,axis=-1,keepdims=True), 0)**spec_alpha)*jnp.array([1,1,1])
                )\
        + background_color * (1-obj_mask)
img = jnp.clip(img, 0, 1)

plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()