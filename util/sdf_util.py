# %%
import jax.numpy as jnp
import numpy as np
import jax

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

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def sphere_sdf(center, r, x):
        return jnp.array((jnp.sum((x - center)**2, axis=-1) < r**2), dtype=jnp.float32)
    
    depth = visualize_sdf(lambda x : sphere_sdf(np.array([0,0,0]), 0.1, x))
    plt.figure()
    plt.imshow(depth)
    plt.axis('off')
    plt.show()