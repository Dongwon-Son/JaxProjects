# %%
import jax.numpy as jnp
import jax
import numpy as np
import einops
import pybullet as p
from moviepy.editor import ImageSequenceClip
from IPython.display import Image

import util.transform_util as tutil
import util.pybullet_render_util as prutil


# %%
# create box
box_pnts = jnp.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[1,1,-1],[1,-1,1],[-1,1,1],[1,1,1]])

# %%
# dynamics step
def dynamics_step(pos, quat, twist, gain_a, gain_b, ext_wrench):
    dt = 0.005
    box_pnts_cur = tutil.pq_action(pos, quat, box_pnts)
    pen_depth = box_pnts_cur[...,2]
    con_pnt_vel = twist[...,:3] + jnp.cross(twist[...,3:], box_pnts_cur - pos)
    normal_rel_vel = con_pnt_vel[...,2]
    con_normals = jnp.array([0,0,1], dtype=jnp.float32)

    con_impulse_value = -pen_depth * gain_a - normal_rel_vel * gain_b

    n_mask = jnp.where((con_impulse_value > 0) & (pen_depth < 0) & (normal_rel_vel < 0), 1, 0)
    con_impulse_n = n_mask[...,None] * con_normals * con_impulse_value[...,None]

    con_impulse_cm = jnp.concatenate([con_impulse_n, jnp.cross(box_pnts_cur - pos, con_impulse_n)], axis=-1)
    con_impulse_cm = jnp.sum(con_impulse_cm, axis=-2)

    # integration
    mass = 1
    inertia = 0.1
    Imat_b = jnp.diag(jnp.array([inertia,inertia,inertia]))
    
    Imat = tutil.qaction(quat, Imat_b)
    Imat = tutil.qaction(quat, einops.rearrange(Imat, '... i j -> ... j i'))
    Imat = einops.rearrange(Imat, '... i j -> ... j i')
    Imat_inv = jnp.linalg.inv(Imat)

    Iw = jnp.einsum('...ij,...j->...i',Imat, twist[...,3:])
    twist = twist.at[...,:3].set(twist[...,:3] + con_impulse_cm[...,:3]/mass + ext_wrench[...,:3]/mass*dt)
    twist = twist.at[...,3:].set(twist[...,3:] + jnp.einsum('...ij,...j->...i',Imat_inv, con_impulse_cm[...,3:]) + jnp.einsum('...ij,...j->...i',Imat_inv, jnp.cross(twist[...,3:], Iw)*dt))

    # if jnp.sum(jnp.abs(twist[...,3:])) > 0.001:
    #     print(1)

    pos += twist[...,:3] * dt
    quat = tutil.qmulti(tutil.qexp(twist[...,3:]*dt/2), quat)
    # quat = quat + dt * tutil.qmulti(jnp.concatenate([0.5*twist[...,3:], jnp.zeros_like(twist[...,:1])], axis=-1), quat)
    # quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)

    return pos, quat, twist

dynamics_step_jit = jax.jit(dynamics_step)

# %%
prutil.init()
prutil.make_objs(np.array([[0,1,1,1], [0,5,5,0.1]]), [[0,0,0], [0,0,-0.1]], [[0,0,0,1],[0,0,0,1]], [[1],[1]])

#%%
# step
pos, quat, twist = jnp.array([0,0,3], dtype=jnp.float32), jnp.array([np.sin(np.pi/12),0,0,np.cos(np.pi/12)] ,dtype=jnp.float32), jnp.array([0,0,0,0,0,0], dtype=jnp.float32)
quat = tutil.qrand(outer_shape=())

frames = []
fps = 20
pp = 1/0.005/fps
for i in range(2000):
    pos, quat, twist = dynamics_step_jit(pos, quat, twist, gain_a=1, gain_b=0.01, ext_wrench=jnp.array([0,0,-9.81,0,0,0]))
    # pos, quat, twist = dynamics_step(pos, quat, twist, gain_a=100, gain_b=0, ext_wrench=jnp.array([0,0,-9.81,0,0,0]))

    if i%pp == 0:
        p.resetBasePositionAndOrientation(0, pos, quat)
        frames.append(prutil.get_rgb([0,6.0,6.0]))

clip = ImageSequenceClip(list(frames), fps=fps)
clip.write_gif('test.gif', fps=fps)

Image('test.gif')

# %%
