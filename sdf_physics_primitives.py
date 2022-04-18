# %%
# import libraries
import jax.numpy as jnp
import jax
import pybullet as p
import numpy as np
import einops
from moviepy.editor import ImageSequenceClip
from IPython.display import Image

import util.sdf_util as sdfutil
import util.transform_util as tutil
import util.pybullet_render_util as prutil

jkey = jax.random.PRNGKey(1)
# %%
# build params
def random_param(jkey, outer_shape=(1,)):
    _, jkey = jax.random.split(jkey)
    pos = jax.random.uniform(jkey, shape=outer_shape+(3,), minval=jnp.array([-0.3,-0.3,0]), maxval=jnp.array([0.3,0.3,0.6]), dtype=jnp.float32)

    _, jkey = jax.random.split(jkey)
    quat = tutil.qrand(outer_shape, jkey)

    _, jkey = jax.random.split(jkey)
    scale = jax.random.uniform(jkey, outer_shape+(1,), jnp.float32, 0.03, 0.10) 

    _, jkey = jax.random.split(jkey)
    ptype = jax.random.randint(jkey, outer_shape+(1,), 0, 4, jnp.int32).astype(jnp.float32)

    _, jkey = jax.random.split(jkey)
    geo_param = jax.random.uniform(jkey, outer_shape+(3,), jnp.float32, 0.2, 1.0)
    geo_param = geo_param.at[jnp.where(ptype[...,0]==1)].set(jnp.concatenate([geo_param[...,:1], geo_param[...,:1], geo_param[...,2:3]], axis=-1)[jnp.where(ptype[...,0]==1)])
    geo_param = geo_param.at[jnp.where(ptype[...,0]==2)].set(jnp.concatenate([geo_param[...,:1], geo_param[...,:1], geo_param[...,2:3]], axis=-1)[jnp.where(ptype[...,0]==2)])
    geo_param = geo_param.at[jnp.where(ptype[...,0]==3)].set(jnp.concatenate([geo_param[...,:1], geo_param[...,:1], geo_param[...,:1]], axis=-1)[jnp.where(ptype[...,0]==3)])
    geo_param = geo_param/jnp.max(geo_param, axis=-1, keepdims=True)

    return jnp.concatenate([ptype, geo_param], axis=-1), pos, quat, scale

# %%
# intersect sdf value and grad
def intsc_sdf(x, geo_paramij, posij, quatij, scaleij, min_value=0):
    sdfij = sdfutil.transform_sdf(sdfutil.primitive_sdf(geo_paramij), posij, quatij, scaleij)(einops.repeat(x, '... k -> ... 2 k'))
    sdf = jnp.max(sdfij, axis=-2)
    # return jnp.maximum(sdf[...,0], 0)
    return sdf[...,0]

def sdf_func(x, geo_param, pos, quat, scale):
    sdf = sdfutil.transform_sdf(sdfutil.primitive_sdf(geo_param), pos, quat, scale)(x)
    return sdf[...,0]

flat_intsc_sdf_grad = jax.vmap(jax.value_and_grad(intsc_sdf))
flat_sdf_grad = jax.vmap(jax.value_and_grad(sdf_func))

def intsc_sdf_grad(*inputs, min_value=0):
    origin_outer_shape = inputs[0].shape[:-1]
    inputs_flat = jax.tree_map(lambda x : x.reshape((-1,)+ x.shape[len(origin_outer_shape):]), inputs)
    sdf, grad = flat_intsc_sdf_grad(*inputs_flat, min_value=einops.repeat(jnp.array(min_value), ' -> nb', nb=inputs_flat[0].shape[0]))
    return sdf.reshape(origin_outer_shape), grad.reshape(origin_outer_shape + (grad.shape[-1],))

def sdf_grad(*inputs):
    origin_outer_shape = inputs[0].shape[:-1]
    inputs_flat = jax.tree_map(lambda x : x.reshape((-1,)+ x.shape[len(origin_outer_shape):]), inputs)
    sdf, grad = flat_sdf_grad(*inputs_flat)
    return sdf.reshape(origin_outer_shape), grad.reshape(origin_outer_shape + (grad.shape[-1],))

# %timeit sdf_func_value_and_grad((x_sample, geo_param, pos, quat, scale))
# %timeit sdf_func_value_and_grad_jit((x_sample, geo_param, pos, quat, scale))

# %%
# dynamics step

def cull_idx(pos, k, fix_idx=None):
    no = pos.shape[-2]
    oi_idx, oj_idx = jnp.triu_indices(no, k=1)
    posi = pos[...,oi_idx,:]
    posj = pos[...,oj_idx,:]
    distanceij = jnp.linalg.norm(posj - posi, axis=-1)
    distanceij = distanceij.at[...,:no-1].set(posj[...,:no-1,2])
    if fix_idx is not None:
        fix_utidx = jnp.triu_indices(fix_idx.shape[-1], k=1)
        fix_idx_i = fix_idx[..., fix_utidx[0]]
        fix_idx_j = fix_idx[..., fix_utidx[1]]
        fix_idx_fut = ((no-1 + no-(fix_idx_i-1)-1) * (fix_idx_i) * 0.5 + fix_idx_j - fix_idx_i - 1).astype(jnp.int32)
        if len(fix_idx.shape) == 1:
            distanceij = distanceij.at[:, fix_idx_fut].add(1000)
        else:
            distanceij = distanceij.at[jnp.tile(jnp.arange(distanceij.shape[0])[...,None], [1, fix_idx_fut.shape[-1]]), fix_idx_fut].add(100)
    sort_idx = jnp.argsort(distanceij)[...,:k]
    return oi_idx[sort_idx], oj_idx[sort_idx]


def dynamics_step(jkey, geo_param, pos, quat, scale, twist, ext_wrench, 
                    cull_k, fix_idx, dt, mass, inertia, gain, cp_no, gain_b):
    no = geo_param.shape[-2]

    # get contact points
    culli, cullj = cull_idx(pos, cull_k, fix_idx)
    geo_paramij, posij, quatij, scaleij, twistij = jax.tree_map(lambda x : jnp.stack([jnp.take_along_axis(x, culli[...,None], axis=-2), jnp.take_along_axis(x, cullj[...,None], axis=-2)], axis=-2), 
                        (geo_param, pos, quat, scale, twist))
    ns = cp_no
    x_samples_ij = jax.random.uniform(jkey, posij.shape[:-2] + (ns,3), posij.dtype, -1, 1)
    x_samples_i = tutil.pq_action(posij[...,0:1,:], quatij[...,0:1,:], x_samples_ij[...,:int(ns/2),:]/scaleij[...,0:1,:]/geo_paramij[...,0:1,1:])
    x_samples_j = tutil.pq_action(posij[...,1:2,:], quatij[...,1:2,:], x_samples_ij[...,int(ns/2):,:]/scaleij[...,1:2,:]/geo_paramij[...,1:2,1:])
    x_samples_i = jnp.where(einops.repeat(culli, '... i -> ... i ns 1', ns=int(ns/2))==0, x_samples_j.at[...,2].set(-x_samples_j[...,2]), x_samples_i) ## if plane.. opposite samples
    x_samples = jnp.concatenate([x_samples_i, x_samples_j], axis=-2)
    addparams = jax.tree_map(lambda x : einops.repeat(x, '... i j -> ... ns i j', ns=ns) , (geo_paramij, posij, quatij, scaleij))
    for _ in range(3):
        _, jkey = jax.random.split(jkey)
        sdf, grad = intsc_sdf_grad(x_samples, *addparams, min_value=0.00) # (NB, NR, NS)
        x_samples -= grad*jnp.maximum(sdf[...,None], 0.001)
    con_pnt_ijs = einops.repeat(x_samples, '... i -> ... 2 i')
    sdfijs, gradijs = sdf_grad(con_pnt_ijs, *addparams)
    normalijs = -sdfutil.normalize(gradijs)
    normalijs = normalijs[...,0,:] - normalijs[...,1,:]
    normalijs = jnp.stack([normalijs, -normalijs], axis=-2)
    normalijs = sdfutil.normalize(normalijs)

    # plane_normal normal
    normalijs_plane = jnp.zeros_like(normalijs).at[...,0,2].set(-1)
    normalijs_plane = normalijs_plane.at[...,1,2].set(1)
    normalijs = jnp.where(einops.repeat(culli, '... i -> ... i ns nij 1', ns=ns, nij=2)==0, normalijs_plane, normalijs)

    pdijs = jnp.maximum(-sdfijs, 0)
    con_vel_ijs = twistij[...,None,:,:3] + jnp.cross(twistij[...,None,:,3:], con_pnt_ijs - posij[...,None,:,:])
    rel_con_vel_ijs = con_vel_ijs[...,0:1,:] - con_vel_ijs[...,1:2,:]
    rel_con_vel_ijs = jnp.concatenate([rel_con_vel_ijs, -rel_con_vel_ijs], axis=-2)
    rel_con_normal_vel_ijs = jnp.sum(rel_con_vel_ijs*normalijs, axis=-1, keepdims=True)
    # con_impulse_value = (pdijs[...,None] * gain + rel_con_normal_vel_ijs * jnp.sqrt(gain))
    con_impulse_value = (pdijs[...,None] * gain - rel_con_normal_vel_ijs * gain_b)
    con_impulse_ijs = normalijs * con_impulse_value

    n_mask = jnp.where((con_impulse_value>0) & (rel_con_normal_vel_ijs<=0.0001) & (jnp.max(sdfijs, axis=-1, keepdims=True)[...,None] < 0.0001), 1, 0)
    con_impulse_ijs = con_impulse_ijs * n_mask
    con_impulse_is = con_impulse_ijs[...,0,:] - con_impulse_ijs[...,1,:]
    con_impulse_ijs = jnp.stack([con_impulse_is, -con_impulse_is], axis=-2)

    impulse_ijs = jnp.concatenate([con_impulse_ijs, jnp.cross((con_pnt_ijs - addparams[1]), con_impulse_ijs)], axis=-1)
    impulse_ij = jnp.sum(impulse_ijs, axis=-3)

    impulse_ijmat = jnp.zeros(impulse_ij.shape[:-3]+(no,no,6))
    nr = culli.shape[-1]
    batch_idx = einops.repeat(jnp.arange(impulse_ij.shape[0]), 'i -> i nr', nr=nr)
    impulse_ijmat = impulse_ijmat.at[batch_idx, culli, cullj].set(impulse_ij[...,0,:])
    impulse_ijmat = impulse_ijmat.at[batch_idx, cullj, culli].set(impulse_ij[...,1,:])

    con_impulse = jnp.sum(impulse_ijmat, axis=-2)

    Imat_b = jnp.diag(jnp.array([inertia,inertia,inertia]))
    Imat = tutil.qaction(quat[...,None,:], Imat_b[None,None])
    Imat = tutil.qaction(quat[...,None,:], einops.rearrange(Imat, '... i j -> ... j i'))
    Imat = einops.rearrange(Imat, '... i j -> ... j i')
    Imat_inv = jnp.linalg.inv(Imat)

    Iw = jnp.einsum('...ij,...j->...i',Imat, twist[...,3:])
    twist = twist.at[...,:3].set(twist[...,:3] + con_impulse[...,:3]/mass + ext_wrench[...,:3]/mass*dt)
    twist = twist.at[...,3:].set(twist[...,3:] + jnp.einsum('...ij,...j->...i',Imat_inv, con_impulse[...,3:]) + jnp.einsum('...ij,...j->...i',Imat_inv, jnp.cross(twist[...,3:], Iw)*dt))
    twist = twist.at[...,fix_idx,:].set(0)

    pos += twist[...,:3] * dt
    quat = tutil.qmulti(tutil.qexp(twist[...,3:]*dt/2), quat)


    return pos, quat, twist

dynamics_step_jit = jax.jit(dynamics_step, static_argnames=['cp_no', 'cull_k', 'mass', 'dt', 'inertia'])

# %%
# physical test init
prutil.init()

# %%
# param random
NB = 2
NO = 30
inputs = random_param(jkey, (NB,NO))

# test primitives
# inputs = (jnp.array([[2,1,1,1]], dtype=jnp.float32), jnp.array([[0.1,0.1,0.50]], dtype=jnp.float32), jnp.array([[np.sin(np.pi/12),0,0,np.cos(np.pi/12)]], dtype=jnp.float32), jnp.array([[0.080]], dtype=jnp.float32))
# inputs = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=NB), inputs)

# add table - should be index 0!!!
plane_param = (jnp.array([[0,1,1,0.05]]), jnp.array([[0,0,-0.05]]), jnp.array([[0,0,0,1]]), jnp.array([[1]]))
plane_param = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=NB), plane_param)
inputs = jax.tree_map(lambda *x : jnp.concatenate(x, axis=1), plane_param, inputs)
geo_param, pos, quat, scale = inputs
prutil.make_objs(geo_param[-1], pos[-1], quat[-1], scale[-1])

# %%
# dynamics step
geo_param, pos, quat, scale = inputs
physics_param = {}
physics_param['mass'] = 1.0
physics_param['inertia'] = 0.01
physics_param['dt'] = 0.003
physics_param['cull_k'] = 80
physics_param['gain'] = 5
physics_param['gain_b'] = 0.05
physics_param['cp_no'] = 6
# physics_param['fix_idx'] = jnp.arange(1).astype(jnp.int32)
physics_param['fix_idx'] = jnp.array([0]).astype(jnp.int32)
twist = jnp.zeros(pos.shape[:-1] + (6,), dtype=jnp.float32)
frames = []
fps = 21
pp = int(1/physics_param['dt']/fps)
for i in range(1000):
    _, jkey = jax.random.split(jkey)
    gv_force = -9.81 * jnp.ones_like(pos) * physics_param['mass']
    gv_force = gv_force.at[:,:,:2].set(0)
    ext_wrench = jnp.concatenate([gv_force, jnp.zeros_like(gv_force)], axis=-1)
    pos, quat, twist = dynamics_step_jit(jkey, geo_param, pos, quat, scale, twist, ext_wrench, **physics_param)
    # pos, quat, twist = dynamics_step(jkey, geo_param, pos, quat, scale, twist, ext_wrench, **physics_param)

    # pybullet test
    if i%pp == 0:
        rpos, rquat = pos[-1], quat[-1]
        for j in range(rpos.shape[0]):
            p.resetBasePositionAndOrientation(j, rpos[j], rquat[j])
        # frames.append(prutil.get_rgb([0,0.8,0.0]))
        frames.append(prutil.get_rgb([0,0.6,0.6]))

clip = ImageSequenceClip(list(frames), fps=fps)
clip.write_gif('test.gif', fps=fps)

Image('test.gif')

# %%
