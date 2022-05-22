# %%
# import libraries
import jax.numpy as jnp
import jax
import pybullet as p
import numpy as np
import einops
from moviepy.editor import ImageSequenceClip
from IPython.display import Image
import time
import matplotlib.pyplot as plt

import util.sdf_util as sdfutil
import util.transform_util as tutil
import util.pybullet_render_util as prutil

jkey = jax.random.PRNGKey(1)
# %%
# build params
def random_param(jkey, outer_shape=(1,)):
    _, jkey = jax.random.split(jkey)
    pos = jax.random.uniform(jkey, shape=outer_shape+(3,), minval=jnp.array([-0.3,-0.3,0.2]), maxval=jnp.array([0.3,0.3,0.6]), dtype=jnp.float32)

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
def box_intsc_sdf(x, posij, quatij, scaleij):
    sdfij = sdfutil.transform_sdf(sdfutil.box_round_sdf(jnp.array([0.98,0.98,0.98]), 0.01), posij, quatij, scaleij)(einops.repeat(x, '... k -> ... 2 k'))
    sdf = jnp.max(sdfij, axis=-2)
    return sdf[...,0]

def intsc_sdf(x, geo_paramij, posij, quatij, scaleij):
    sdfij = sdfutil.transform_sdf(sdfutil.primitive_sdf(geo_paramij), posij, quatij, scaleij)(einops.repeat(x, '... k -> ... 2 k'))
    sdf = jnp.max(sdfij, axis=-2)
    return sdf[...,0]

def sdf_func(x, geo_param, pos, quat, scale):
    sdf = sdfutil.transform_sdf(sdfutil.primitive_sdf(geo_param), pos, quat, scale)(x)
    return sdf[...,0]

flat_box_intsc_sdf_grad = jax.vmap(jax.value_and_grad(box_intsc_sdf))
flat_intsc_sdf_grad = jax.vmap(jax.value_and_grad(intsc_sdf))
flat_sdf_grad = jax.vmap(jax.value_and_grad(sdf_func))

def box_intsc_sdf_grad(*inputs):
    origin_outer_shape = inputs[0].shape[:-1]
    inputs_flat = jax.tree_map(lambda x : x.reshape((-1,)+ x.shape[len(origin_outer_shape):]), inputs)
    sdf, grad = flat_box_intsc_sdf_grad(*inputs_flat)
    return sdf.reshape(origin_outer_shape), grad.reshape(origin_outer_shape + (grad.shape[-1],))

def intsc_sdf_grad(*inputs):
    origin_outer_shape = inputs[0].shape[:-1]
    inputs_flat = jax.tree_map(lambda x : x.reshape((-1,)+ x.shape[len(origin_outer_shape):]), inputs)
    sdf, grad = flat_intsc_sdf_grad(*inputs_flat)
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

def calculate_contact_points(jkey, cp_no, geo_paramij, posij, quatij, scaleij, culli, visualize=False):
    # get contact points
    ns = cp_no
    assert ns >= 18
    # x_samples_ij = jax.random.uniform(jkey, posij.shape[:-2] + (ns,3), posij.dtype, -1, 1)
    # x_samples_ij = jax.random.normal(jkey, posij.shape[:-2] + (int(ns/2),3,2), posij.dtype)*jnp.array([0.1,1])
    # x_samples_ij = jnp.stack([jax.random.uniform(jkey, posij.shape[:-2] + (int(ns/2),3), posij.dtype, -2, 2), jax.random.normal(jkey, posij.shape[:-2] + (int(ns/2),3), posij.dtype)*0.1], axis=-1)
    # x_samples_ij = einops.rearrange(x_samples_ij, '... i j k -> ... (i k) j')
    # x_samples_i = tutil.pq_action(posij[...,0:1,:], quatij[...,0:1,:], x_samples_ij[...,:int(ns/2),:]*scaleij[...,0:1,:]*geo_paramij[...,0:1,1:])
    # x_samples_j = tutil.pq_action(posij[...,1:2,:], quatij[...,1:2,:], x_samples_ij[...,int(ns/2):,:]*scaleij[...,1:2,:]*geo_paramij[...,1:2,1:])

    corner = jnp.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[1,1,-1],[1,-1,1],[-1,1,1],[-1,-1,-1]]).astype(jnp.float32)
    corner = einops.repeat(corner, 'i j -> nb nr i j', nb=posij.shape[0], nr=posij.shape[1])
    corner = jnp.concatenate([corner, jax.random.normal(jkey, posij.shape[:-2] + (int((ns-16)*0.5),3), posij.dtype)*0.1], axis=-2)
    x_samples_i = tutil.pq_action(posij[...,0:1,:], quatij[...,0:1,:], corner*scaleij[...,0:1,:]*geo_paramij[...,0:1,1:])
    x_samples_j = tutil.pq_action(posij[...,1:2,:], quatij[...,1:2,:], corner*scaleij[...,1:2,:]*geo_paramij[...,1:2,1:])
    # x_samples_i = jnp.where(einops.repeat(culli, '... i -> ... i ns 1', ns=int(ns/2))==0, x_samples_j.at[...,2].set(-x_samples_j[...,2])+jax.random.normal(jkey,x_samples_j.shape)*0.010, x_samples_i) ## if plane.. opposite samples
    x_samples_i = jnp.where(einops.repeat(culli, '... i -> ... i ns 1', ns=int(ns/2))==0, (posij[...,1:2,:]+jax.random.normal(jkey,x_samples_j.shape)*0.20*scaleij[...,1:2,:]) * jnp.array([1,1,-1]).astype(jnp.float32), x_samples_i) ## if plane.. opposite samples
    x_samples = jnp.concatenate([x_samples_i, x_samples_j], axis=-2)
    addparams = jax.tree_map(lambda x : einops.repeat(x, '... i j -> ... ns i j', ns=ns) , (geo_paramij, posij, quatij, scaleij))

    # broad phase
    # for _ in range(2):
    #     _, jkey = jax.random.split(jkey)
    #     sdf, grad = box_intsc_sdf_grad(x_samples, *addparams[1:])
    #     x_samples -= grad*jnp.maximum(sdf[...,None], 0.001)

    if visualize:
        ns_vis = 10000
        pnts = jax.random.uniform(jkey, [1,1,ns_vis,3], jnp.float32, -0.10, 0.10)
        addparams_vis = jax.tree_map(lambda x : einops.repeat(x, '... i j -> ... ns i j', ns=ns_vis) , (geo_paramij, posij, quatij, scaleij))
        pnts = pnts.at[...,1].set(0)
        test_sdfs = intsc_sdf(pnts, *addparams_vis)
        pnts_plot = pnts.reshape(-1,3)
        test_sdfs_plot = test_sdfs.reshape(-1)
        plt.figure()
        plt.scatter(pnts_plot[...,0], pnts_plot[...,2], s=10.0, c=test_sdfs_plot)
        plt.scatter(pnts_plot[np.where(test_sdfs_plot<=0),0], pnts_plot[np.where(test_sdfs_plot<=0),2], s=10.0, c='red')
        plt.show()

        pnts_xy = jax.random.uniform(jkey, [1,1,ns_vis,3], jnp.float32, -0.010, 0.010)
        pnts_xy = pnts_xy.at[...,2].set(-0.001)
        test_sdfs_xy = intsc_sdf(pnts_xy, *addparams_vis)
        pnts_xy_plot = pnts_xy.reshape(-1,3)
        test_sdfs_xy_plot = test_sdfs_xy.reshape(-1)
        plt.figure()
        plt.scatter(pnts_xy_plot[...,0], pnts_xy_plot[...,1], s=10.0, c=test_sdfs_xy_plot)
        plt.scatter(pnts_xy_plot[np.where(test_sdfs_xy_plot<=0),0], pnts_xy_plot[np.where(test_sdfs_xy_plot<=0),1], s=10.0, c='red')
        plt.show()

    # narrow phase
    for _ in range(1):
        if visualize:
            sample_plot = x_samples.reshape(-1,3)
            # sdf_plot = sdf.reshape(-1)
            plt.figure()
            plt.subplot(1,2,1)
            plt.scatter(pnts_plot[np.where(test_sdfs_plot>0),0], pnts_plot[np.where(test_sdfs_plot>0),2], s=10.0, c=test_sdfs_plot[np.where(test_sdfs_plot>0)])
            plt.scatter(sample_plot[...,0], sample_plot[...,2], s=10.0, c='red')
            plt.subplot(1,2,2)
            plt.scatter(pnts_xy_plot[np.where(test_sdfs_xy_plot>0),0], pnts_xy_plot[np.where(test_sdfs_xy_plot>0),1], s=10.0, c=test_sdfs_xy_plot[np.where(test_sdfs_xy_plot>0)])
            plt.scatter(sample_plot[...,0], sample_plot[...,1], s=10.0, c='red')
            plt.show()
        _, jkey = jax.random.split(jkey)
        sdf, grad = intsc_sdf_grad(x_samples, *addparams) # (NB, NR, NS)
        x_samples -= sdfutil.normalize(grad)*jnp.maximum(sdf[...,None], 0.001)

    if visualize:
        sample_plot = x_samples.reshape(-1,3)
        # sdf_plot = sdf.reshape(-1)
        plt.figure()
        plt.subplot(1,2,1)
        plt.scatter(pnts_plot[np.where(test_sdfs_plot>0),0], pnts_plot[np.where(test_sdfs_plot>0),2], s=10.0, c=test_sdfs_plot[np.where(test_sdfs_plot>0)])
        plt.scatter(sample_plot[...,0], sample_plot[...,2], s=10.0, c='red')
        plt.subplot(1,2,2)
        plt.scatter(pnts_xy_plot[np.where(test_sdfs_xy_plot>0),0], pnts_xy_plot[np.where(test_sdfs_xy_plot>0),1], s=10.0, c=test_sdfs_xy_plot[np.where(test_sdfs_xy_plot>0)])
        plt.scatter(sample_plot[...,0], sample_plot[...,1], s=10.0, c='red')
        plt.show()

    csdf_sij, grad_sij = sdf_grad(einops.repeat(x_samples, '... i -> ... 2 i'), *addparams)
    cps_si = x_samples
    normal_sij_tmp = sdfutil.normalize(grad_sij)
    normal_si = normal_sij_tmp[...,0,:] - normal_sij_tmp[...,1,:]
    cns_si = sdfutil.normalize(normal_si)

    # plane_normal normal
    # norma_si_plane = jnp.zeros_like(normal_si).at[...,2].set(1)
    # cns_si = jnp.where(einops.repeat(culli, '... i -> ... i ns 1', ns=ns)==0, norma_si_plane, normal_si)
    
    return jnp.stack([cps_si, cps_si], axis=-2), jnp.stack([cns_si, -cns_si], axis=-2), csdf_sij

def dynamics_step(jkey, geo_param, pos, quat, scale, twist, ext_wrench, 
                    cull_k, fix_idx, dt, mass, inertia, cp_no, baumgarte_erp, elasticity, mu, substep):
    no = geo_param.shape[-2]
    
    for _ in range(substep):
        _, jkey = jax.random.split(jkey)
        # cull and pick i j objects
        culli, cullj = cull_idx(pos, cull_k, fix_idx)
        cull_ij = jnp.stack([culli, cullj], axis=-1)
        geo_paramij, posij, quatij, scaleij, twistij = jax.tree_map(lambda x : jnp.stack([jnp.take_along_axis(x, culli[...,None], axis=-2), jnp.take_along_axis(x, cullj[...,None], axis=-2)], axis=-2), 
                            (geo_param, pos, quat, scale, twist))
        
        # get contact points
        cps_sij, cns_sij, csdf_sij = calculate_contact_points(jkey, cp_no, geo_paramij, posij, quatij, scaleij, culli)
        cpd_sij = jnp.maximum(-csdf_sij, 0)

        # get contact velocities
        relpos_sij = cps_sij - posij[...,None,:,:]
        cvel_sij = twistij[...,None,:,:3] + jnp.cross(twistij[...,None,:,3:], cps_sij - posij[...,None,:,:])
        cvel_si_vec = cvel_sij[...,0,:] - cvel_sij[...,1,:]
        cvel_n_value = jnp.sum(cvel_si_vec*cns_sij[...,0,:], axis=-1)
        cvel_n_si_vec = cvel_n_value[...,None] * cns_sij[...,0,:]
        cvel_d_si_vec = cvel_si_vec - cvel_n_si_vec
        cvel_d_value = jnp.linalg.norm(cvel_d_si_vec, axis=-1)
        cvel_d_si_dir = cvel_d_si_vec / (1e-6 + cvel_d_value[...,None])

        # contact resolution
        # baumgarte_vel_value_sij = baumgarte_erp * cpd_sij
        baumgarte_vel_value_s = jnp.sum(baumgarte_erp * cpd_sij, axis=-1)
        # baumgarte_vel_value_s = jnp.max(baumgarte_erp * cpd_sij, axis=-1)
        tmp_sij = inertia * jnp.cross(relpos_sij, cns_sij[...,0:1,:])
        tmp_sij = jnp.where(einops.repeat(cull_ij, '... i j -> ... i ns j 1', ns=tmp_sij.shape[-3])==0, 0, tmp_sij)
        ang_s = jnp.sum(cns_sij[...,0,:] * jnp.sum(jnp.cross(tmp_sij, relpos_sij), axis=-2), axis=-1)

        imp_n_s_value = ((1. + elasticity) * cvel_n_value + baumgarte_vel_value_s) / (
                        1. / mass + 1. / mass + ang_s)

        # friction contact
        imp_d_s_value = cvel_d_value / (1./mass + 1./mass + ang_s)
        imp_d_s_value = jnp.minimum(imp_d_s_value, mu * imp_n_s_value)

        # calculate dp
        apply_n_si = jnp.where((jnp.max(csdf_sij, axis=-1) <= 0.0001) & (cvel_n_value >= -0.0001) & (imp_n_s_value > 0.), 1., 0.)
        apply_d_si = apply_n_si * jnp.where(cvel_d_value > 0.01, 1., 0.)
        imp_nd_si_vec = -apply_n_si[..., None] * imp_n_s_value[...,None] * cns_sij[...,0,:] - \
                        apply_d_si[..., None] * imp_d_s_value[...,None] * cvel_d_si_dir
        imp_nd_sij_vec = jnp.stack([imp_nd_si_vec, -imp_nd_si_vec], axis=-2)
        dp_sij = jnp.concatenate([imp_nd_sij_vec, jnp.cross(relpos_sij, imp_nd_sij_vec)], axis=-1)
        dp_ij = jnp.sum(dp_sij, axis=-3) / (1e-6+jnp.sum(apply_n_si, axis=-1, keepdims=True))[..., None]

        # recover dp_mat
        dp_mat = jnp.zeros((dp_ij.shape[0], no, no, 6))
        nr = culli.shape[-1]
        batch_idx = einops.repeat(jnp.arange(dp_ij.shape[0]), 'i -> i nr', nr=nr)
        dp_mat = dp_mat.at[batch_idx, culli, cullj].set(dp_ij[...,0,:])
        dp_mat = dp_mat.at[batch_idx, cullj, culli].set(dp_ij[...,1,:])
        dp_o = jnp.sum(dp_mat, axis=-2)

        # integration
        Imat_b = jnp.diag(jnp.array([inertia,inertia,inertia]))
        Imat = tutil.qaction(quat[...,None,:], Imat_b[None,None])
        Imat = tutil.qaction(quat[...,None,:], einops.rearrange(Imat, '... i j -> ... j i'))
        Imat = einops.rearrange(Imat, '... i j -> ... j i')
        Imat_inv = jnp.linalg.inv(Imat)

        Iw = jnp.einsum('...ij,...j->...i',Imat, twist[...,3:])
        twist = twist.at[...,:3].set(twist[...,:3] + dp_o[...,:3]/mass + ext_wrench[...,:3]/mass*dt)
        twist = twist.at[...,3:].set(twist[...,3:] + jnp.einsum('...ij,...j->...i',Imat_inv, dp_o[...,3:]) + jnp.einsum('...ij,...j->...i',Imat_inv, jnp.cross(twist[...,3:], Iw)*dt))
        twist = twist.at[...,fix_idx,:].set(0)

        pos += twist[...,:3] * dt
        quat = tutil.qmulti(tutil.qexp(twist[...,3:]*dt/2), quat)

    return pos, quat, twist

dynamics_step_jit = jax.jit(dynamics_step, static_argnames=['cp_no', 'cull_k', 'mass', 'inertia', 'substep'])

# %% cd test
# _, jkey = jax.random.split(jkey)
# inputs_cd = (jnp.array([[0,1,1,1]], dtype=jnp.float32), jnp.array([[0.,0.,0.07]], dtype=jnp.float32), jnp.array([[0,0,0,1]], dtype=jnp.float32), jnp.array([[0.080]], dtype=jnp.float32))
# inputs_cd = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=1), inputs_cd)

# # add table - should be index 0!!!
# plane_param = (jnp.array([[0,1,1,0.05]]), jnp.array([[0,0,-0.05]]), jnp.array([[0,0,0,1]]), jnp.array([[1]]))
# plane_param = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=1), plane_param)
# inputs_cd = jax.tree_map(lambda *x : jnp.concatenate(x, axis=1)[...,None,:,:], plane_param, inputs_cd)

# culli = jnp.zeros_like(inputs_cd[-1][...,0,0])
# cps, cns, sdfs = calculate_contact_points(jkey, 20, *inputs_cd, culli, visualize=True)

# %%
# physical test init
prutil.init()

# %%
# param random
NB = 100
NO = 30
inputs = random_param(jkey, (NB,NO))

# test primitives
# inputs = (jnp.array([[3,1,1,1]], dtype=jnp.float32), jnp.array([[0.1,0.1,0.10]], dtype=jnp.float32), jnp.array([[np.sin(np.pi/12),0,0,np.cos(np.pi/12)]], dtype=jnp.float32), jnp.array([[0.080]], dtype=jnp.float32))
# inputs = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=NB), inputs)

# add table - should be index 0!!!
plane_param = (jnp.array([[0,1,1,0.05]]), jnp.array([[0,0,-0.05]]), jnp.array([[0,0,0,1]]), jnp.array([[1]]))
plane_param = jax.tree_map(lambda x : einops.repeat(x, 'i j -> tile i j', tile=NB), plane_param)
inputs = jax.tree_map(lambda *x : jnp.concatenate(x, axis=1), plane_param, inputs)
geo_param, pos, quat, scale = inputs
prutil.make_objs_primitives(geo_param[-1], pos[-1], quat[-1], scale[-1])

# %%
# dynamics step
geo_param, pos, quat, scale = inputs
physics_param = {}
physics_param['mass'] = 1.0
physics_param['inertia'] = 0.02
physics_param['dt'] = 0.002
physics_param['substep'] = 10
physics_param['cull_k'] = 80
physics_param['baumgarte_erp'] = 40.0
physics_param['elasticity'] = 0
physics_param['mu'] = 0.8
physics_param['cp_no'] = 18
physics_param['fix_idx'] = jnp.array([0]).astype(jnp.int32)
twist = jnp.zeros(pos.shape[:-1] + (6,), dtype=jnp.float32)
frames = []
fps = 20
pp = int(1/physics_param['dt']/physics_param['substep']/fps)
st = time.time()
for i in range(200):
    _, jkey = jax.random.split(jkey)
    gv_force = -9.81 * jnp.ones_like(pos) * physics_param['mass']
    gv_force = gv_force.at[:,:,:2].set(0)
    ext_wrench = jnp.concatenate([gv_force, jnp.zeros_like(gv_force)], axis=-1)
    pos, quat, twist = dynamics_step_jit(jkey, geo_param, pos, quat, scale, twist, ext_wrench, **physics_param)
    # pos, quat, twist = dynamics_step(jkey, geo_param, pos, quat, scale, twist, ext_wrench, **physics_param)

    # pybullet capture
    if i%pp == 0:
        print('dt : {}'.format((time.time() - st)/pp))
        rpos, rquat = pos[-1], quat[-1]
        for j in range(rpos.shape[0]):
            p.resetBasePositionAndOrientation(j, rpos[j], rquat[j])
        frames.append(prutil.get_rgb([0,0.6,0.6]))
        st = time.time()

clip = ImageSequenceClip(list(frames), fps=fps)
clip.write_gif('test.gif', fps=fps)

Image('test.gif')

# %%
# %timeit dynamics_step_jit(jkey, geo_param, pos, quat, scale, twist, ext_wrench, **physics_param)