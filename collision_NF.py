# %%
# import libraries
import jax
import jax.numpy as jnp
import numpy as np
import pybullet as p
import flax.linen as nn
import optax
import itertools
from moviepy.editor import ImageSequenceClip
from IPython.display import display, Image
import matplotlib.pyplot as plt
import util.transform_util as tutil
import util.transform_util_np as tutilnp
import einops
import fcl
from scipy.spatial.transform.rotation import Rotation as sciR
import random
import pickle
import jaxlie

PEN_CLIP= 0.010

# %%
# query fcl
def fcl_query(type, gparam, pos, quat):
    o_list = []
    for ty, gp in zip(type, gparam):
        ty = np.argmax(ty)
        if ty == 0:
            g1 = fcl.Box(2*gp[0], 2*gp[1], 2*gp[2])
        elif ty == 1:
            g1 = fcl.Capsule(gp[0], gp[1])
        elif ty == 2:
            g1 = fcl.Cylinder(gp[0], gp[1])
        o_list.append(fcl.CollisionObject(g1, fcl.Transform()))

    drequest = fcl.DistanceRequest()
    crequest = fcl.CollisionRequest(enable_contact=True)

    for o, ps, q in zip(o_list, pos, quat):
        o.setTransform(fcl.Transform(sciR.from_quat(q).as_matrix(), ps))
    
    cresult = fcl.CollisionResult()
    cret = fcl.collide(*o_list, crequest, cresult)
    if cret != 0:
        ans = -cresult.contacts[0].penetration_depth
        dir = cresult.contacts[0].normal
    else:
        dresult = fcl.DistanceResult()
        ans = fcl.distance(*o_list, drequest, dresult)
        if ans == -1:
            ans = 0
        dir = dresult.nearest_points[1] - dresult.nearest_points[0]

    return ans, dir

def batch_query(type, gparam, pos, quat):
    ans = np.zeros([type.shape[0],1])
    dir = np.zeros([type.shape[0],3])
    for i in range(type.shape[0]):
        ans[i], dir[i] = fcl_query(type[i], gparam[i], pos[i], quat[i])
    dir = dir / (np.linalg.norm(dir, axis=-1, keepdims=True) + 1e-7)
    return ans, dir

# %%
def x_sample(outer_shape, pos_scale=0.13):
    type = (np.random.randint(0, 3, size=outer_shape+[1]) == np.arange(3)).astype(np.float32)
    # type = (np.random.randint(0, 1, size=outer_shape+[1]) == np.arange(3)).astype(np.float32) # test
    gparam = np.random.uniform(0.03,0.06, size=outer_shape+[3])
    pos = pos_scale*tutil.qrand(size=outer_shape+[4])[...,:3]
    quat = tutil.qrand(size=outer_shape+[4])

    # # # test data
    # ns = outer_shape[0]
    # type = np.array([[[1,0,0],[1,0,0]]])
    # gparam = np.array([[[0.05,0.032,0.055],[0.05,0.035,0.058]]])
    # ang = np.random.uniform(-np.pi, np.pi, size=[ns*2,1])
    # quat = sciR.from_euler('x', ang).as_quat().reshape(ns,2,4)
    # #quat[:,0,:] = [0,0,0,1]
    # type, gparam = [einops.repeat(e, 'i j k -> (i tile) j k', tile=ns) 
    #             for e in (type, gparam)]
    # pos = np.random.uniform(-0.15,0.15,size=[ns,2])
    # pos = np.concatenate([np.zeros_like(pos[...,:1]), pos], axis=-1)
    # pos = np.stack([np.zeros_like(pos), pos], axis=-2)

    return type, gparam, pos, quat

def x_sample_noise(x, ns):
    type, gparam, pos, quat = x
    gparam += np.random.normal(scale=0.01, size=gparam.shape)
    gparam = np.clip(gparam, 0.02, 0.05)
    pos += np.random.normal(scale=0.01, size=gparam.shape)
    quat = tutilnp.qnoise(quat)
    return type, gparam, pos, quat

def make_dataset(ns):
    while True:
        x = x_sample([ns,2])
        y, dir = batch_query(*x)

        res = [(x_[np.where(y[...,0]>0)], x_[np.where(y[...,0]<=0)]) for x_ in  x + (y,dir)]
        pnx, pny, pndir = res[:-2], res[-2], res[-1]
        px, nx = [x_[0] for x_ in pnx], [x_[1] for x_ in pnx]
        py, ny = pny
        pdir, ndir = pndir
        p_cnt, n_cnt = py.shape[0], ny.shape[0]

        def generate_proximity_data(x, len, dir, n=5, uniform_len=False):
            type, gparam, pos, quat = x
            if not uniform_len:
                pos[:,0] += dir * len
            type, gparam, pos, quat = [einops.repeat(e, '... i j -> ... tile i j', tile=n) for e in (type, gparam, pos, quat)]
            if uniform_len:
                len_distribution = len[...,None] * np.random.uniform(size=[len.shape[0],n,1])
                pos[...,0,:] += dir[...,None,:] * len_distribution
                return (type, gparam, pos, quat), len[...,None]-len_distribution
            else:
                len_distribution = np.random.normal(scale=0.02, size=[len.shape[0],n,1])
                len_distribution = np.abs(len_distribution)
                len_distribution[np.where(len[...,0]<0)] *= -1
                pos[...,0,:] += -len_distribution * dir[...,None,:]
                return (type, gparam, pos, quat), len_distribution

        px_gen, py_gen = generate_proximity_data(px, py, pdir, n=2)
        # py_gt, pdir_gt = batch_query(*[e.reshape(-1,2,e.shape[-1]) for e in px_gen])
        # py_gt = py_gt.reshape(-1,5,1)
        # pdir_gt = pdir_gt.reshape(-1,5,3)
        px_gen = [e.reshape(-1,2,e.shape[-1]) for e in px_gen]

        nx_gen, ny_gen = generate_proximity_data(nx, ny, ndir, n=int(2*p_cnt/n_cnt), uniform_len=True)
        # ny_gt, ndir_gt = batch_query(*[e.reshape(-1,2,e.shape[-1]) for e in nx_gen])
        # ny_gt = ny_gt.reshape(-1,5,1)
        # ndir_gt = ndir_gt.reshape(-1,5,3)
        nx_gen = [e.reshape(-1,2,e.shape[-1]) for e in nx_gen]

        x = [np.concatenate(e, axis=0).reshape(-1,2,e[0].shape[-1]) for e in zip(px_gen, nx_gen, x)]
        y = np.concatenate([py_gen.reshape(-1,1), ny_gen.reshape(-1,1), y], axis=0).reshape(-1,1)

        return x, y
        
test_data = make_dataset(ns=100)


# %%
p.connect(p.DIRECT)

# %%
# get rgb function
def get_rgb(cam_pos = [0,0.45,0.45]):
    PIXEL_SIZE = [500,500]
    fov = 60
    near = 0.1
    far = 4.0
    pm = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=near, farVal=far)
    vm = p.computeViewMatrix(cam_pos,[0,0,0],[0,0,1])

    img_out = p.getCameraImage(*PIXEL_SIZE, viewMatrix=vm, projectionMatrix=pm)
    rgb = np.array(img_out[2]).reshape([*PIXEL_SIZE, 4])[...,:3]

    return rgb

def random_pos_quat(close=True):
    if close:
        return (np.random.uniform([-0.09,-0.09,-0.09],[0.09,0.09,0.09]),
            tutil.qrand(size=[4]))
    else:
        return (np.random.uniform([-0.2,-0.2,-0.2],[0.2,0.2,0.2]),
            tutil.qrand(size=[4]))

def make_objs(obj_no, pos_scale=0.4):
    
    x = x_sample([obj_no], pos_scale=pos_scale)
    p.resetSimulation()
    obj_id_list = []
    obj_type_list = []

    for ty, gp, ps, qt in zip(*x):
        ty = np.argmax(ty)
        if ty == 0:
            obj_id_list.append(p.createMultiBody(baseMass=1,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=gp),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=gp,
                                                        rgbaColor=np.random.uniform(0,1,size=4)),
                    ))
        elif ty == 1:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=gp[0], height=gp[1]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=gp[0], length=gp[1],rgbaColor=np.random.uniform(0,1,size=4)),
            ))
        elif ty == 2:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=gp[0], height=gp[1]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=gp[0], length=gp[1],rgbaColor=np.random.uniform(0,1,size=4)),
            ))
        elif ty == 3:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.07),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.07,rgbaColor=np.random.uniform(4)),
            ))
        else:
            raise ValueError
        
        p.resetBasePositionAndOrientation(obj_id_list[-1], ps, qt)
    return x
    

# %%
# define CNF

class CNF(nn.Module):
    penetration_clip: float
    feature_dim: int = 256

    def positional_embedding(self, x, embedding_size=10):
        pe = []
        for i in range(embedding_size):
            pe.append(jnp.cos((2**i)*np.pi*x))
            pe.append(jnp.sin((2**i)*np.pi*x))
        x = jnp.concatenate(pe, axis=-1)
        return x

    @nn.compact
    def col_query(self, type1, gparam1, type2, gparam2, pos2, quat2):
        gparam1 = nn.BatchNorm(True)(gparam1)
        gparam2 = nn.BatchNorm(True)(gparam2)
        rel_pos_limit = 0.20
        quat_mask = (quat2[...,-1:] > 0).astype(jnp.float32)
        quat = quat2 * quat_mask + (-quat2) * (1-quat_mask)
        far_mask = jnp.any(jnp.abs(pos2) > rel_pos_limit, axis=-1).astype(jnp.float32)
        pos = self.positional_embedding(pos2/rel_pos_limit)
        quat = self.positional_embedding(quat, 6)
        gparam1 = self.positional_embedding(gparam1, 6)
        gparam2 = self.positional_embedding(gparam2, 6)

        x = jnp.concatenate([type1, gparam1, type2, gparam2, pos, quat], axis=-1)

        for i in range(2):
            x = nn.Dense(self.feature_dim)(x)
            x = nn.relu(x)
            if i == 0:
                skip = x
        x += skip

        for i in range(2):
            x = nn.Dense(self.feature_dim)(x)
            x = nn.relu(x)
            if i == 0:
                skip = x
        x += skip
        
        x = nn.Dense(1)(x)
        x = self.penetration_clip*nn.tanh(x)
        x = jnp.squeeze(x, axis=-1)
        return x * (1-far_mask) + self.penetration_clip*far_mask

    # def penetration_and_constraint(self, param, inputs):
        
    
    def __call__(self, type, gparam, pos, quat):
        
        NN = pos.shape[-2]
        typei, gparami, posi, quati = jax.tree_map(
                            lambda x : einops.repeat(x, 'i j k -> i j tile k', tile=NN),
                                         (type, gparam, pos, quat))
        typej, gparamj, posj, quatj = jax.tree_map(
                            lambda x : einops.repeat(x, 'i j k -> i tile j k', tile=NN),
                                         (type, gparam, pos, quat))

        pos_dif, quat_dif = tutil.pq_multi(*tutil.pq_inv(posi, quati), posj, quatj)

        inputs = (typei, gparami, typej, gparamj, pos_dif, quat_dif)
        uidx = jnp.triu_indices(NN, k=1)
        res1 = [x[...,uidx[0],uidx[1],:] for x in inputs]
        res2 = [einops.rearrange(x, 'i j k d -> i k j d')[...,uidx[0],uidx[1],:] for x in inputs]
        
        inputs = jax.tree_map(lambda *x : jnp.stack(x, axis=-2), res1, res2)

        col_res = self.col_query(*inputs)
        if col_res.shape[-2] == 1:
            return jnp.squeeze(col_res, axis=-2)
        else:
            return jnp.mean(col_res, axis=-1)

jkey = jax.random.PRNGKey(0)
model = CNF(penetration_clip=PEN_CLIP)
param = model.init(jkey, *test_data[0])

model_apply_jit = jax.jit(model.apply)
# %%
# test
res = model.apply(param, *test_data[0])

# %%
# def loss
def loss_func(param, x, y):
    y_clip = jnp.clip(y, -PEN_CLIP, PEN_CLIP)
    yp = model.apply(param, *x)
    # loss = jnp.mean(yp[...,0,0:1]**2+(yp[...,0,1:2]-y)**2 + (yp[...,1,0:1]-y)**2, axis=-1)
    loss = jnp.mean(jnp.abs(yp-y_clip), axis=-1)
    return jnp.mean(loss)

loss_func(param, *test_data)
loss_func_jit = jax.jit(loss_func, static_argnums=[3])
loss_func_value_and_grad = jax.value_and_grad(loss_func)

# %%
# train step
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(param)
step_no = 1
def train_step(param, opt_state, x, y):
    # gap = int(x[0].shape[0]/step_no)
    # for i in range(step_no):
    #     x_ = [xe[gap*i:gap*(i+1)] for xe in x]
    #     value, grad = loss_func_value_and_grad(param, x_, y[gap*i:gap*(i+1)])
    #     updates, opt_state = optimizer.update(grad, opt_state)
    #     param = optax.apply_updates(param, updates)
    value, grad = loss_func_value_and_grad(param, x, y)
    updates, opt_state = optimizer.update(grad, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state, value, model.apply(param, *x)

train_step_jit = jax.jit(train_step)

# %%
# replay buffer
class replay_buffer:
    def __init__(self, capacity=8000):
        self.capacity = capacity
        self.total_data = None
        self.priority = None

    def push(self, data, data_num, default_priority=2.0):
        if self.total_data is None:
            self.total_data = data
            self.priority = default_priority*np.ones(data_num)
        else:
            sort_idx = np.argsort(self.priority)
            self.total_data, self.priority = jax.tree_map(lambda x : x[sort_idx], [self.total_data, self.priority])
            
            self.total_data = jax.tree_map(lambda *x : np.concatenate(x, axis=0)[-self.capacity:], self.total_data, data)
            self.priority = np.concatenate([self.priority, default_priority*np.ones(data_num)], axis=0)[-self.capacity:]


    def sample(self, n):
        tn = self.priority.shape[0]
        idx = np.random.choice(np.arange(tn), size=[n], p=self.priority/np.sum(self.priority))
        pick_td, pick_pty = jax.tree_map(lambda x : x[idx], [self.total_data, self.priority])
        return pick_td, pick_pty, idx
    
    def update_priority(self, idx, priority):
        self.priority[idx] = priority

# %%
# draw prediction results

def eval_draw(param, ang1=0, ang2=0):
    ns = 10000
    type = np.array([[[1,0,0],[1,0,0]]])
    gparam = np.array([[[0.05,0.032,0.055],[0.05,0.035,0.058]]])
    quat = np.array([[sciR.from_euler('x', ang1).as_quat(),sciR.from_euler('x', ang2).as_quat()]])
    type, gparam, quat = [einops.repeat(e, 'i j k -> (i tile) j k', tile=ns) 
                for e in (type, gparam, quat)]
    pos = np.random.uniform(-0.15,0.15,size=[ns,2])
    pos = np.concatenate([np.zeros_like(pos[...,:1]), pos], axis=-1)
    pos = np.stack([np.zeros_like(pos), pos], axis=-2)

    res_gt, dir_gt = batch_query(type, gparam, pos, quat)
    res_gt = np.clip(res_gt, -PEN_CLIP, PEN_CLIP)

    res = model.apply(param, type, gparam, pos, quat)
    res = np.mean(res, axis=-1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(pos[:,1,1], pos[:,1,2], s=5, c=res)
    plt.subplot(1,2,2)
    plt.scatter(pos[:,1,1], pos[:,1,2], s=5, c=res_gt)
    plt.show()
# %%
# start train

def get_priority(yp, y):
    tau = 0.02
    pred_dif = np.mean(np.abs(y - yp), axis=-1)
    # log_val = -np.abs(np.mean(yp, axis=-1)) + np.std(yp, axis=-1) + pred_dif
    log_val = np.std(yp, axis=-1) + pred_dif
    log_val -= np.max(log_val)
    return np.exp(log_val/tau)

rb = replay_buffer()
for i in range(9001):
    x, y = make_dataset(ns=800)
    rb.push((x, y), data_num = y.shape[0])
    
    for k in range(8):
        data, _, data_idx = rb.sample(512)
        param, opt_state, value, yp = train_step_jit(param, opt_state, *data)
        rb.update_priority(data_idx, get_priority(yp, data[1]))
    if i % 100 == 0:
        print("loss {} // pred {} // gt {}".format(value, yp[-2:,0], y[-2:,0]))
    if i % 1000 == 0:
        eval_draw(param, ang2=0)
        eval_draw(param, ang1=90*np.pi/180)
        eval_draw(param, ang2=45*np.pi/180)
        eval_draw(param, ang2=60*np.pi/180)

with open('param.pickle', 'wb') as handle:
    pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
# define jit func
with open('param.pickle', 'rb') as handle:
    param = pickle.load(handle)

constraint_grad = jax.jacobian(lambda x, type, gparam: jnp.sum(model.apply(param, type, gparam, *x), axis=0))
constraint_grad = jax.jit(constraint_grad)
model_apply_jit = jax.jit(model.apply)


# %%
x = make_objs(20, 0.15)

# %%
# physics evaulation
def physics_eval(x):
    type, gparam, pos, quat = [e[None] for e in x]
    NN = type.shape[1]
    dt = 0.001
    mass = 1.0
    frames = []
    fps = 20
    pp = 1/dt/fps
    free_idx = np.arange(12)
    #pos[:,free_idx,2] += 0.2
    tvel = np.zeros_like(pos)
    avel = np.zeros_like(pos)
    for i in range(5000):
        # print(i)
        penetration_value = model_apply_jit(param, type, gparam, pos, quat)
        penetration_value = jnp.maximum(-penetration_value-0.001, 0)
        constraints = constraint_grad((pos,quat), type, gparam)
        constrain_quat = constraints[1]

        zeta = tutil.qmulti(tutil.qinv(quat), constrain_quat)[...,:3] # log quaternion
        constraint_concat = jnp.concatenate([constraints[0], zeta], axis=-1)
        const_norm = jnp.maximum(jnp.linalg.norm(constraint_concat, axis=-1, keepdims=True), 1e-6)
        constraint_concat = constraint_concat / const_norm
        
        constraint_concat = jnp.sum(constraint_concat * jnp.transpose(penetration_value, axes=[1,0])[:,:,None,None], axis=0)
        const_pos, const_zeta = constraint_concat[...,:3], constraint_concat[...,3:]
        
        gain = 600.0 * dt
        ang_gain = 500.0*dt
        inertia = 0.01
        pos_impulse = gain * const_pos
        ang_impulse = ang_gain * const_zeta
        gv_center = [0.05*np.sin(i*dt*5), -0.1*np.sin(i*dt*2), 0.05*np.cos(i*dt*5)]
        gv_center = np.array(gv_center)
        gv_acc = - 6.0 * (pos-gv_center)
        #gv_acc = np.array([0,0,-1])
        tvel[:,free_idx] = (tvel + pos_impulse/mass + gv_acc*dt)[:,free_idx] 
        avel[:,free_idx] = (avel + ang_impulse/inertia)[:,free_idx]
        
        # corriori term....

        pos[:,free_idx] = (pos + tvel*dt)[:,free_idx]  # last term => gravity
        quat[:,free_idx] = tutil.qmulti(quat, tutil.qexp(2*avel*dt))[:,free_idx]
        #quat = quat/jnp.linalg.norm(quat, axis=-1, keepdims=True)


        # pybullet test
        if i%pp == 0:
            rpos, rquat = pos[-1], quat[-1]
            for i in range(rpos.shape[0]):
                p.resetBasePositionAndOrientation(i, rpos[i], rquat[i])
            frames.append(get_rgb())

    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif('test.gif', fps=fps)

physics_eval(x)
Image('test.gif')

# %%