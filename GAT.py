# %%
# import libraries
# import jraph
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jax
import optax
import pybullet as p
import glob
import matplotlib.pyplot as plt
import random
import itertools
from moviepy.editor import ImageSequenceClip
from IPython.display import display, Image


timestep = 0.001
obj_no = 6
PIXEL_SIZE = [500,500]
fov = 60
near = 0.1
far = 4.0
NS = 8
NR = 16
NB = NS*NR
penetration_depth_alpha= 10
obj_category_no = 5
obj_mesh_list = glob.glob('obj_mesh/*/meshes/model.obj')[:obj_category_no]
p.connect(p.DIRECT)
p.setTimeStep(timestep)

# %%
# dataset

def random_pos_quat(close=True):
    quat_rand = np.random.normal(size=[4])
    quat_rand = quat_rand / np.linalg.norm(quat_rand)
    if close:
        return (np.random.uniform([-0.20,-0.10,-0.10],[0.20,0.10,0.10]),
            quat_rand)
    else:
        return (np.random.uniform([-0.5,-0.5,-0.5],[0.5,0.5,0.5]),
            quat_rand)

def make_dataset():
    obj_type_stack = []
    pos_stack = []
    quat_stack = []
    con_res_stack = []

    for i in range(NS):
        p.resetSimulation()
        obj_id_list = []
        obj_type_list = []
        # obj_id_list.append(p.createMultiBody(baseMass=0,
        #                 baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.01]),
        #                 baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.01],
        #                                             rgbaColor=[0.9,0.2,0.2,1.0]),
        #                 ))
        # obj_type_list.append(-1)

        for _ in range(obj_no):
            # object model
            # obj_file_name = random.choice(obj_mesh_list)
            # obj_id_list.append(p.createMultiBody(
            #     baseMass=1,
            #     baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=obj_file_name),
            #     baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=obj_file_name),
            # ))
            # obj_type_list.append(obj_mesh_list.index(obj_file_name))

            # primitive generations
            primitive_idx = np.random.randint(0,4)
            if primitive_idx == 0:
                obj_id_list.append(p.createMultiBody(baseMass=1,
                        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06,0.04,0.07]),
                        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06,0.04,0.07],
                                                            rgbaColor=np.random.uniform(0,1,size=4)),
                        ))
            elif primitive_idx == 1:
                obj_id_list.append(p.createMultiBody(
                    baseMass=1,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.07),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.07,rgbaColor=np.random.uniform(4)),
                ))
            elif primitive_idx == 2:
                obj_id_list.append(p.createMultiBody(
                    baseMass=1,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=0.05, height=0.02),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=0.05, length=0.02,rgbaColor=np.random.uniform(0,1,size=4)),
                ))
            elif primitive_idx == 3:
                obj_id_list.append(p.createMultiBody(
                    baseMass=1,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.06, height=0.06),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.06, length=0.06,rgbaColor=np.random.uniform(0,1,size=4)),
                ))
            else:
                raise ValueError
            obj_type_list.append(primitive_idx)

        obj_type_list = np.stack(obj_type_list)

        for j in range(NR):
            close_rand = np.random.uniform() < 0.85
            pos_list = []
            quat_list = []
            for oid in obj_id_list:
                rpos, rquat = random_pos_quat(close_rand)
                p.resetBasePositionAndOrientation(oid, rpos, rquat)
                pos_list.append(rpos)
                quat_list.append(rquat)
            p.stepSimulation()
            con_res_list = []
            for a, b in itertools.combinations(obj_id_list, 2):
                # conres = p.getContactPoints(a, b)
                # con_res_list.append(int(len(conres) != 0))
                conres = p.getClosestPoints(a, b, distance=0.05)
                if len(conres) == 0:
                    penetration_depth = 0.05
                else:
                    penetration_depth = [cr[8] for cr in conres]
                    penetration_depth = np.mean(penetration_depth)
                if penetration_depth < 0:
                    penetration_depth = penetration_depth_alpha * penetration_depth
                con_res_list.append(penetration_depth)
            pos_list = np.stack(pos_list)
            quat_list = np.stack(quat_list)

            # data stack
            obj_type_stack.append(obj_type_list)
            pos_stack.append(pos_list)
            quat_stack.append(quat_list)
            con_res_stack.append(con_res_list)

    obj_type_stack = np.stack(obj_type_stack)
    pos_stack = np.stack(pos_stack)
    quat_stack = np.stack(quat_stack)
    con_res_stack = np.stack(con_res_stack)

    # make data
    oh_obj_type = (obj_type_stack[...,None] == np.arange(obj_category_no)).astype(np.float32)
    # input = np.concatenate([oh_obj_type, pos_stack, quat_stack], axis=-1)
    input = oh_obj_type, pos_stack, quat_stack

    return input, con_res_stack

test_data = make_dataset()

# %%
# get rgb function
def get_rgb(cam_pos = [0,0.45,0.45]):

    pm = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=near, farVal=far)
    vm = p.computeViewMatrix(cam_pos,[0,0,0],[0,0,1])

    img_out = p.getCameraImage(*PIXEL_SIZE, viewMatrix=vm, projectionMatrix=pm)
    rgb = np.array(img_out[2]).reshape([*PIXEL_SIZE, 4])[...,:3]

    return rgb

plt.figure()
plt.imshow(get_rgb())
plt.axis('off')
plt.show()
# %%
# define GNN
class GAT(nn.Module):
    feature_dim: int = 128
    multihead_no: int = 4

    def positional_embedding(self, x, embedding_size=10):
        pe = []
        for i in range(embedding_size):
            pe.append(jnp.cos((2**i)*np.pi*x))
            pe.append(jnp.sin((2**i)*np.pi*x))
        x = jnp.concatenate(pe, axis=-1)
        return x

    @nn.compact
    def __call__(self, type, pos, quat):
        '''
        x : (NB, NN, ND)
        '''
        NN = type.shape[-2]
        pos = self.positional_embedding(pos)
        # quat = self.positional_embedding(pos, embedding_size=4)
        x = jnp.concatenate([type, pos, quat], axis=-1)

        # baseline model
        for _ in range(2):
            x = nn.Dense(self.feature_dim)(x)
            x = jax.nn.relu(x)
            # x = nn.LayerNorm()(x)

        # # GAT model
        # for j in range(1):
        #     mh_hp = []
        #     for i in range(self.multihead_no):
        #         wh = nn.Dense(self.feature_dim, use_bias=False)(x)
        #         wh1 = jnp.tile(wh[...,None,:], [1,1,NN,1])
        #         wh2 = jnp.tile(wh[...,None,:,:], [1,NN,1,1])
        #         wh12 = jnp.concatenate([wh1,wh2], axis=-1)
        #         # a = self.param('a'+str(i), nn.initializers.lecun_normal(), (1, 2*self.feature_dim,), jnp.float32)
        #         a = self.param('a'+str(i)+str(j), nn.initializers.lecun_normal(), (1, self.feature_dim,), jnp.float32)
        #         a = jnp.concatenate([a,a], axis=-1)
        #         # a = a/(1e-6+jnp.linalg.norm(a))
        #         alpha = jnp.sum(wh12 * a, axis=-1) # (NB, NN, NN)
        #         alpha = jax.nn.leaky_relu(alpha)
        #         alpha = nn.softmax(alpha, axis=-1)
        #         hp = jnp.sum(wh2 * alpha[...,None], axis=-2)
        #         hp = jax.nn.leaky_relu(hp)
        #         mh_hp.append(hp)
        #     mh_hp = jnp.concatenate(mh_hp, axis=-1)
        #     x = mh_hp

        # final edge output
        edges = []
        for a, b in itertools.combinations(np.arange(NN), 2):
            pick_node = x[...,(a,b),:]
            edges.append(jnp.sum(pick_node, axis=-2))
        edges = jnp.stack(edges, axis=-2)

        for _ in range(2):
            edges = nn.Dense(self.feature_dim)(edges)
            edges = nn.relu(edges)
        edges = nn.Dense(1)(edges)
        # edges = nn.sigmoid(edges)
        edges = jnp.squeeze(edges, axis=-1)
        return edges

jkey = jax.random.PRNGKey(0)
model = GAT()
param = model.init(jkey, *test_data[0])

# %%
# loss func
def loss_func(param, x, y):
    yp = model.apply(param, *x)
    # yp = jnp.clip(yp, 1e-7, 1-1e-7)

    # focal loss
    # p = y*yp + (1-y)*(1-yp)
    # gamma = 1
    # loss = -jnp.mean(jax.lax.stop_gradient(1-p)**gamma*jnp.log(p), axis=-1)
    
    # BCE
    # loss = -jnp.mean(y*jnp.log(yp) + (1-y)*jnp.log(1-yp), axis=-1)
    loss = jnp.mean((yp-y)**2, axis=-1)
    return jnp.mean(loss)

loss_func_value_and_grad = jax.value_and_grad(loss_func)


# %%
# train step
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(param)
step_no = 4
def train_step(param, opt_state, x, y):
    gap = int(x[0].shape[0]/step_no)
    for i in range(step_no):
        x_ = [xe[gap*i:gap*(i+1)] for xe in x]
        value, grad = loss_func_value_and_grad(param, x_, y[gap*i:gap*(i+1)])
        updates, opt_state = optimizer.update(grad, opt_state)
        param = optax.apply_updates(param, updates)
    return param, opt_state, value

train_step_jit = jax.jit(train_step)

# %%
# start train
for i in range(1200):
    x, y = make_dataset()
    param, opt_state, value = train_step_jit(param, opt_state, x, y)
    # print(i, value)
    if i % 100 == 0:
        yp = model.apply(param, *x)
        acc = 1-jnp.abs(yp - y)
        acc = jnp.mean(acc)
        print("acc {} // pred {} // gt {}".format(acc, yp[-1], y[-1]))
        plt.figure()
        plt.imshow(get_rgb())
        plt.axis('off')
        plt.show()
        
# %%
# define jit func
constraint_grad = jax.jacobian(lambda x, type: jnp.sum(model.apply(param, type, *x), axis=0))
constraint_grad = jax.jit(constraint_grad)
model_apply_jit = jax.jit(model.apply)

# %%
x, y = make_dataset()

# %%
# physics evaulation
def physics_eval(x):
    type, pos, quat = x
    NN = type.shape[1]
    dt = 0.001
    mass = 1.0
    frames = []
    fps = 20
    pp = 1/dt/fps
    for i in range(3000):
        # print(i)
        penetration_value = model_apply_jit(param, type, pos, quat)
        penetration_value = jnp.maximum(-penetration_value/penetration_depth_alpha+0.005, 0)
        constraints = constraint_grad((pos,quat), type)
        gain = 6.0 * dt
        pos_impulse = gain * jnp.sum(constraints[0] * jnp.transpose(penetration_value, axes=[1,0])[:,:,None,None], axis=0)
        gv_center = [0.2*np.sin(i*dt*5), -0.2*np.sin(i*dt*2), 0.2*np.cos(i*dt*5)]
        gv_center = np.array(gv_center)
        gv_acc = - 3.0 * (pos-gv_center)
        pos = pos + pos_impulse/mass + gv_acc*dt  # last term => gravity

        quat_gain = 10000.0*dt
        quat_constraint_prj = constraints[1] - quat*jnp.sum(constraints[1] * quat, axis=-1, keepdims=True)
        quat_impulse = quat_gain * jnp.sum(quat_constraint_prj * jnp.transpose(penetration_value, axes=[1,0])[:,:,None,None], axis=0)
        quat = quat + quat_impulse/mass
        quat = quat/jnp.linalg.norm(quat, axis=-1, keepdims=True)

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