# %%
# import libraries
import brax
import jax
import jax.numpy as jnp
import numpy as np
from brax.io import html
from IPython.display import HTML
import time
import argparse
parser = argparse.ArgumentParser(description='parsing')
parser.add_argument('--num_envs', default=10,
                    help='hlef')

args = parser.parse_args()


import util.transform_util as tutil

jkey = jax.random.PRNGKey(0)

# %%
# define hyper parameters
BOXN = 0
SPHEREN = 14

SUBSTEP = 10
STEPNO = 100
INITSTEPNO = 200

BATCHNO = int(args.num_envs)

# %%
# build environment
# N box M ball
brax_envs = brax.Config(dt=0.1, substeps=SUBSTEP)
# brax_envs = brax.Config(dt=0.002, substeps=2)

# brax_envs.dynamics_mode = 'legacy_spring'

ground = brax_envs.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof
brax_envs.gravity.z = -9.8
brax_envs.friction = 0.8

for i in range(BOXN):
    box_b = brax_envs.bodies.add(name='box'+str(i), mass=1)
    box_c = box_b.colliders.add().box
    box_c.halfsize.x, box_c.halfsize.y, box_c.halfsize.z = 0.3, 0.3, 0.3
    
for i in range(SPHEREN):
    ball_b = brax_envs.bodies.add(name='ball' + str(i), mass=1)
    ball_c = ball_b.colliders.add().capsule
    ball_c.radius, ball_c.length = 0.5, 1.5
    # ball_c = ball_b.colliders.add().sphere
    # ball_c.radius = 0.5

# %%
# build grippers


# %%
# init brax and step test
# batch_no = 200
sys = brax.System(brax_envs)
qp = sys.default_qp()

for i in range(qp.pos.shape[0]):
    if i==0:
        continue
    _, jkey = jax.random.split(jkey)
    qp.pos[i] = jax.random.uniform(jkey, shape=[3,], dtype=jnp.float32, minval=jnp.array([-3,-3,1.6]), maxval=jnp.array([3,3,5.0]))
    qp.ang[i] = tutil.q2aa(tutil.qrand(outer_shape=(), jkey=jkey))

qp_list = [qp]
qp = jax.tree_map(lambda *args: jnp.stack(args), *[qp] * BATCHNO)
step_vmap = jax.vmap(lambda x : sys.step(x, []))

step_jit = jax.jit(step_vmap)

# @jax.jit
# def step_jit(qp):
#     for j in range(SUBSTEP):
#         qp, _ = step_vmap(qp)
#     return qp, _

# step_jit = jax.jit(lambda x : sys.step(x, []))
step_jit(qp)
for i in range(STEPNO+INITSTEPNO):
    if i==INITSTEPNO:
        st = time.time()
    qp, _ = step_jit(qp)
    # qp_list.append(jax.tree_map(lambda x : x[0], qp))
et = time.time()
res = [BATCHNO, et - st, (et - st)/STEPNO/SUBSTEP, (et - st)/STEPNO/SUBSTEP/BATCHNO]
# print(res)
# print(BATCHNO)
# print(et - st)
# # print((et - st)/STEPNO)
# print((et - st)/STEPNO/SUBSTEP)
# print((et - st)/STEPNO/SUBSTEP/BATCHNO)


f = open("BRAX_SPHERE.txt", "a")
for re in res:
    print(re)
    f.write(str(re)+" ")
f.write("\n")
f.close()

# HTML(html.render(sys, qp_list))

# %%
