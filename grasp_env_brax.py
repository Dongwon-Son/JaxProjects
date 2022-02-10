# %%
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import brax
from brax.io import html
from brax.io import image
from IPython.display import HTML, Image 
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

def draw_system(ax, pos, alpha=1):
  for i, p in enumerate(pos):
    ax.add_patch(Circle(xy=(p[0], p[2]), radius=cap.radius, fill=False, color=(0, 0, 0, alpha)))
    if i < len(pos) - 1:
      pn = pos[i + 1]
      ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(1, 0, 0, alpha)))

#@title A pendulum config for Brax
pendulum = brax.Config(dt=0.01, substeps=10)

ground = pendulum.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof

# start with a frozen anchor at the root of the pendulum
anchor = pendulum.bodies.add(name='anchor', mass=1.0)
anchor.frozen.all = True
cap1 = anchor.colliders.add().capsule
cap1.radius, cap1.length = 0.5, 1

ball = pendulum.bodies.add(name='middle', mass=1)
cap = ball.colliders.add().capsule
cap.radius, cap.length = 0.5, 1
# now add a middle and bottom ball to the pendulum
# pendulum.bodies.append(ball)
pendulum.bodies.append(ball)
# pendulum.bodies[1].name = 'middle'
pendulum.bodies[-1].name = 'bottom'

# connect anchor to middle
joint = pendulum.joints.add(name='joint1', parent='anchor',
                            child='middle', stiffness=10000, angular_damping=1)
joint.angle_limit.add(min = -180, max = 180)
joint.child_offset.z = 1.5
# joint.rotation.y = 45

# connect middle to bottom
pendulum.joints.append(joint)
pendulum.joints[1].name = 'joint2'
pendulum.joints[1].parent = 'middle'
pendulum.joints[1].child = 'bottom'
# pendulum.joints[1].child_offset.x = 1.0

# pendulum.joints[0].rotation.z = 40
# actuating the joint connecting the anchor and middle
angle = pendulum.actuators.add(name='actuator', joint='joint1',
                                        strength=100).angle
angle.SetInParent()  # for setting an empty oneof

# gravity is -9.8 m/s^2 in z dimension
pendulum.gravity.z = -9.8

# %%
#@title Simulating the pendulum config { run: "auto"}
ball_impulse = 8 #@param { type:"slider", min:-15, max:15, step: 0.5 }

sys = brax.System(pendulum)
qp = sys.default_qp()
qp.pos[0][2] = -0.5
# qp.pos[2][2] = 5.0

# provide an initial velocity to the ball
qp.vel[-1, 1] = ball_impulse

act = np.array([45])

batch_no = 10

qp = jax.tree_map(lambda *args: jnp.stack(args), *[qp] * batch_no)
act = jax.tree_map(lambda *args: jnp.stack(args), *[act] * batch_no)
# B = jnp.array([[]*batch_no])
qp_list = []
jit_step = jax.jit(sys.step)
for i in range(500):
  qp, _ = jax.vmap(jit_step)(qp, act)
  # qp, _ = partial(jit_step, act=[])(qp)
  # qp, _ = sys.step(qp, [])
  qp_list.append(qp)

HTML(html.render(sys, jax.tree_map(lambda x : x[0], qp_list)))
# Image(image.render(sys, qp_list, width=320, height=240))



# %%
