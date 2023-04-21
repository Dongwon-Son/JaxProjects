'''
test differentiable ensemble transform (DET) for differentiable resampling
Differentiable Particle Filtering via Entropy-Regularized Optimal Transport
'''
# %%
# import libraries
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
from ott.geometry import pointcloud, costs
from ott.solvers.linear import sinkhorn

jkey = jax.random.PRNGKey(0)

# %%
# test dataset
npts = 100
test_pnts = np.random.normal(size=(npts, 2))
# weights = np.random.uniform(size=(npts))
weights = np.ones((npts,)) * 0.01
weights[0] = 1
weights[1] = 1
weights = weights / np.sum(weights, axis=0)

@jax.tree_util.register_pytree_node_class
class Custom(costs.TICost):
    def h(self, z):
        return jnp.sum(z**2)

def det_func(pnts, weights):
    weights = weights/jnp.sum(weights, axis=-1)
    geom = pointcloud.PointCloud(pnts, pnts, cost_fn=Custom())
    ot = sinkhorn.sinkhorn(geom, a=None, b=weights)
    return npts*jnp.einsum('ji,kj->ki', pnts, ot.matrix)

def loss_test(weights, pnts):
    weights_pnt = det_func(pnts, weights)
    return jnp.mean(weights_pnt**2)

# %%
# differentiability test
# loss_jit = jax.jit(jax.value_and_grad(loss_test))
# for _ in range(1000):
#     value, grad_res = loss_jit(weights, test_pnts)
#     weights -= 0.001*grad_res
#     weights = weights.clip(1e-6)
#     weights = weights/jnp.sum(weights)
#     print(value)
    # print(weights)

# %%

weighted_pnt = det_func(test_pnts, weights)

plt.figure()
plt.scatter(test_pnts[:,0], test_pnts[:,1])
plt.scatter(weighted_pnt[:,0], weighted_pnt[:,1])
plt.scatter(test_pnts[0,0], test_pnts[0,1])
plt.scatter(test_pnts[1,0], test_pnts[1,1])
plt.show()

print(1)
