# %%
# import libraries
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.linen as nn
import optax

jkey = jax.random.PRNGKey(0)
BATCH_SIZE = 2048

# %%
# 
def gen_dataset(jkey, n):
    # sample1 = jax.random.normal(jkey, shape=(n//2, 2)) * jnp.array([1.0, 0.5]) + jnp.array([0.5, 0])
    # sample2 = jax.random.normal(jkey, shape=(n//2, 2)) * jnp.array([0.2, 1.0]) + jnp.array([-0.5, 0.2])
    
    sample = jax.random.normal(jkey, shape=(n, 2))
    _, jkey = jax.random.split(jkey)
    rand_idx = jax.random.randint(jkey, shape=(n,), minval=0, maxval=2)
    _, jkey = jax.random.split(jkey)

    A = jnp.array([[0.5, 0.2], [0.2, 1.0]])
    B = jnp.array([[0.8, 0], [-0.8, 0.2]])

    return sample * A[rand_idx] + B[rand_idx]


# %%
sample = gen_dataset(jkey, 1000)
# plt.figure()
# plt.scatter(sample[:,0], sample[:,1])
# plt.show()

# %%
sigma_min = 0.002
sigma_max = 80
sigma_data = 0.5
rho = 7
Pmean = -1.2
Pstd = 1.2
max_time_steps = 20

# %%
# train noise sampler
def train_noise_sampler(jkey, outer_shape):
    lnsigma = jax.random.normal(jkey, outer_shape + (1,)) * Pstd + Pmean
    return jnp.exp(lnsigma)

sigma_samples = train_noise_sampler(jkey, (10000,))[...,0]
sigma_samples = jnp.sort(sigma_samples)
# plt.figure()
# plt.hist(sigma_samples, bins=200)
# plt.show()

# %%
# sample time steps
def get_time_steps(indices, max_time_steps=max_time_steps):
    ts = jnp.power(jnp.power(sigma_max, 1/rho) + indices/(max_time_steps-1)*(jnp.power(sigma_min, 1/rho)-jnp.power(sigma_max, 1/rho)), rho)
    return jnp.where(indices <= max_time_steps, ts, 0)

sigma_res = []
for i in range(max_time_steps+1):
    sigma_res.append(get_time_steps(i))
# plt.figure()
# plt.plot(sigma_res)
# plt.show()

# %%
# forward process
def forward_process(jkey, samples, indices):
    return samples + jax.random.normal(jkey, samples.shape) * get_time_steps(indices)

noise_idx = 700
fwd_samples = forward_process(jax.random.split(jkey)[1], gen_dataset(jkey, 1000), noise_idx)
# plt.figure()
# plt.scatter(fwd_samples[:,0], fwd_samples[:,1])
# plt.show()

# %%
# design network
class Denoiser(nn.Module):

    @nn.compact
    def __call__(self, x, cond, sigma):
        if sigma.ndim == 0:
            sigma = sigma[None]
        cskip = sigma_data**2/(sigma**2+sigma_data**2)
        cout =sigma * sigma_data / jnp.sqrt(sigma_data**2 + sigma**2)
        cin = 1/jnp.sqrt(sigma**2 + sigma_data**2)
        cnoisecond = 1/4.*jnp.log(sigma)

        origin_x = x
        x = cin*x
        for _ in range(2):
            x = nn.Dense(32)(x)
            x = nn.relu(x)
        
        for _ in range(2):
            cond = nn.Dense(32)(cond)
            cond = nn.relu(cond)
        
        for _ in range(2):
            cnoisecond = nn.Dense(32)(cnoisecond)
            cnoisecond = nn.relu(cnoisecond)

        if cnoisecond.ndim != cond.ndim:
            cnoisecond = cond[...,:1] * 0 + cnoisecond
        x = jnp.concatenate([x, cond, cnoisecond], axis=-1)

        for _ in range(3):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        
        x = nn.Dense(1)(x)

        return cskip*origin_x + cout*x
        
# %%
# init model
xy = gen_dataset(jkey, 2)
sigma_train_sample = train_noise_sampler(jkey, (xy.shape[0],))
x = xy[:,0:1] + jax.random.normal(jkey, sigma_train_sample.shape) * sigma_train_sample
y = xy[:,1:2]
model = Denoiser()
params = model.init(jkey, x, y, sigma_train_sample)

model_apply_jit = jax.jit(model.apply)
# %%
# sampler
def euler_sampler(params, jkey, y, max_time_steps=max_time_steps):
    x = jax.random.normal(jkey, shape=y.shape)*get_time_steps(0, max_time_steps)
    _, jkey = jax.random.split(jkey)
    for i in range(max_time_steps+1):
        sigma = get_time_steps(i, max_time_steps)
        x0_pred = model_apply_jit(params, x, y, sigma)
        if i==max_time_steps:
            x = x0_pred
            break
        sigma_next = get_time_steps(i+1, max_time_steps)
        x = x0_pred + jax.random.normal(jkey, x0_pred.shape)*sigma_next
        _, jkey = jax.random.split(jkey)
    return x

def heun_sampler(params, jkey, y, max_time_steps=max_time_steps):
    x = jax.random.normal(jkey, shape=y.shape)*get_time_steps(0, max_time_steps)
    _, jkey = jax.random.split(jkey)
    for i in range(max_time_steps+1):
        sigma = get_time_steps(i, max_time_steps)
        sigma_next = get_time_steps(i+1, max_time_steps)
        x0_pred = model_apply_jit(params, x, y, sigma)
        d = (1/sigma)*x - 1/sigma*x0_pred
        x_next = x + (sigma_next-sigma)*d
        if i+1==max_time_steps:
            x = x_next
            break
        x0_pred_next = model_apply_jit(params, x_next, y, sigma_next)
        dprime = (1/sigma_next)*x_next - 1/sigma_next*x0_pred_next
        x = x + (sigma_next-sigma)*(dprime+d)*0.5
    return x

# %%
# define loss
def cal_loss(params, x, y, jkey):
    sigma_train_sample = train_noise_sampler(jkey, (x.shape[0],))
    _, jkey = jax.random.split(jkey)
    x_ptb = x + jax.random.normal(jkey, sigma_train_sample.shape) * sigma_train_sample
    x0_pred = model.apply(params, x_ptb, y, sigma_train_sample)

    loss_weight = (sigma_train_sample**2 + sigma_data**2)/(sigma_train_sample*sigma_data)**2
    return jnp.mean(loss_weight*(x - x0_pred)**2)

cal_loss_grad = jax.grad(cal_loss)

cal_loss_jit = jax.jit(cal_loss)

# %%
# define train func
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)
def train_func(params, opt_state, jkey):
    for _ in range(8):
        xy = gen_dataset(jkey, BATCH_SIZE)
        _, jkey = jax.random.split(jkey)
        grad = cal_loss_grad(params, xy[...,0:1], xy[...,1:2], jkey)
        _, jkey = jax.random.split(jkey)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, opt_state, jkey

train_func_jit = jax.jit(train_func)
# train_func_jit = train_func

def eval_func(params, jkey):
    y = jax.random.uniform(jkey,shape=(2000,1),minval=-2,maxval=2)
    _, jkey = jax.random.split(jkey)
    x_pred_10 = heun_sampler(params, jkey, y, 10)
    _, jkey = jax.random.split(jkey)
    x_pred_100 = heun_sampler(params, jkey, y, 100)
    _, jkey = jax.random.split(jkey)
    x_pred_500 = euler_sampler(params, jkey, y, 100)
    _, jkey = jax.random.split(jkey)
    xy_gt = gen_dataset(jkey, 1000)
    _, jkey = jax.random.split(jkey)
    plt.figure()
    plt.subplot(1,3,1)
    plt.scatter(xy_gt[:,0], xy_gt[:,1])
    plt.scatter(x_pred_10, y)
    plt.subplot(1,3,2)
    plt.scatter(xy_gt[:,0], xy_gt[:,1])
    plt.scatter(x_pred_100, y)
    plt.subplot(1,3,3)
    plt.scatter(xy_gt[:,0], xy_gt[:,1])
    plt.scatter(x_pred_500, y)
    plt.show()


# %%
for itr in range(10000):
    params, opt_state, jkey = train_func_jit(params, opt_state, jkey)
    
    xy = gen_dataset(jkey, n=1000)
    loss = cal_loss_jit(params, xy[:,0:1], xy[:,1:2], jkey)
    if itr%10 == 0:
        print(itr, loss)
    if itr%100 == 0:
        eval_func(params, jkey)
# %%
