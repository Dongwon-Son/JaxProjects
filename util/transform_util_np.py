# try:
#     import quat_util as qutil
# except:
#     import util.quat_util as qutil

import numpy as np

# quaternion operations

def quw2wu(quw):
    return np.concatenate([quw[...,-1:], quw[...,:3]], axis=-1)

def qrand(size):
    q1 = np.random.normal(size=size)
    q1 = q1 / np.linalg.norm(q1)
    return q1

def qmulti(q1, q2):
    b,c,d,a = np.split(q1, 4, axis=-1)
    f,g,h,e = np.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return np.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = np.split(q, 4, axis=-1)
    return np.concatenate([-x,-y,-z,w], axis=-1)

def sign(q):
    return (q > 0).astype(q.dtype)*2-1

def qlog(q):
    alpha = np.arccos(q[...,3:])
    sinalpha = np.sin(alpha)
    abssinalpha = np.maximum(np.abs(sinalpha), 1e-6)
    n = q[...,:3]/(abssinalpha*sign(sinalpha))
    return n*alpha

def q2aa(q):
    return 2*qlog(q)

def aa2q(aa):
    return qexp(aa*0.5)

def qexp(logq):
    alpha = np.linalg.norm(logq, axis=-1, keepdims=True)
    alpha = np.maximum(alpha, 1e-6)
    return np.concatenate([logq/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)

def qaction(quat, pos):
    return qmulti(qmulti(quat, np.concatenate([pos, np.zeros_like(pos[...,:1])], axis=-1)), qinv(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

# posquat operations
def pq_inv(pos, quat):
    quat_inv = qinv(quat)
    return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt):
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2, quat2):
    return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)