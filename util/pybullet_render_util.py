import pybullet as p
import numpy as np


def init():
    p.connect(p.DIRECT)

def get_rgb(cam_pos = [0,0.45,0.45]):
    PIXEL_SIZE = [500,500]
    fov = 60
    near = 0.1
    far = 10.0
    pm = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=near, farVal=far)
    vm = p.computeViewMatrix(cam_pos,[0,0,0],[0,0,1])

    img_out = p.getCameraImage(*PIXEL_SIZE, viewMatrix=vm, projectionMatrix=pm, shadow=1)
    rgb = np.array(img_out[2]).reshape([*PIXEL_SIZE, 4])[...,:3]

    return rgb

def make_objs(gparam, pos, quat, scale):
    type, param = gparam[...,0].astype(np.int32), gparam[...,1:]
    p.resetSimulation()
    obj_id_list = []

    for ty, gp, ps, qt, sc in zip(type, param, pos, quat, scale):
        gp *= sc
        if ty == 0:
            obj_id_list.append(p.createMultiBody(baseMass=1,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=gp),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=gp,
                                                        rgbaColor=np.random.uniform(0,1,size=4)),
                    ))
        elif ty == 1:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=gp[0], height=gp[2]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CAPSULE, radius=gp[0], length=gp[2],rgbaColor=np.random.uniform(0,1,size=4)),
            ))
        elif ty == 2:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=gp[0], height=gp[2]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=gp[0], length=gp[2],rgbaColor=np.random.uniform(0,1,size=4)),
            ))
        elif ty == 3:
            obj_id_list.append(p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=gp[0]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=gp[0], rgbaColor=np.random.uniform(size=4)),
            ))
        else:
            raise ValueError
        
        p.resetBasePositionAndOrientation(obj_id_list[-1], ps, qt)