import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lab_rl.envs.hotel_to_fluent import HotelToFluentEnv, PLATE_BODY_FROM_TOOL

env = HotelToFluentEnv()
d = env._ik_data
rail = 0.808
target = np.array([rail - 0.25, -0.02, 0.75])
lower = np.full(6, -2*np.pi)
upper = np.full(6, 2*np.pi)
for i, (jid, aid) in enumerate(zip(env.arm_joint_ids, env.arm_actuator_ids, strict=True)):
    if env.model.jnt_limited[jid]: lower[i], upper[i] = env.model.jnt_range[jid]
    if env.model.actuator_ctrllimited[aid]:
        a,b=env.model.actuator_ctrlrange[aid]; lower[i],upper[i]=max(lower[i],a),min(upper[i],b)

def solve(yaw, seeds):
    desired = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ PLATE_BODY_FROM_TOOL.T
    def res(q):
        d.qpos[env.rail_qadr]=rail; d.qpos[env.arm_qadr]=q; mujoco.mj_forward(env.model,d)
        p=d.site_xpos[env.tcp_site_id]; r=d.site_xmat[env.tcp_site_id].reshape(3,3)
        return np.r_[30*(p-target), Rotation.from_matrix(desired.T@r).as_rotvec()]
    results=[least_squares(res,np.clip(s,lower,upper),bounds=(lower,upper),max_nfev=600) for s in seeds]
    return min(results,key=lambda x:np.linalg.norm(res(x.x))),res

rng=np.random.default_rng(92)
start_reader = np.array([2.1860357018, -1.8847690629, 2.2721452066, 2.7491174014, -2.1863255158, -6.2697602143])
seed_batch = [rng.uniform(lower,upper) for _ in range(400)]
desired180 = Rotation.from_euler('z', 180, degrees=True).as_matrix() @ PLATE_BODY_FROM_TOOL.T
def res180(q):
    d.qpos[env.rail_qadr]=rail; d.qpos[env.arm_qadr]=q; mujoco.mj_forward(env.model,d)
    p=d.site_xpos[env.tcp_site_id]; r=d.site_xmat[env.tcp_site_id].reshape(3,3)
    return np.r_[30*(p-target), Rotation.from_matrix(desired180.T@r).as_rotvec()]
all180=[least_squares(res180,np.clip(s,lower,upper),bounds=(lower,upper),max_nfev=500) for s in seed_batch]
exact180=[x for x in all180 if np.linalg.norm(res180(x.x)) < 1e-4]
for item in sorted(exact180,key=lambda x:np.linalg.norm(x.x-start_reader))[:8]:
    print('YAW180_CAND',np.linalg.norm(item.x-start_reader),np.array2string(item.x,precision=8))
result=min(exact180,key=lambda x:np.linalg.norm(x.x-start_reader))
res=res180
previous=result.x
print('YAW',180,'ERR',np.linalg.norm(res(previous)[:3])/30*1000,np.rad2deg(np.linalg.norm(res(previous)[3:])),np.array2string(previous,precision=8))
for yaw in range(190,361,10):
    seeds=[previous]
    for _ in range(12): seeds.append(np.clip(previous+rng.normal(0,.25,6),lower,upper))
    result,res=solve(yaw,seeds)
    delta=np.linalg.norm(result.x-previous)
    previous=result.x
    print('YAW',yaw,'ERR',np.linalg.norm(res(previous)[:3])/30*1000,np.rad2deg(np.linalg.norm(res(previous)[3:])),'DELTA',delta,np.array2string(previous,precision=8))
