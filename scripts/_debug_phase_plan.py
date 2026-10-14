import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from lab_rl.envs.reader_to_waste import ReaderToWasteEnv
from lab_rl.envs.hotel_to_fluent import PLATE_BODY_FROM_TOOL

env = ReaderToWasteEnv(max_episode_steps=9000)
env.reset()
env.payload_mode = "carried"
env.data.qpos[env.rail_qadr] = 0.808
env.data.qpos[env.arm_qadr] = np.array([0.7777251264, -1.0153240998, 0.8072782794, 0.2824193925, -2.16721181524, -3.1015249812])
mujoco.mj_forward(env.model, env.data)
env.phase_index = 28
env._begin_phase()
previous = env._phase_target_arm.copy()
print(28, env.phase.name, previous)
lower=np.full(6,-2*np.pi); upper=np.full(6,2*np.pi)
desired=PLATE_BODY_FROM_TOOL.T
for label,target in (("WASTE_HIGH",np.array([1.05,-0.15,.85])),("WASTE_MID",np.array([1.05,-.30,.75])),("WASTE_PRE",np.array([1.05,-.4,.65]))):
    def residual(q):
        env._ik_data.qpos[env.rail_qadr]=.808; env._ik_data.qpos[env.arm_qadr]=q; mujoco.mj_forward(env.model,env._ik_data)
        p=env._ik_data.site_xpos[env.tcp_site_id]; r=env._ik_data.site_xmat[env.tcp_site_id].reshape(3,3)
        return np.r_[30*(p-target), Rotation.from_matrix(desired.T@r).as_rotvec(), .0001*(q-previous)]
    rng=np.random.default_rng(123)
    results=[least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=700) for seed in [previous,*[rng.uniform(lower,upper) for _ in range(100)]]]
    exact=[r for r in results if np.linalg.norm(residual(r.x))<1e-4]
    result=min(exact,key=lambda r:np.linalg.norm(r.x-previous)) if exact else min(results,key=lambda r:np.linalg.norm(residual(r.x)))
    previous=result.x
    err=residual(previous)
    print(label,np.array2string(previous,precision=10,separator=", "),"ERR",np.linalg.norm(err[:3])/30*1000,np.rad2deg(np.linalg.norm(err[3:6])))
