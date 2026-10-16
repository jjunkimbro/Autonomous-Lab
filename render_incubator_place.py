# 배양기 plate 배치를 보기 위한 그림. STX44 본체는 시야를 가리므로 투명 처리한다.
import numpy as np, math, mujoco, cv2
import place_plate_env_v9 as pe

HIDE = ("visual_stx44_body1",)
OLD_TARGET_X = 1.144001          # 수정 전 목표 (선반 끝을 66 mm 벗어나 있었다)


def hide(model, names):
    for n in names:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if g >= 0:
            model.geom_rgba[g, 3] = 0.0


def label(im, lines, org=(14, 30), color=(255, 255, 255), scale=0.62):
    for i, t in enumerate(lines):
        y = org[1] + int(26 * scale / 0.62) * i
        cv2.putText(im, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def shelf_extent(model, data, sz=0.690):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_stx44_plateholder")
    mid = model.geom_dataid[g]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    V = model.mesh_vert[a:a + n].reshape(-1, 3)
    W = (V @ data.geom_xmat[g].reshape(3, 3).T) + data.geom_xpos[g]
    top = W[np.abs(W[:, 2] - sz) < 0.0015]
    return top[:, 0].min(), top[:, 0].max(), top[:, 1].min(), top[:, 1].max()


def main():
    env = pe.PlacePlateEnv(randomize=False, seed=0)
    m, d = env.model, env.data
    obs, _ = env.reset(options={"station": "incubator_plate"})
    done = False
    while not done:
        obs, _, t, tr, info = env.step(np.zeros(5))
        done = t or tr
    for _ in range(600):
        mujoco.mj_step(m, d)

    x0, x1, y0, y1 = shelf_extent(m, d)
    hide(m, HIDE)
    renderer = mujoco.Renderer(m, 560, 760)
    opt = mujoco.MjvOption(); opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.690])

    good = d.qpos[env.plate_qpos:env.plate_qpos + 7].copy()
    bad = good.copy(); bad[0] = OLD_TARGET_X

    views = (("\uc704\uc5d0\uc11c", -180.0, -89.0, 0.40), ("\ube44\uc2a4\ub4ef\ud788", -150.0, -28.0, 0.46),
             ("\uc606\uc5d0\uc11c", -90.0, -4.0, 0.44))
    columns = []
    for tag, qpos, colour in (("\uc218\uc815 \ud6c4 (\ud604\uc7ac)", good, (120, 240, 140)),
                              ("\uc218\uc815 \uc804 (\ube44\uad50\uc6a9)", bad, (120, 140, 255))):
        d.qpos[env.plate_qpos:env.plate_qpos + 7] = qpos
        mujoco.mj_forward(m, d)
        tiles = []
        for name, az, el, dist in views:
            cam.lookat[:] = centre; cam.azimuth = az; cam.elevation = el; cam.distance = dist
            renderer.update_scene(d, camera=cam, scene_option=opt)
            im = renderer.render().copy()
            label(im, [f"{tag}  \u2014  {name}"], color=colour)
            tiles.append(im)
        columns.append(np.vstack(tiles))
    out = np.hstack(columns)

    px = d.qpos[env.plate_qpos]
    label(out, [
        f"\uc120\ubc18 x {x0:.4f} .. {x1:.4f}  (132.0 mm)",
        f"\ud604\uc7ac plate \uc911\uc2ec x {good[0]:.4f}   \uc55e {(x1-(good[0]+0.0635))*1000:+.1f} mm  \ub4a4 {((good[0]-0.0635)-x0)*1000:+.1f} mm",
        f"\uc218\uc815 \uc804 \uc911\uc2ec   x {OLD_TARGET_X:.4f}   plate \uc758 {((OLD_TARGET_X+0.0635)-x1)*1000:.0f} mm \uac00 \uc120\ubc18 \ubc16",
    ], org=(14, out.shape[0] - 74), scale=0.58)
    cv2.imwrite("incubator_place.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    renderer.close(); env.close()
    print("shelf x %.4f..%.4f  plate x %.4f" % (x0, x1, px))
    print("Output : incubator_place.png")


if __name__ == "__main__":
    main()
