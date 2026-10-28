# record_incubator_issue_v11.py
#
# 배양기 놓은 뒤 판정의 남은 문제를 **영상으로** 보여 준다 (시뮬레이션을 실제로 돌리며 찍는다).
#
#   판은 제자리에 놓였는데, 놓은 뒤 사진으로 다시 잰 배양기 위치가 좌우로 1~2 mm 치우쳐 '어긋남' 으로 판정한다.
#   판단이 학습 보정을 고르면 그 치우친 값을 믿고 멀쩡한 판을 옮기고, 다시 확인도 같은 치우침이라 통과한다
#   -> 조용한 실패 (judgment_chain_all.json 의 R 1 건과 같은 현상). 다시 놓기를 고르면 두 번 다 '어긋남' -> 사람 호출.
#
# 놓은 뒤 사진 위에 세 자리를 겹쳐 그린다 (판 둘레를 확대):
#   초록  사진으로 잰 판
#   빨강  놓은 뒤 다시 잰 배양기 기준으로 판이 있어야 할 자리 (판정이 비교하는 자리)
#   노랑  시뮬레이터 정답 자리 (로봇은 모른다)
#
# 실행:  py record_incubator_issue_v11.py --seed 307669177 --d6 learned
# 결과:  rl_runs/demo/incubator_issue.webm

import argparse
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

import record_flow_v11 as rf

RED, YELLOW = (255, 80, 80), (255, 220, 40)


class IssueRecorder(rf.Recorder):
    """찍은 사진을 자세와 함께 모아 두고, 놓은 뒤 판정 장면을 확대 화면으로 보여 준다."""

    def __init__(self, path):
        super().__init__(path)
        self.history = []

    def photo(self, camera, rgb):
        if camera is self.camera:
            self.history.append((rgb.copy(), camera.K.copy(), np.linalg.inv(camera.world_from_camera())))
        super().photo(camera, rgb)

    def analysis(self, photo, items, title, lines, seconds=7.0):
        """photo = (rgb, K, T). items = [(중심, yaw, frame, 색, 이름)]. 판 둘레를 잘라 크게 그린다."""
        import station_scan_v11 as ss
        rgb, K, T = photo
        polys = []
        for centre, yaw, frame, color, _ in items:
            rot = frame @ np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
            hf, hl = ss.PLATE_SIZE[0] / 2.0, ss.PLATE_SIZE[1] / 2.0
            corners = np.array([centre + rot @ np.array([sf * hf, sl * hl, 0.0])
                                for sf, sl in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            local = corners @ T[:3, :3].T + T[:3, 3]
            uv = local @ K.T
            polys.append((uv[:, :2] / uv[:, 2:3], color))
        allp = np.vstack([p for p, _ in polys])
        x0, y0 = np.floor(allp.min(0) - 40).astype(int)
        x1, y1 = np.ceil(allp.max(0) + 40).astype(int)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        box_w, box_h = rf.W - 360, rf.H - 200
        scale = min(box_w / (x1 - x0), box_h / (y1 - y0))
        widths = (2, 3, 6)          # 초록 (잰 판) 이 맨 위, 노랑 (정답) 이 굵게 맨 아래

        def draw_polys(image, origin, s, thick=1):
            for (uv, color), wd in reversed(list(zip(polys, widths))):
                pts = ((uv - origin) * s).astype(np.int32)
                cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, color, wd * thick, cv2.LINE_AA)
        crop = cv2.resize(rgb[y0:y1, x0:x1], (int((x1 - x0) * scale), int((y1 - y0) * scale)), interpolation=cv2.INTER_CUBIC)
        draw_polys(crop, np.array([x0, y0]), scale)
        # 좌우 차이가 보이도록 판의 두 긴 변(좌우 쪽 변) 가운데를 더 확대한다 (꼭짓점 순서: 변 0 = l -, 변 2 = l +)
        yellow = polys[2][0]
        mids = (yellow + np.roll(yellow, -1, axis=0)) / 2.0
        image = Image.new("RGB", (rf.W, rf.H), (18, 22, 28))
        image.paste(Image.fromarray(crop), (20, 70))
        draw = ImageDraw.Draw(image)
        half, size = 18, 230
        zoom = size / (2 * half)
        for row, (k, name) in enumerate(((0, "한쪽 긴 변"), (2, "다른 쪽 긴 변"))):
            cx, cy = mids[k]
            zx0, zy0 = int(cx - half), int(cy - half)
            patch = rgb[max(zy0, 0):zy0 + 2 * half, max(zx0, 0):zx0 + 2 * half]
            inset = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
            draw_polys(inset, np.array([zx0, zy0]), zoom)
            px, py = rf.W - size - 30, 96 + row * (size + 40)
            image.paste(Image.fromarray(inset), (px, py))
            draw.rectangle([px - 1, py - 1, px + size, py + size], outline=(200, 200, 200))
            draw.text((px, py - 26), f"{name} (확대)", font=self.small, fill=rf.WHITE)
            bx, by = 20 + (cx - x0) * scale, 70 + (cy - y0) * scale
            draw.rectangle([bx - half * scale, by - half * scale, bx + half * scale, by + half * scale], outline=(200, 200, 200))
        draw.text((20, 14), title, font=self.big, fill=rf.WHITE)
        x = 20
        for _, _, _, color, name in items:
            draw.rectangle([x, 48, x + 18, 64], fill=color)
            draw.text((x + 26, 44), name, font=self.small, fill=rf.WHITE)
            x += 46 + int(draw.textlength(name, font=self.small))
        for i, line in enumerate(lines):
            draw.text((20, rf.H - 120 + 28 * i), line, font=self.small, fill=(215, 215, 215))
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for _ in range(int(seconds * rf.FPS)):
            self.writer.write(frame)
            self.frames += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=307669177)
    ap.add_argument("--d6", default="learned", help="놓은 뒤 실패일 때 행동 (learned / again)")
    ap.add_argument("--out", default="incubator_issue.webm")
    a = ap.parse_args()
    import judgment_rl_v11 as jr
    import judgment_chain_v11 as jc
    import place_verify_v11 as pv
    import incubator_pick_v11 as ip
    rf.OUT.mkdir(parents=True, exist_ok=True)
    rec = IssueRecorder(rf.OUT / a.out)
    # 렌더러는 쓰기 전에 다 만든다 (record_flow_v11 참고)
    runner = jr.Runner()
    rec.bind(runner.prep_env, runner.prep_cam)
    rec.bind(runner.env, runner.camera)
    rf.install(rec)
    policy = jr.load_policy()
    target = "incubator_plate"
    env = runner.env
    state = dict(round=0)

    rec.title_card([("배양기 놓은 뒤 판정 — 지금 남은 문제", "big", rf.WHITE),
                    ("판은 제자리에 놓였는데, 놓은 뒤 사진으로 다시 잰 배양기 위치가 좌우로 1~2 mm 치우쳐 '어긋남' 으로 본다.", "small", rf.WHITE),
                    ("판단 RL 이 학습 보정을 고르면, 그 치우친 값을 믿고 멀쩡한 판을 옮긴다.", "small", rf.WHITE),
                    ("다시 확인할 때도 같은 치우침이라 '맞다' 로 통과 -> 조용한 실패 (1200 상황 중 1 건과 같은 현상).", "small", rf.WHITE),
                    ("초록 = 사진으로 잰 판     빨강 = 다시 잰 배양기 기준 판이 있어야 할 자리     노랑 = 정답 (로봇은 모름)", "small", rf.GREEN),
                    (f"시뮬레이션 (MuJoCo, UR5e + RG2 + 플랜지 D415) 을 실제로 돌리며 찍었다. 시드 {a.seed}.", "small", (170, 170, 170))],
                   seconds=6.0)

    # 준비 · 집기 · 운반 자막 (함수를 감싸 단계마다 자막을 바꾼다)
    prepare = runner.prepare

    def prepare_cap(seed, tgt):
        rec.bind(runner.prep_env, runner.prep_cam)
        rec.say("준비 (장비마다 한 번): 빈 배양기 받침 스캔 + 데모 판 3 장", ["CAD 없이 로봇 사진으로 장비 모델과 놓을 자리를 만든다."], "[준비 단계]", rf.GREEN, every=300)
        out = prepare(seed, tgt)
        rec.bind(env, runner.camera)
        return out
    runner.prepare = prepare_cap

    look = ip.look_and_replan

    def look_cap(e, camera, n, report=None):
        rec.say("1 · 2 단계: 출발 장비에서 판을 찍고 집는다", ["집기 전 판 1 장 -> 잰 자리로 집기 -> 손 안 1 장."], "[운영]", rf.GREEN, every=150)
        return look(e, camera, n, report)
    ip.look_and_replan = look_cap

    traverse = env.traverse

    def traverse_cap(q_goal):
        rec.say("3 단계: 배양기로 운반", ["레일로 이동 (빨리 감기)."], "", rf.WHITE, every=500)
        return traverse(q_goal)
    env.traverse = traverse_cap

    place = jc.place_by_vision

    def place_cap(*args, **kw):
        if state["round"] == 0:
            rec.say("4 · 5 단계: 손 안 1 장 + 옆에서 배양기 1 장 -> 고친 자리에 놓기", ["배양기가 준비 때보다 밀린 만큼 놓을 자리를 옮긴다."], "[운영]", rf.GREEN, every=120)
        else:
            rec.say("다시 놓기 (학습 보정): 장비는 다시 찍지 않고 보정량만큼 옮겨 놓는다" if a.d6 == "learned" else "다시 놓기: 장비를 다시 찍고 놓는다",
                    ["놓은 뒤 사진(치우친 측정) 에서 배운 보정을 지난 놓을 자리에 더한다." if a.d6 == "learned" else "같은 자세에서 배양기를 다시 잰다."],
                    "[판단 RL 행동]", rf.ORANGE, every=120)
        return place(*args, **kw)
    jc.place_by_vision = place_cap

    replace = jc.replace_here

    def replace_cap(*args, **kw):
        rec.say("판정 '어긋남' -> 판을 그 자리에서 다시 집는다", ["놓은 뒤 사진으로 잰 판 자리로 집고 손 안 1 장."], "[판단 RL 행동]", rf.ORANGE, every=120)
        return replace(*args, **kw)
    jc.replace_here = replace_cap

    look_after = pv.look_after_place

    def look_after_cap(e, camera, view=("front_up",)):
        rec.say("6 단계: 손을 빼고 놓은 뒤 2 장 (위 · 옆 -70)", ["판 위치를 재고, 같은 사진으로 배양기 위치를 다시 재서 판이 받침 기준 제자리인지 본다."], "[운영]", rf.GREEN, every=80)
        rec.history.clear()
        return look_after(e, camera, view)
    pv.look_after_place = look_after_cap

    station_target = pv.station_target
    verdict = pv.verdict

    def verdict_cap(e, centre, yaw, report, reference, tgt=None, flat=False, station=None):
        ok, why, seen = verdict(e, centre, yaw, report, reference, tgt, flat, station)
        if flat and station is not None and report.get("_shots"):
            cs, ys, _ = station_target(e, station, report["_shots"], plate=(centre, yaw))
            frame = np.asarray(e.station["rotation"])
            pivot = np.asarray(e.perturber.pivot(e.station_name))
            truth_c = pivot + np.asarray(e.truth[:3])
            truth_y = float(e.truth[3])
            et = frame.T @ (centre - truth_c)
            es = frame.T @ (cs - truth_c)
            state["round"] += 1
            first = state["round"] == 1
            judged = "어긋남 -> 실패" if not ok else "제자리 -> 통과"
            really = "실제로는 제자리" if abs(et[0]) < 1e-3 and abs(et[1]) < 1e-3 else "실제로는 어긋남"
            lines = [f"판정: {judged}  ({why or '1 mm · 0.5° 안'})   |   {really}: 판 - 정답 = 앞뒤 {et[0] * 1000:+.2f} · 좌우 {et[1] * 1000:+.2f} mm",
                     f"다시 잰 배양기 기준 자리 - 정답 = 앞뒤 {es[0] * 1000:+.2f} · 좌우 {es[1] * 1000:+.2f} mm  ← 측정 자체가 치우쳤다",
                     ("판은 멀쩡한데 치우친 기준과 비교해 실패로 본다." if first else
                      "같은 치우침이 이번에는 옮겨진 판과 맞아 '통과' — 로봇은 실패를 모른다 (조용한 실패)." if ok else
                      "다시 놓아도 같은 치우침으로 또 실패 -> 사람 호출.")]
            photo = rec.history[-2] if len(rec.history) >= 2 else rec.history[-1]
            rec.analysis(photo, [(centre, yaw, frame, rf.GREEN, "사진으로 잰 판"),
                                 (cs, ys, frame, RED, "다시 잰 배양기 기준 자리 (판정 기준)"),
                                 (truth_c, truth_y, frame, YELLOW, "정답 (로봇은 모름)")],
                         f"놓은 뒤 판정 {state['round']} 번째 — 위에서 찍은 사진, 판 둘레 확대", lines, seconds=9.0)
        return ok, why, seen
    pv.verdict = verdict_cap

    def act6(feats):
        return a.d6
    row = runner.episode(a.seed, target, policy.act5, act6, correct=policy.correct)
    rec.say("결과", [f"단계 {row['stage']}  ·  성공 {row['done_ok']}  ·  로봇 판정 {(row.get('verify2') or row.get('verify1'))['ok']}"])
    rec.hold(2.0)
    rec.title_card([("정리", "big", rf.WHITE),
                    ("문제의 뿌리: 평평한 받침은 판과 크기가 같아, 판이 제자리면 받침 윗면 테두리가 판 테두리 밑에 숨는다.", "small", rf.WHITE),
                    ("그래서 놓은 뒤 사진에서 배양기 위치를 멀리 있는 몸체 위주로 재고, 드물게 좌우 1~2 mm 치우친다.", "small", rf.WHITE),
                    ("오거부 약 1 % (다시 놓기 -> 사람 호출), 학습 보정을 고르면 조용한 실패.", "small", rf.ORANGE),
                    ("다음: 판 옆면과 받침 옆면의 단차로 판-받침 상대 위치를 직접 재기 / 두 측정이 다르면 믿지 않는 안전장치.", "small", rf.GREEN)],
                   seconds=7.0)
    rec.close()
    print("stage", row["stage"], "done", row["done_ok"], "frames", rec.frames, f"({rec.frames / rf.FPS:.0f} s)")
    print("Output :", rf.OUT / a.out)


if __name__ == "__main__":
    main()
