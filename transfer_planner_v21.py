# transfer_planner_v21.py
#
# 시퀀스 실행 중에 이송 하나씩 계획한다 (process_rule_v21.Planner 의 사슬 방식 그대로, 13 단계 전체 동적계획 대신).
#   - 출발 = 지금 관절값 (앞 이송이 끝난 자세, 다시 집기 · 다시 놓기 중이면 그 자세), 물체 위치 = 지금 상태.
#   - 걸음 = [(물체, 자리, 동작)] — 동작 pick / place / drop / look. 보통 이송은 [pick 출발, place 도착],
#     다시 집기는 [place 출발, pick 출발, place 도착], 다시 놓기는 [pick 도착, place 도착].
#   - look = 도착 자리 미리 보기 (사진0): 손에 든 물체가 사진4 에서 도착 자리를 가리면, 집기 전에 빈손으로
#     도착 자리 사진 자세에 먼저 가서 찍는다 (11-12 사용자 방향). 가림 판정은 camera_view_v21 의 광선 추적.
#   - 레일과 팔은 따로 (레일만 -> 팔만, 안 되면 레일 고정 RRT). J6 고정 c, 손가락 방향 부호 xs 는 시작 때 한 번 정한다.

import math
import time

import mujoco
import numpy as np

import camera_view_v21 as CV
import process_rule_v21 as P


class OnlinePlanner(P.Planner):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.c, self.xs = None, None
        self.cache, self.look_cache, self.chain_src = {}, {}, []
        self.photo_frames = P.PHOTO_FRAMES          # 사진 한 장 정지 (프레임) — 순서표 [policy] photo_s 로 바꾼다
        self.photos = {1, 2, 4, 5, 6}                # 찍는 사진 (순서표 [policy] photos), 0 은 plan_transfer 의 prelook 이 정함
        self.photo6_seats = None                     # 사진6 을 찍을 도착 자리 (None = 모든 놓기, 순서표 photo6_at = "device" 면 장비 자리만)
        # 가림 판정용 별도 장면 (카메라 몸체 · 받침은 광선에서 뺀다)
        self.V = P.Scene()
        for n in ("wrist_camera_body_visual", "wrist_camera_bracket"):
            self.V.m.geom_group[self.V.m.geom(n).id] = 3
        self.cam = self.V.m.camera("wrist_camera").id

    # --- 손목 고정값 ---------------------------------------------------
    def choose_wrist(self, transfers, objs):
        """process_rule_v21.Planner.plan 과 같이 카메라가 위로 오는 (J6 c, 롤 xs) 부터, 순서표의 모든 자리에서
        집기 · 놓기 사슬이 있는 첫 쌍 (원본 계획기도 c = π 가 판 보관대 4 에서 막혀 c = 0 으로 갔다)."""
        first = transfers[0][1]
        g0, a0 = self.pose_of(first)
        pairs = []
        for c in (np.pi, 0.0, np.pi / 2, -np.pi / 2):
            for xs in (1, -1):
                qs = self.ik_all(g0 - a0 * 0.2, a0, xs * np.cross([0, 0, 1.0], a0), c, 12)
                pairs.append((round(float(self.camera_up(qs[0], c)), 2) if qs else -1.0, c, xs))
        checks = []
        for obj, src, dst in transfers:
            checks += [(obj, src, "pick"), (obj, dst, "drop" if dst.startswith("waste") else "place")]
        checks = list(dict.fromkeys(checks))
        for up, c, xs in sorted(pairs, key=lambda p: -p[0]):
            if up <= P.CAMERA_MOUNT_MIN_UP:
                break
            self.c, self.xs = c, xs
            bad = None
            cache = {}
            for obj, tgt, verb in checks:
                holding = obj if verb != "pick" else None
                # 다른 물체 없이 (장비 · 테이블만): 그 걸음 때 옆 자리 물체는 순서마다 달라 손목 고르기에는 넣지 않는다
                # (넘기지 않은 물체는 키프레임 처음 자리에 남으므로 바닥 아래 멀리 치운다)
                far = {o: (np.array([3.0 + k, 0.0, -5.0]), np.array([1.0, 0, 0, 0])) for k, o in enumerate(P.KIND) if o != obj}
                if verb == "pick":
                    far[obj] = self.held_pose(P.KIND[obj], *self.pose_of(tgt))
                lv = self.levels_for([(obj, tgt, verb)], far, holding, "")
                for L in lv:
                    L["xs"] = xs * np.cross([0, 0, 1.0], L["a"])
                chains, _ = self.build_chains(lv)
                if not chains:
                    bad = f"{obj} {tgt} {verb}"
                    break
                cache[(P.KIND[obj], tgt, verb)] = chains
            print(f"  손목 J6 {c:+.2f} 롤 {xs:+d} (카메라 위쪽 {up:+.2f}): {'모든 자리 사슬 있음' if bad is None else '사슬 없음 ' + bad}",
                  flush=True)
            if bad is None:
                # 장비가 고정이라 자리별 사슬 (IK) 은 늘 같다 -> 저장해 두고 실행 중에는 그때 물체와의 충돌만 본다
                self.cache = cache
                return c, xs, up
        raise RuntimeError("모든 자리에서 사슬이 있는 손목 자세 없음")

    NEAR = 0.30          # 사슬 손끝 점에서 이보다 가까운 다른 물체가 있을 때만 물체 충돌 다시 검사

    def chains_for(self, Ls, key):
        """저장한 사슬 (장비 · 테이블만 본 것) 을 지금 물체로 다시 확인. 가까운 물체가 없으면 그대로, 있으면 충돌 검사.
        저장한 것이 없거나 다 막히면 새로 만든다. 반환 (사슬, G 해 수, 출처)."""
        cached = getattr(self, "cache", {}).get(key)
        if cached:
            L0 = Ls[0]
            others = [p for o, (p, _) in L0["objs"].items() if o != L0["obj"] and o != L0.get("held")]
            pts = np.array([L["p"] for L in Ls])
            near = any(np.min(np.linalg.norm(pts - p, axis=1)) < self.NEAR for p in others)
            if not near:
                return cached, len(cached), "저장"
            ok = [ch for ch in cached if self.chain_ok(Ls, ch["q"])]
            if ok:
                return ok, len(cached), "저장+검사"
        chains, ng = self.build_chains(Ls)
        for n in (64, 64):              # IK 는 무작위 시작점이라 해를 적게 찾는 때가 있다 -> 사람 부르기 전에 시작점을 늘려 두 번 더
            if chains:
                break
            chains, ng2 = self.build_chains(Ls, n)
            ng = max(ng, ng2)
        return chains, ng, "새로"

    def chain_ok(self, Ls, ch):
        """사슬 관절값 ch (진입 순서) 가 지금 물체 상태에서 막히지 않는지 (build_chains 와 같은 검사, IK 없이)."""
        fwd = [L for L in Ls if L["tag"] == "진입"]
        back = [L for L in Ls if L["tag"] == "빼기"]
        for k, (L, q) in enumerate(zip(fwd, ch)):
            if not self.okf(L, q):
                return False
            if k and not self.edge_ok(fwd[k - 1], ch[k - 1], q, max(fwd[k - 1]["station"], L["station"])):
                return False
        prevq, prevL = ch[-1], fwd[-1]
        for j, Lb in enumerate(back):
            q = ch[len(fwd) - 2 - j]
            if not self.okf(Lb, q) or not self.edge_ok(Lb, prevq, q, max(Lb["station"], prevL["station"])):
                return False
            prevq, prevL = q, Lb
        return True

    def build_chains(self, Ls, n=32):
        """한 걸음 (집기 / 놓기) 의 사슬: 잡는 점 G 의 IK 해마다 C 까지 거꾸로, 빼기는 같은 해로 되짚는다."""
        c = self.c
        fwd = [L for L in Ls if L["tag"] == "진입"]
        back = [L for L in Ls if L["tag"] == "빼기"]
        G = fwd[-1]
        gc = [q for q in self.ik_all(G["p"], G["a"], G["xs"], c, n) if self.okf(G, q)]
        chains = []
        for qg in gc:
            ch = [qg]
            for k in range(len(fwd) - 2, -1, -1):
                Lk = fwd[k]
                qn = self.ik_near(ch[-1], Lk["p"], Lk["a"], Lk["xs"], c)
                if qn is None or not self.okf(Lk, qn) or \
                        not self.edge_ok(Lk, qn, ch[-1], max(Lk["station"], fwd[k + 1]["station"])):
                    ch = None
                    break
                ch.append(qn)
            if ch is None:
                continue
            ch = ch[::-1]
            good, prevq, prevL = True, ch[-1], G
            for j, Lb in enumerate(back):
                q = ch[len(fwd) - 2 - j]
                if not self.okf(Lb, q) or not self.edge_ok(Lb, prevq, q, max(Lb["station"], prevL["station"])):
                    good = False
                    break
                prevq, prevL = q, Lb
            if good and all(np.abs((ch[0] - o["q"][0]) * P.WJ).max() > 0.03 for o in chains):
                chains.append(dict(q=ch, cost=2 * self.jcost(ch)))
        return chains, len(gc)

    def home(self, first_tgt, objs):
        """시작 자세 = 첫 자리 사진 자세 (운반 높이 C) 의 IK 해 중 하나."""
        kind = "plate" if "plate" in first_tgt else "tube"
        pts, a, _ = self.station_pts(kind, first_tgt)
        L = dict(w=P.W_OPEN[kind], objs=objs, held=None, kind=kind, station=0, obj=None)
        for q in self.ik_all(pts[0], a, self.xs * np.cross([0, 0, 1.0], a), self.c, 24):
            if self.okf(L, q):
                return q
        raise RuntimeError("시작 자세 없음")

    # --- 충돌 도우미 (Planner.build 안 함수들을 메서드로) -------------------
    def okf(self, L, q):
        return self.ok_at(np.r_[q, self.c], L["w"], L["objs"], L["held"], L["kind"], L["station"], L["obj"])

    def edge_ok(self, L, qa, qb, st):
        return all(self.ok_at(np.r_[q, self.c], L["w"], L["objs"], L["held"], L["kind"], st, L["obj"])
                   for q in self.interp(qa, qb)[:-1])

    @staticmethod
    def jcost(qs):
        return sum(np.abs((b - a) * P.WC).sum() for a, b in zip(qs[:-1], qs[1:]))

    def seq_path(self, L0, qa, qb, rrt=False):
        qa_r = qa.copy()
        qa_r[0] = qb[0]
        qb_r = qb.copy()
        qb_r[0] = qa[0]
        if abs(qa[0] - qb[0]) < 1e-6:
            if self.edge_ok(L0, qa, qb, 0):
                return [qa, qb]
        else:
            if self.edge_ok(L0, qa, qa_r, 0) and self.edge_ok(L0, qa_r, qb, 0):
                return [qa, qa_r, qb]
            if self.edge_ok(L0, qa, qb_r, 0) and self.edge_ok(L0, qb_r, qb, 0):
                return [qa, qb_r, qb]
        if not rrt:
            return None
        moved = abs(qa[0] - qb[0]) > 1e-6
        for first, last in ((qa_r, None), (None, qb_r)):
            if first is not None and moved and not self.edge_ok(L0, qa, qa_r, 0):
                continue
            if last is not None and not (moved and self.edge_ok(L0, qb_r, qb, 0)):
                continue
            s0, s1 = (qa_r, qb) if first is not None else (qa, qb_r)
            mid = self.rrt6(s0, s1, lambda q: self.okf(L0, q), lambda x, y: self.edge_ok(L0, x, y, 0), fix_rail=s0[0])
            if mid is None:
                continue
            return ([qa] if first is not None and moved else []) + mid + ([qb] if last is not None else [])
        return None

    # --- 경유점 열 -------------------------------------------------------
    def levels_for(self, steps, objs, holding, label):
        """steps = [(물체, 자리, 동작)], objs = 지금 물체 자세, holding = 지금 든 물체 (없으면 None)."""
        objs = dict(objs)
        levels = []
        for si, (obj, tgt, verb) in enumerate(steps):
            kind = P.KIND[obj]
            pts, a, sts = self.station_pts(kind, tgt)
            w_in = P.W_HOLD[P.KIND[holding]] if holding else P.W_OPEN[kind]
            base = dict(a=a, kind=kind, label=label, transfer=0, verb=verb, tgt=tgt, obj=obj, step=si)
            if verb == "look":
                levels.append(dict(base, wait=True, p=pts[0], w=w_in, objs=dict(objs), held=holding, station=0,
                                   tag="사진", section="사진0"))
                continue
            sec_in, sec_grip, sec_out = {"pick": ("집기", "쥐기", "빼기"), "place": ("놓기", "펴기", "물러나기"),
                                         "drop": ("버리기", "펴기", "물러나기")}[verb]
            pre = [x for x in {"pick": ["사진1"], "place": ["사진4", "사진5"], "drop": []}[verb] if int(x[2:]) in self.photos]
            post = [x for x in {"pick": ["사진2"], "place": ["사진6"], "drop": []}[verb] if int(x[2:]) in self.photos
                    and not (x == "사진6" and self.photo6_seats is not None          # 장비 자리만: 이송의 마지막 놓기 (다시 집기 때 출발 자리에 내려놓기는 빼고)
                             and (tgt not in self.photo6_seats or si != len(steps) - 1))]
            for name in pre:                # 사진5 = 사진4 자리에서 옆으로 (v11 'aside+70', side_q)
                levels.append(dict(base, wait=True, p=pts[0], w=w_in, objs=dict(objs), held=holding, station=0,
                                   tag="사진", section=name, side=name == "사진5"))
            for k, p in enumerate(pts):
                levels.append(dict(base, p=p, w=w_in, objs=dict(objs), held=holding, station=sts[k], tag="진입",
                                   section=sec_in))
            g, aa = self.pose_of(tgt)
            if verb == "pick":
                holding = obj
                objs.pop(obj, None)
            elif verb == "drop":
                bottom = (self.waste_lo + self.waste_hi) / 2
                bottom[2] = self.waste_lo[2] + 0.005 + (0.007 if kind == "plate" else P.TUBE_HALF)
                objs[obj] = (bottom, P.quat_z(math.atan2(aa[1], aa[0]) - math.pi / 2))
                holding = None
            else:
                objs[obj] = self.held_pose(kind, g, aa)
                holding = None
            w_out = P.W_HOLD[kind] if holding else P.W_OPEN[kind]
            levels.append(dict(base, grip=(w_in, w_out, verb), p=pts[-1], w=w_out, objs=dict(objs), held=holding,
                               station=2, tag="잡기", section=sec_grip))
            for k, p in list(enumerate(pts))[::-1][1:]:
                levels.append(dict(base, p=p, w=w_out, objs=dict(objs), held=holding, station=sts[k], tag="빼기",
                                   section=sec_out))
            for name in post:
                levels.append(dict(base, wait=True, p=pts[0], w=w_out, objs=dict(objs), held=holding, station=0,
                                   tag="사진", section=name))
        # 든 물체는 objs 에 넣어 둔다 (state_objs 가 손 위치로 덮어쓴다)
        for L in levels:
            if L["held"] and L["held"] not in L["objs"]:
                L["objs"][L["held"]] = (np.zeros(3), np.array([1.0, 0, 0, 0]))
        return levels

    # --- 사슬 고르기 -------------------------------------------------------
    MAX_COMBOS = 400

    def best_first(self, q0, blocks):
        """사슬 조합을 하한 비용 (사슬 비용 + 자리 사이 관절 거리) 이 작은 순서로 보고, 자리 사이가 '레일만 -> 팔만' 으로
        모두 뚫린 첫 조합을 고른다. 이 경로의 비용은 관절 거리와 같아 첫 조합이 곧 최소 (동적계획과 같은 답, 검사는 훨씬 적게).
        반환 (고른 사슬 번호들, 자리 사이 경유점, 비용) / None (MAX_COMBOS 안에 없으면 -> dp_rrt)."""
        import itertools
        starts = [[np.asarray(ch["q"][0]) for ch in B["chains"]] for B in blocks]
        costs = [[ch["cost"] for ch in B["chains"]] for B in blocks]
        combos = []
        for sel in itertools.product(*[range(len(B["chains"])) for B in blocks]):
            lb = np.abs((starts[0][sel[0]] - q0) * P.WC).sum() + costs[0][sel[0]]
            for si in range(1, len(blocks)):
                lb += np.abs((starts[si][sel[si]] - starts[si - 1][sel[si - 1]]) * P.WC).sum() + costs[si][sel[si]]
            combos.append((lb, sel))
        combos.sort(key=lambda x: x[0])
        memo = {}

        def path(si, ia, ib):
            key = (si, ia, ib)
            if key not in memo:
                qa = q0 if si == 0 else starts[si - 1][ia]
                memo[key] = self.seq_path(blocks[si]["fwd"][0], qa, starts[si][ib], False)
            return memo[key]
        for lb, sel in combos[:self.MAX_COMBOS]:
            paths = []
            for si in range(len(blocks)):
                pth = path(si, sel[si - 1] if si else -1, sel[si])
                if pth is None:
                    break
                paths.append(pth[1:-1])
            else:
                return list(sel), {si: p for si, p in enumerate(paths)}, lb
        return None

    def dp_rrt(self, q0, blocks, steps):
        """원래 동적계획 (레일만 -> 팔만, 안 되면 레일 고정 RRT). best_first 가 못 찾을 때만."""
        V = [None] * len(blocks[0]["chains"])
        for rr in (False, True):
            for ib, ch in enumerate(blocks[0]["chains"]):
                if V[ib] is not None:
                    continue
                pth = self.seq_path(blocks[0]["fwd"][0], q0, ch["q"][0], rr)
                if pth is not None:
                    V[ib] = (self.jcost(pth) + ch["cost"], None, pth[1:-1])
            if any(v is not None for v in V):
                break
        if all(v is None for v in V):
            return "출발 자세에서 첫 자리로 가는 경로 없음"
        blocks[0]["V"] = V
        for si in range(1, len(blocks)):
            Pb, B = blocks[si - 1], blocks[si]
            nv = [None] * len(B["chains"])
            L0 = B["fwd"][0]
            for rr in (False, True):
                for ia, cha in enumerate(Pb["chains"]):
                    if V[ia] is None:
                        continue
                    for ib, chb in enumerate(B["chains"]):
                        cst0 = V[ia][0] + np.abs((chb["q"][0] - cha["q"][0]) * P.WC).sum() + chb["cost"]
                        if nv[ib] is not None and cst0 >= nv[ib][0]:
                            continue
                        pth = self.seq_path(L0, cha["q"][0], chb["q"][0], rr)
                        if pth is None:
                            continue
                        cst = V[ia][0] + self.jcost(pth) + chb["cost"]
                        if nv[ib] is None or cst < nv[ib][0]:
                            nv[ib] = (cst, ia, pth[1:-1])
                    if rr and any(v is not None for v in nv):
                        break
                if any(v is not None for v in nv):
                    break
            if all(v is None for v in nv):
                return f"자리 사이 경로 없음: {steps[si]}"
            B["V"] = nv
            V = nv
        ib = min((i for i, v in enumerate(V) if v is not None), key=lambda i: V[i][0])
        total = V[ib][0]
        sel, transits = [0] * len(blocks), {}
        for si in range(len(blocks) - 1, -1, -1):
            sel[si] = ib
            _, ia, mid = blocks[si]["V"][ib]
            transits[si] = mid
            ib = ia if ia is not None else ib
        return sel, transits, total

    # --- 계획 -----------------------------------------------------------
    def plan_steps(self, q0, steps, objs, holding, label):
        """q0 (6) 에서 출발하는 경유점 열 (관절값 채움). 실패하면 (None, 이유)."""
        c, xsign = self.c, self.xs
        levels = self.levels_for(steps, objs, holding, label)
        for L in levels:
            L["xs"] = xsign * np.cross([0, 0, 1.0], L["a"])
        blocks = []
        for si in range(len(steps)):
            Ls = [L for L in levels if L["step"] == si]
            if steps[si][2] == "look":
                L = Ls[0]
                key = ("look", L["kind"], steps[si][1])
                if key not in self.look_cache:                 # 미리 보기 자세 IK 도 자리마다 한 번만
                    self.look_cache[key] = self.ik_all(L["p"], L["a"], L["xs"], c, 24)
                qs = [q for q in self.look_cache[key] if self.okf(L, q)]
                if not qs:
                    return None, f"미리 보기 자세 없음: {steps[si][1]}"
                blocks.append(dict(look=L, first=L, fwd=[L], chains=[dict(q=[q], cost=0.0) for q in qs]))
                continue
            B = dict(fwd=[L for L in Ls if L["tag"] == "진입"], grip=next(L for L in Ls if L["tag"] == "잡기"),
                     back=[L for L in Ls if L["tag"] == "빼기"], waits=[L for L in Ls if L["tag"] == "사진"], first=Ls[0])
            chains, ng, src = self.chains_for(Ls, (Ls[0]["kind"], steps[si][1], steps[si][2]))
            self.chain_src.append(src)
            if not chains:
                return None, f"사슬 없음: {steps[si][0]} {steps[si][1]} {steps[si][2]} (G 해 {ng})"
            B["chains"] = chains
            blocks.append(B)
        found = self.best_first(q0, blocks)
        if found is None:
            found = self.dp_rrt(q0, blocks, steps)
            if isinstance(found, str):
                return None, found
        sel, transits, total = found
        for si, B in enumerate(blocks):
            ch = B["chains"][sel[si]]["q"]
            if "look" in B:
                B["look"]["q"] = ch[0]
            else:
                for L, q in zip(B["fwd"], ch):
                    L["q"] = q
                for j, L in enumerate(B["back"]):
                    L["q"] = ch[len(B["fwd"]) - 2 - j]
                B["grip"]["q"] = ch[-1]
                for L in B["waits"]:
                    L["q"] = self.side_q(L, ch[0]) if L.get("side") else ch[0]
        out = []
        for si, B in enumerate(blocks):
            L0 = B["first"]
            for q in transits.get(si, []):
                out.append(dict(L0, q=q, wait=False, tag="이동", section="운반" if L0["held"] else "빈손 이동",
                                station=0))
            out += [L for L in levels if L["step"] == si]
        return out, total

    SIDE = 0.070          # 사진5: 사진4 자세에서 옆으로 (v11 station_scan DEMO_VIEWS 'aside+70')

    def side_q(self, L, q0):
        """사진5 자세: 사진4 자세 q0 에서 자리 옆쪽 (위에서 봐 진입 방향의 왼쪽 = +) 으로 SIDE. 팔만 -> 레일 포함 순서로 IK,
        충돌 · 가는 길 검사. + 가 막히면 -, 둘 다 막히면 사진4 자리 그대로 (L['side_mm'] = 0)."""
        lat = np.cross([0, 0, 1.0], L["a"])
        for sgn in (1, -1):
            p = L["p"] + lat * self.SIDE * sgn
            for fix in (True, False):
                q = self.ik_near(q0, p, L["a"], L["xs"], self.c, fix_rail=fix)
                if q is not None and self.okf(L, q) and self.edge_ok(L, q0, q, 0):
                    L["side_mm"] = round(self.SIDE * 1000) * sgn
                    return q
        L["side_mm"] = 0
        return q0

    # --- 프레임 (24 fps) -------------------------------------------------
    def frames_from(self, levels, q0):
        """Planner.frames 와 같은 보간, 출발은 q0, 끝 정지 없음. [(q7, 반폭, 그리퍼 명령, 걸음, 구간)]"""
        c = self.c
        out = []

        def smooth(qa, qb, step, rstep):
            dr, da = abs(qb[0] - qa[0]), np.abs(qb - qa)[1:].max()
            if dr > 1e-6:
                ta, L = P.RAIL_V / P.RAIL_A, dr
                if L < P.RAIL_V * ta:
                    ta = math.sqrt(L / P.RAIL_A)
                    T, vm = 2 * ta, P.RAIL_A * ta
                else:
                    vm, T = P.RAIL_V, 2 * ta + (L - P.RAIL_V * ta) / P.RAIL_V

                def sdist(t):
                    if t < ta:
                        return 0.5 * P.RAIL_A * t * t
                    if t > T - ta:
                        return L - 0.5 * P.RAIL_A * (T - t) ** 2
                    return 0.5 * P.RAIL_A * ta * ta + vm * (t - ta)
                n = max(2, int(np.ceil(T * P.FPS)))
                move = [qa + (qb - qa) * (sdist(t) / L) for t in np.linspace(0, T, n + 1)[1:]]
                return [qa] * P.RAIL_SETTLE_FRAMES + move + [qb] * P.RAIL_SETTLE_FRAMES
            n = int(np.ceil(max(da / step, dr / rstep))) + 1
            return [qa + (qb - qa) * (0.5 - 0.5 * math.cos(math.pi * t)) for t in np.linspace(0, 1, n)[1:]]

        def cmd_of(kind, w, held):
            return P.W_HOLD[kind] - P.SQUEEZE[kind] if held else w

        prev = np.asarray(q0, float)
        for L in levels:
            kind = L["kind"]
            if L.get("grip"):
                w0, w1, verb = L["grip"]
                q7 = np.r_[L["q"], c]
                closing = verb == "pick"
                for t in np.linspace(0, 1, P.GRIP_FRAMES):
                    w = w0 + (w1 - w0) * t
                    if closing:
                        cmd = w0 + (P.W_HOLD[kind] - P.SQUEEZE[kind] - w0) * t
                    else:
                        cmd = P.W_HOLD[kind] - P.SQUEEZE[kind] + (w1 - (P.W_HOLD[kind] - P.SQUEEZE[kind])) * t
                    out.append((q7, w, cmd, L["step"], L["section"]))
                out += [out[-1]] * (P.GRIP_HOLD if closing else P.RELEASE_HOLD)
                continue
            q = L["q"]
            if L.get("wait") and np.abs(q - prev).max() < 1e-9:
                seg = []
            else:
                seg = smooth(prev, q, 0.02 if L["station"] else 0.04, 0.006 if L["station"] else 0.012)
            held_kind = P.KIND[L["held"]] if L["held"] else kind
            if L.get("wait"):
                moving = "운반" if L["held"] else "빈손 이동"
                for k, qq in enumerate(seg):
                    out.append((np.r_[qq, c], L["w"], cmd_of(held_kind, L["w"], L["held"]), L["step"], moving))
                for _ in range(self.photo_frames):
                    out.append((np.r_[q, c], L["w"], cmd_of(held_kind, L["w"], L["held"]), L["step"], L["section"]))
            else:
                for qq in seg:
                    out.append((np.r_[qq, c], L["w"], cmd_of(held_kind, L["w"], L["held"]), L["step"], L["section"]))
            prev = q
        return out

    # --- 가림 판정 -------------------------------------------------------
    def seat_visible(self, q7, w, objs, held, kind, tgt):
        """사진 자세 q7 에서 도착 자리 (판 받침 / 튜브 구멍 입구) 가 보이는 비율 (든 물체 포함)."""
        V = self.V
        m, d = V.m, V.d
        d.qpos[:] = V.key_qpos
        d.qpos[V.qadr] = q7
        d.qpos[V.fing] = w
        objs = dict(objs)
        if held:
            p = None
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[V.tcp].copy()
            z = d.site_xmat[V.tcp].reshape(3, 3)[:, 2]
            a = np.array([z[0], z[1], 0.0]) / math.hypot(z[0], z[1])
            objs[held] = V.held_pose(P.KIND[held], p, a)
        for o, (c, qt) in objs.items():
            adr = m.jnt_qposadr[m.body_jntadr[m.body(P.BODY[o]).id]]
            d.qpos[adr:adr + 3] = c
            d.qpos[adr + 3:adr + 7] = qt
        mujoco.mj_forward(m, d)
        pts = CV.targets(V, kind, "seat", (np.zeros(3), np.array([1.0, 0, 0, 0])), tgt)
        ok, _, _ = CV.visible(pts, m, d, d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy(),
                              CV.FOVY, 0.17)
        return float(ok.mean())

    def plan_transfer(self, q0, obj, src, dst, objs, prelook="auto", label=""):
        """보통 이송: [pick src, place dst] — 사진4 에서 도착 자리가 가리면 [look dst, pick src, place dst] 로 다시."""
        t0 = time.time()
        self.chain_src = []
        verb = "drop" if dst.startswith("waste") else "place"
        steps = [(obj, src, "pick"), (obj, dst, verb)]
        lv, info = self.plan_steps(q0, steps, objs, None, label)
        if lv is None:
            return None, info, {}
        meta = dict(seat_visible_photo4=None, prelook=False)
        L4 = next((L for L in lv if L.get("section") == "사진4"), None)
        if verb == "place" and prelook in ("auto", "always") and (L4 is not None or prelook == "always"):
            vis = self.seat_visible(np.r_[L4["q"], self.c], L4["w"], L4["objs"], L4["held"], P.KIND[obj], dst) \
                if L4 is not None else 0.0
            meta["seat_visible_photo4"] = round(vis, 2) if L4 is not None else None
            if prelook == "always" or vis < CV.PASS["seat"]:
                lv2, info2 = self.plan_steps(q0, [(obj, dst, "look")] + steps, objs, None, label)
                if lv2 is None:
                    meta["prelook_fail"] = info2
                else:
                    L0 = next(L for L in lv2 if L.get("section") == "사진0")
                    meta["seat_visible_photo0"] = round(self.seat_visible(np.r_[L0["q"], self.c], L0["w"], L0["objs"],
                                                                          None, P.KIND[obj], dst), 2)
                    L4b = next((L for L in lv2 if L.get("section") == "사진4"), None)
                    if L4b is not None:                    # 사진0 (빈손) 과 사진4 (든 채) 카메라 자세 차이
                        (p0, z0, x0), (p4, z4, x4) = self.fk(np.r_[L0["q"], self.c]), self.fk(np.r_[L4b["q"], self.c])
                        meta["photo0_4_mm"] = round(float(np.linalg.norm(p4 - p0)) * 1000, 2)
                        meta["photo0_4_deg"] = round(float(np.degrees(np.arccos(min(1.0, z0 @ z4)))), 2)
                    meta["prelook"] = True
                    lv, info = lv2, info2
        meta["plan_s"] = round(time.time() - t0, 1)
        meta["chains"] = list(self.chain_src)
        meta["cost"] = round(float(info), 1)
        return lv, info, meta
