실험실 셀 배치 v21 — T3 장비 순서: STX44(배양기) - Reader(리더기) - SH800(소팅기) - NGS - 폐기통
(테이블·리니어 스테이지·로봇 위치는 v19 지금 배치 그대로: T2-T1 간격 211 mm, 스테이지 T1 앞뒤 중앙)

[VSCode 에서 MuJoCo 로 돌리기]
  1) zip 을 풀고 VSCode 로 폴더 열기 (mujoco/ 폴더 구조 그대로 두기 — assets/ 경로를 상대경로로 씀)
  2) 터미널:  pip install mujoco numpy scipy
  3) 장면만 보기:      python -m mujoco.viewer --mjcf mujoco/lab_layout_v21_reorder.xml
     동작 재생 (13단계): python mujoco/play_viewer.py            (--speed 2 배속, 스페이스 일시정지)
  4) (선택) 경로 다시 계획 (Mac/Linux bash 기준, 약 30 s):
       cd mujoco
       RAIL_DX=0.15 RAIL_DY=-0.39 SEED=0 BACK_C=rup WASTE=1 NGS=1 CARRY_Z=1.0 PLAN_ONLY=1 DUMP=my_plan.pkl python planner/pnp_video_v13.py lab_layout_v21_reorder.xml out.mp4
       python play_viewer.py --plan my_plan.pkl
     Windows PowerShell 은 환경변수를 $env:RAIL_DX="0.15"; ... 처럼 먼저 설정
  ※ 재생은 계획된 관절 궤적을 그대로 보여주는 운동학 재생. 힘·마찰로 집는 물리 시뮬레이션은 아님.

[파일]
  mujoco/lab_layout_v21_reorder.xml  MuJoCo 장면 (메시·텍스처 assets/ 포함)
  mujoco/plan_v21.npz                계획된 동작 (레일 이동 -> 팔 이동, 레일 215 mm/s, 81.5 s)
  mujoco/play_viewer.py              뷰어 재생 스크립트
  mujoco/planner/                    경로 계획기 (pnp_video_v13.py + check_reach_mockup_v12.py)
  lab_layout_v21_reorder.glb         Blender 등 3D 툴용 (File > Import > glTF 2.0), 단위 m
  table_layout_v21_reorder_topview.png, layout_dimensions_top_v21.png  배치도·치수도

장비 x 범위 (m): STX44 0.895~1.045 | Reader 1.100~1.350 | SH800 1.410~1.660 | NGS 1.704~1.954 | 폐기통 2.000~2.200
검증: 13단계 모두 충돌 없음, 80~81 s, 팔끼리 최소 38 mm, 스테이지 91 mm, 테이블 15 mm
