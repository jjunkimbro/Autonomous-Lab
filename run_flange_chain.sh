#!/bin/bash
# 카메라를 flange D415 로 바꾼 뒤 관측 자세 · 등록 · 정확도 · 워크플로우를 다시 만든다
set -e
export PYTHONIOENCODING=utf-8
cd /c/Users/jkim1/autonomous_lab
py vision_observation_v8.py fluent_plate incubator_plate reader_plate sorter_left_plate plate_hotel_4 plate_hotel_3 > chain_obs.log 2>&1
py register_markers_v8.py > chain_reg1.log 2>&1
py vision_accuracy_test_v8.py --trials 20 > chain_acc.log 2>&1
py vision_observation_loaded_v8.py > chain_loaded.log 2>&1
py register_markers_v8.py > chain_reg2.log 2>&1
py vision_workflow_v8.py > workflow_all.log 2>&1
py vision_workflow_v8.py --scenario route_a --scenario fluent_bumped --scenario fluent_taken --record > workflow_record.log 2>&1
echo CHAIN_DONE
