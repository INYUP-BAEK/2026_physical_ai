# RaccoonBot OpenVLA reference 대비 최종 V9 변경 정리

작성일: 2026-06-09

## 기준

- reference code: `/data/biy/Raccoonbot_Openvla_ref`
- final V9 project: `/data/biy/Raccoonbot_Openvla`
- 최종 학습 대상: 4색 실린더 중 자연어로 지정된 색상을 4축 RaccoonBot 로봇팔이 찾아가 파지하고 lift까지 수행하는 OpenVLA fine-tuning
- 최종 보존 버전: V9 deep-grasp command_delta, 1200 raw episode, 30000-step LoRA adapter

## 전체 구조 변화

reference는 기본적으로 단일 raw 생성, 단순 grasp, 기본 RLDS 변환, 기본 OpenVLA fine-tuning 구조였다.

최종 V9는 다음 구조로 확장됐다.

- 자연어 instruction 다양화
- 한 scene에서 red/blue/green/yellow 4색 target을 모두 수집하는 scene bundle 구조
- 단순 접촉이 아닌 close 후 lift까지 포함하는 trajectory
- FK 기반 command-space EE pose 정렬
- `command_delta` action label 도입
- scene 기준 train/val split
- 색상 task에 맞는 augmentation 조정
- dataset statistics cache 갱신 안정화
- adapter-only LoRA 학습 및 adapter evaluation 지원
- rollout 평가, diagnostics, GIF 시각화 파이프라인 추가

## 최종 주요 코드 변경 파일

reference 대비 의미 있는 변경이 있는 파일은 다음과 같다.

- `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`
- `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`
- `Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py`
- `openvla/prismatic/vla/datasets/datasets.py`
- `openvla/prismatic/vla/datasets/rlds/dataset.py`
- `openvla/prismatic/vla/datasets/rlds/utils/data_utils.py`
- `openvla/vla-scripts/finetune.py`
- `openvla/openvla_server.py`
- `scripts/05_generate_v9_stable_lift.sh`
- `scripts/06_convert_v9_fk_to_rlds.sh`
- `scripts/07_build_tfds_v9.sh`
- `scripts/08_train_lora_v9.sh`
- `scripts/09_eval_v9_rollout.py`
- `scripts/analyze_raccoon_dataset_health.py`
- `scripts/visualize_raccoon_dataset_health.py`

## Raw dataset 생성기 변경

파일: `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`

### 자연어 instruction 확장

reference는 색상 지시가 단순했다. 최종 V9는 색상별 instruction template을 확장했다.

최종 lift template 예:

- `grasp the {color} cylinder`
- `pick up the {color} cylinder`
- `lift the {color} cylinder`
- `raise the {color} cylinder`
- `grab and lift the {color} cylinder`
- `grasp and lift the {color} cylinder`
- `pick up only the {color} cylinder`
- `grasp only the {color} cylinder`
- `lift the {color} cylinder without touching the others`

최종 raw 1200ep 기준 unique instruction template은 15종이다.

### workspace 및 scene 안정화

최종 object sampling:

- x range: `[-0.10, 0.10]`
- y range: `[0.16, 0.195]`
- min object distance: `0.042m`

이 변경의 목적:

- rollout workspace와 raw generation workspace를 일치시킴
- 물리적으로 어려운 edge 배치를 줄임
- 4색 실린더가 너무 가까워져 wrong-color touch가 늘어나는 것을 완화

### FK 기반 EE pose 정렬

reference는 관측 EE pose를 MuJoCo `Link4` body pose 중심으로 기록했다. 최종 V9는 `move_to()`와 IK가 사용하는 command-space end-effector endpoint를 FK로 계산해 사용한다.

이유:

- `Link4.xpos`와 실제 controller target point 사이에 offset이 있음
- raw label, close-quality 분석, rollout delta execution이 서로 다른 좌표계를 쓰면 action label이 왜곡됨
- 최종 V9는 `get_ee_pose()`를 FK 기반으로 바꿔 생성, 변환, rollout의 좌표계를 맞춤

### deep-grasp trajectory 추가

최종 trajectory mode:

- `final_align_lift_deep`

핵심 값:

- `DEEP_GRASP_Z = 0.016`
- `DEEP_PREGRASP_Z = 0.045`
- compact waypoint schedule: `[6, 3, 4, 12, 4, 4, 3, 3, 5, 6, 2]`
- raw episode length: 53 step
- first gripper close step: 29

목적:

- 그리퍼가 실린더 윗면이 아니라 옆면 수준까지 충분히 하강한 뒤 close하도록 유도
- close 이후 lift command를 포함해 "잡기"를 넘어 "통제" 동작까지 데이터에 포함
- raw episode 길이는 너무 길지 않게 유지해 학습 비용을 제한

### close-quality success 기준 추가

최종 성공 기준:

- `strict_lift_success`
- `close_ee_z <= 0.025m`
- `close_xy_distance_to_target <= 0.006m`

기존에는 lift가 성공했더라도 close 위치가 너무 높으면 학습에 나쁜 성공 예제로 남을 수 있었다. 최종 V9에서는 close 품질이 낮은 episode를 성공으로 저장하지 않도록 했다.

### 같은 scene에서 4색 모두 수집

최종 옵션:

- `scene_reuse_all_colors=True`

목적:

- 한 배치에서 4색 실린더 위치는 동일하게 유지
- target color만 바꿔 4개의 episode 생성
- 색상 grounding 학습에서 scene variation과 color instruction을 분리

### 실패 scene 처리 및 번호 compact

문제:

- 같은 scene/color가 물리적으로 실패하는 경우 무한 재시도 또는 4색 bundle 깨짐이 발생할 수 있음
- 실패한 scene에서 이미 성공 저장된 episode를 삭제하면 episode 번호 gap이 생길 수 있음

최종 처리:

- `scene_color_max_failures=3`
- 같은 scene/color가 3회 실패하면 해당 scene 폐기
- 해당 scene의 부분 성공 episode도 rollback
- `success_counts` rollback
- 삭제된 가장 낮은 episode ID부터 재사용
- resume 시작 시 episode directory와 `meta.json["episode_id"]` compact

결과:

- 최종 raw 1200ep는 episode ID 1~1200 연속
- partial scene 0개
- scene 300개가 모두 4색 bundle 완성

## Raw to intermediate 변환 변경

파일: `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`

### FK EE pose source

추가 옵션:

- `--ee_pose_source {fk, logged}`

최종 사용:

- `ee_pose_source=fk`

목적:

- raw generator, intermediate label, rollout execution의 EE 좌표계를 command-space 기준으로 통일

### command_delta label

추가 옵션:

- `--action_label_source {next_ee_delta, command_delta}`

최종 사용:

- `action_label_source=command_delta`

정의:

```text
action[:3] = raw waypoint action[:3] - current FK EE pose[:3]
action[3:6] = 0, 0, 0
action[6] = gripper command
```

도입 이유:

- `next_ee_delta`는 실제 10Hz 한 tick 뒤 이동량을 label로 쓰기 때문에 closed-loop target command로 재사용하면 motion이 작아질 수 있음
- `command_delta`는 expert waypoint target을 직접 controller delta로 표현하므로 rollout 실행 방식과 더 잘 맞음

### idle transition drop

최종 변환은 작은 motion transition을 제거한다.

주요 기준:

- `min_joint_delta_norm=0.01`
- `min_gripper_delta=0.0001`
- `min_ee_delta_norm=0.0005`

최종 intermediate:

- raw length: 53
- converted length mean: 43.21
- total transitions: 51851

### scene 기준 split

추가 기능:

- `split_by_scene=True`

최종 split:

- train 1080 episode, 270 scene
- val 120 episode, 30 scene
- train/val scene overlap 0

목적:

- 같은 scene의 4색 target이 train과 val에 나뉘어 들어가서 leakage가 생기는 것을 방지

## TFDS builder 변경

파일: `Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py`

변경점:

- intermediate root를 hard-coded path에서 `RACCOON_RLDS_INTERMEDIATE_ROOT` 환경변수 기반으로 변경
- project-local default path 지원
- FK 기반 action build helper 추가
- TFDS feature schema는 OpenVLA 학습에 맞춰 image/state/action/language_instruction 중심으로 유지

최종 TFDS:

- `tensorflow_datasets/raccoon_pick_place/1.0.0`
- train 1080 episode
- val 120 episode
- dataset size 약 1.17GiB

## OpenVLA dataset pipeline 변경

파일: `openvla/prismatic/vla/datasets/datasets.py`

색상 기반 task에서는 hue/saturation augmentation이 color grounding을 해칠 수 있다. 최종 V9에서는 `raccoon_pick_place` dataset일 때 augmentation을 완화했다.

최종 raccoon augmentation:

- random resized crop scale: `[0.95, 1.0]`
- brightness: `[0.08]`
- contrast: `[0.9, 1.1]`
- hue/saturation augmentation 제거

## Dataset statistics cache 변경

파일:

- `openvla/prismatic/vla/datasets/rlds/dataset.py`
- `openvla/prismatic/vla/datasets/rlds/utils/data_utils.py`

문제:

- 같은 dataset name을 재사용하면 TFDS를 다시 빌드해도 기존 statistics cache를 잘못 재사용할 수 있음
- action normalization이 stale이면 정책 출력이 크게 왜곡됨

최종 변경:

- TFDS shard path, size, mtime 기반 fingerprint를 hash dependency에 추가
- `save_dir`가 명시된 경우 local stale statistics를 무조건 재사용하지 않도록 수정
- `scripts/08_train_lora_v9.sh`에서 `REFRESH_DATASET_STATS=1` 기본값으로 기존 statistics cache 삭제

최종 30000-step 학습 statistics:

- `num_trajectories=1200`
- `num_transitions=51851`
- action mean/std가 V9 final TFDS 기준으로 새로 계산됨

## LoRA 학습 변경

파일:

- `openvla/vla-scripts/finetune.py`
- `scripts/08_train_lora_v9.sh`

### adapter-only 저장

최종 기본:

- `MERGE_LORA_CHECKPOINT=0`

목적:

- full fused checkpoint 저장을 피하고 용량 절약
- final adapter만 약 463MB 수준으로 보존

### step별 adapter checkpoint 저장 보완

문제:

- 기존 `save_latest_checkpoint_only=True` 상태에서 LoRA adapter-only 학습은 같은 adapter directory를 계속 덮어씀
- 이번 30000-step 학습에서 10000/20000 adapter가 별도 보존되지 않고 최종 30000 adapter만 남음

최종 수정:

- `SAVE_LATEST_CHECKPOINT_ONLY=0` 환경변수 추가
- `finetune.py`에서 LoRA adapter-only도 `--10000_chkpt`, `--20000_chkpt` 식의 step별 adapter directory 저장 지원

주의:

- 이 수정은 이번 최종 30000 학습 이후 적용됨
- 이번 산출물에는 30000 final adapter만 존재

## OpenVLA server 변경

파일: `openvla/openvla_server.py`

변경점:

- PEFT LoRA adapter loading 지원
- local base model path 기본값 추가
- adapter path에서 processor/stat run dir 자동 추론
- default unnorm key를 `raccoon_pick_place`로 변경
- prompt instruction을 lowercase로 정규화

목적:

- fused model 없이 adapter-only checkpoint를 바로 serving/evaluation에 사용
- 다운로드 없이 local OpenVLA 7B snapshot 사용

## Rollout 평가 스크립트 추가

파일: `scripts/09_eval_v9_rollout.py`

기능:

- adapter-only OpenVLA checkpoint loading
- local base model + run statistics 자동 연결
- MuJoCo closed-loop rollout 평가
- contact success, strict lift success, wrong-color touch, final lift delta 집계
- per-step action/EE/object/contact trace 저장
- PNG frame 저장 및 GIF 생성용 자료 저장

최종 평가 스크립트 보정:

- adapter 평가 시 기본적으로 `merge_and_unload()`를 하지 않음
- PEFT wrapper와 내부 base model 양쪽에 normalization statistics 주입

이유:

- CPU merge가 7B 모델에서 매우 오래 걸림
- PEFT 상태로 GPU에 올려 평가하는 것이 빠르고 충분함
- norm stats가 wrapper 내부 base model에 전달되지 않으면 `raccoon_pick_place` unnorm key 오류 발생

## 최종 V9 shell scripts

최종 재현에 필요한 script:

- `scripts/05_generate_v9_stable_lift.sh`
- `scripts/06_convert_v9_fk_to_rlds.sh`
- `scripts/07_build_tfds_v9.sh`
- `scripts/08_train_lora_v9.sh`
- `scripts/09_eval_v9_rollout.py`
- `scripts/analyze_raccoon_dataset_health.py`
- `scripts/visualize_raccoon_dataset_health.py`

최종 raw 생성 명령:

```bash
cd /data/biy/Raccoonbot_Openvla

NUM_EPISODES=1200 \
DATASET_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v9_deep_lift_1200 \
MAX_ATTEMPTS=24000 \
SCENE_COLOR_MAX_FAILURES=3 \
MAX_CLOSE_EE_Z=0.025 \
MAX_CLOSE_XY_ERROR=0.006 \
SEED=20260608 \
bash scripts/05_generate_v9_stable_lift.sh
```

최종 변환 명령:

```bash
RAW_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v9_deep_lift_1200 \
OUT_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v9_deep_lift_1200_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
bash scripts/06_convert_v9_fk_to_rlds.sh
```

최종 TFDS build 명령:

```bash
RACCOON_RLDS_INTERMEDIATE_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v9_deep_lift_1200_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
bash scripts/07_build_tfds_v9.sh
```

최종 학습 명령:

```bash
RUN_ID_NOTE=v9-deep-command-delta-1200eps-30000steps \
MAX_STEPS=30000 \
SAVE_STEPS=10000 \
MERGE_LORA_CHECKPOINT=0 \
REFRESH_DATASET_STATS=1 \
bash scripts/08_train_lora_v9.sh
```

향후 step별 adapter를 남기려면 다음 옵션을 추가한다.

```bash
SAVE_LATEST_CHECKPOINT_ONLY=0
```

## 최종 보존 대상

정리 후 보존할 핵심 산출물:

- raw: `Mujoco/raccoon_grasp_v9_deep_lift_1200`
- intermediate: `Mujoco/raccoon_dataset/openvla_rlds_intermediate_v9_deep_lift_1200_fk_command_delta`
- TFDS: `tensorflow_datasets/raccoon_pick_place/1.0.0`
- adapter: `openvla/openvla-adapter-tmp/...v9-deep-command-delta-1200eps-30000steps--image_aug`
- processor/stat: `openvla/openvla-runs/...v9-deep-command-delta-1200eps-30000steps--image_aug`
- final health reports:
  - `reports/v9_deep_lift_1200_final_health`
  - `reports/v9_deep_lift_1200_final_intermediate_health`
  - `reports/v9_deep_command_delta_1200_30000_rollout8_20260609`
  - `reports/v9_deep_command_delta_1200_30000_rollout1_gif_20260609`
- final diagnostics:
  - `diagnostics/v9_deep_command_delta_1200_30000_rollout8_20260609.*`
  - `diagnostics/v9_deep_command_delta_1200_30000_rollout1_gif_20260609.*`

## 요약 결론

reference에서 최종 V9까지의 핵심 변화는 단순 grasp dataset을 color-conditioned, scene-bundled, FK-aligned, command-delta-labeled, lift-aware OpenVLA fine-tuning pipeline으로 확장한 것이다.

최종 모델은 target color 접근과 contact/grasp는 학습했지만, closed-loop lift 단계 전환은 아직 실패했다. 따라서 다음 lift 모델 재학습은 모델 규모나 step 수를 늘리기보다 post-close/lift label 구조를 다시 설계하는 방향이 우선이다.
