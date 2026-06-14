# V10 Lift Immediate 재학습 수정 및 실행 계획

작성일: 2026-06-09

## 목적

V9 1200ep + 30000step 모델은 색상별 cylinder 접촉/그립은 의미 있게 학습했지만, closed-loop rollout에서 lift action이 충분히 유지되지 않아 strict lift 성공률이 0/8이었다.

이번 V10 방향은 기존 V9의 장점인 색상 선택과 접근/그립 능력은 유지하면서, close 이후 정책이 낮은 z 위치에 머무르지 않고 즉시 lift action을 출력하도록 데이터와 변환 파이프라인을 수정하는 것이다.

## 반영된 코드 수정

### 0. 과거 trajectory 코드 정리

파일: `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`

V10 재학습 전용 코드로 정리하면서 더 이상 사용하지 않는 trajectory 분기를 제거했다.

제거한 실행 분기:

- `baseline`
- `final_align`
- `lift_explicit`
- `final_align_lift_ramp`
- `final_align_lift_direct`
- `final_align_lift_deep`

현재 generator가 받는 trajectory mode는 다음 하나뿐이다.

```text
final_align_lift_deep_immediate
```

기존 실험 흐름과 비교 설명은 문서에 남기고, 실제 데이터 생성 코드는 V10 immediate lift trajectory만 생성하도록 단순화했다. 또한 v8/v9 전용 metadata 이름을 새 raw episode에서는 `lift_target_z`, `lift_delta_z`, `waypoint_steps`, `grasp_target_z`처럼 현재 목적에 맞는 이름으로 정리했다.

### 1. post-close hold transition 제거

파일: `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`

새 trajectory mode를 추가했다.

```text
final_align_lift_deep_immediate
```

기존 V9 `final_align_lift_deep`는 다음 구조였다.

```text
safe approach -> pregrasp -> deep descend -> deep close -> deep close hold -> lift mid -> lift high
```

새 V10 mode는 다음 구조다.

```text
safe approach -> pregrasp -> deep descend -> first close frame -> lift mid -> lift high
```

핵심 차이는 close 이후 low-z closed hold waypoint를 제거한 것이다. 첫 close command는 유지하되, 그 다음 closed-frame부터 lift target action이 나오도록 waypoint step을 조정했다.

### 2. first close 이후 첫 closed-frame부터 lift target label 출력

파일: `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`

V10 mode의 waypoint step은 다음과 같다.

```text
[6, 3, 4, 12, 4, 1, 8, 8, 2]
```

의미:

```text
safe approach 6
safe align 3
pregrasp 4
deep descend 12
open deep settle 4
first close 1
lift mid 8
lift high 8
high hold 2
```

이 구조에서는 closed gripper image가 들어오는 시점부터 action label의 z 성분이 lift 방향으로 커진다.

### 3. closed gripper + small z transition 필터

파일: `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`

추가 옵션:

```text
--drop_post_close_hold_steps
--drop_closed_gripper_small_z_actions
--closed_gripper_min_z_action
```

기본 변환 스크립트에서는 다음 값으로 켜져 있다.

```text
DROP_POST_CLOSE_HOLD_STEPS=2
DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS=1
CLOSED_GRIPPER_MIN_Z_ACTION=0.002
```

필터 원칙:

- 첫 close frame은 항상 보존한다.
- terminal frame은 보존한다.
- 첫 close 이후 닫힌 gripper 상태에서 z action 절댓값이 `0.002`보다 작은 transition은 제거한다.
- 제거 카운트는 episode metadata와 manifest에 기록한다.

기존 V9 episode 1개에 대한 임시 변환 확인 결과:

```text
num_steps_raw: 53
num_steps_converted: 40
dropped_idle_steps: 11
dropped_post_close_hold_steps: 2
dropped_closed_small_z_steps: 0
```

### 4. 기본 출력 경로 변경

기존 V9 최종 데이터가 덮어써지지 않도록 새 기본 경로를 사용한다.

raw:

```text
Mujoco/raccoon_grasp_v10_lift_immediate
```

intermediate:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_fk_command_delta
```

### 5. step별 adapter 저장

파일: `scripts/08_train_lora_v9.sh`

`SAVE_LATEST_CHECKPOINT_ONLY=0`이 기본값이다. 따라서 본학습에서 `MAX_STEPS=30000`, `SAVE_STEPS=10000`으로 실행하면 다음 adapter가 보존된다.

```text
10000 checkpoint
20000 checkpoint
30000 final
```

## Smoke Dataset 실행 순서

먼저 40ep 또는 120ep로 짧게 검증한다. 40ep는 빠른 구조 검증용, 120ep는 색상/scene 다양성을 조금 더 본 rollout 검증용이다.

### 1. 40ep 생성

```bash
cd /data/biy/Raccoonbot_Openvla

DATASET_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v10_lift_immediate_smoke40 \
NUM_EPISODES=40 \
SEED=1010 \
MAX_ATTEMPTS=8000 \
TRAJECTORY_MODE=final_align_lift_deep_immediate \
bash scripts/05_generate_v9_stable_lift.sh
```

### 2. 40ep intermediate 변환

```bash
RAW_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v10_lift_immediate_smoke40 \
OUT_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_smoke40_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
DROP_POST_CLOSE_HOLD_STEPS=2 \
DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS=1 \
CLOSED_GRIPPER_MIN_Z_ACTION=0.002 \
bash scripts/06_convert_v9_fk_to_rlds.sh
```

### 3. TFDS 빌드

주의: 현재 builder dataset name은 `raccoon_pick_place` 하나이므로 TFDS build는 기존 `tensorflow_datasets/raccoon_pick_place`를 새 데이터셋으로 교체한다.

```bash
RACCOON_RLDS_INTERMEDIATE_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_smoke40_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
bash scripts/07_build_tfds_v9.sh
```

### 4. smoke LoRA 학습

```bash
RUN_ID_NOTE=v10-lift-immediate-smoke40-5000steps \
MAX_STEPS=5000 \
SAVE_STEPS=2500 \
SAVE_LATEST_CHECKPOINT_ONLY=0 \
REFRESH_DATASET_STATS=1 \
bash scripts/08_train_lora_v9.sh
```

### 5. smoke rollout 평가

학습 후 adapter path를 확인한다.

```bash
ls -td openvla/openvla-adapter-tmp/*v10-lift-immediate-smoke40-5000steps* | head -1
```

확인된 adapter path를 넣어 rollout을 실행한다.

```bash
python scripts/09_eval_v9_rollout.py \
  --adapter_path <ADAPTER_PATH> \
  --run_name v10_lift_immediate_smoke40_rollout8 \
  --num_rollouts 8 \
  --max_steps_per_rollout 50
```

GIF가 필요하면 1개 rollout을 frame 저장으로 다시 실행한다.

```bash
python scripts/09_eval_v9_rollout.py \
  --adapter_path <ADAPTER_PATH> \
  --run_name v10_lift_immediate_smoke40_gif1 \
  --num_rollouts 1 \
  --max_steps_per_rollout 50 \
  --save_frames \
  --make_gif
```

## 1200ep 본학습 실행 순서

smoke rollout에서 contact/grasp가 유지되고 z action이 lift 방향으로 살아나면 1200ep를 재생성한다.

### 1. 1200ep 생성

```bash
cd /data/biy/Raccoonbot_Openvla

DATASET_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v10_lift_immediate_1200 \
NUM_EPISODES=1200 \
SEED=2026 \
MAX_ATTEMPTS=60000 \
TRAJECTORY_MODE=final_align_lift_deep_immediate \
bash scripts/05_generate_v9_stable_lift.sh
```

### 2. 1200ep 변환

```bash
RAW_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_grasp_v10_lift_immediate_1200 \
OUT_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_1200_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
DROP_POST_CLOSE_HOLD_STEPS=2 \
DROP_CLOSED_GRIPPER_SMALL_Z_ACTIONS=1 \
CLOSED_GRIPPER_MIN_Z_ACTION=0.002 \
bash scripts/06_convert_v9_fk_to_rlds.sh
```

### 3. TFDS 빌드

```bash
RACCOON_RLDS_INTERMEDIATE_ROOT=/data/biy/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_1200_fk_command_delta \
ACTION_LABEL_SOURCE=command_delta \
bash scripts/07_build_tfds_v9.sh
```

### 4. 30000step 학습

```bash
RUN_ID_NOTE=v10-lift-immediate-1200eps-30000steps \
MAX_STEPS=30000 \
SAVE_STEPS=10000 \
SAVE_LATEST_CHECKPOINT_ONLY=0 \
REFRESH_DATASET_STATS=1 \
bash scripts/08_train_lora_v9.sh
```

평가 대상 adapter는 10000, 20000, 30000 세 개다.

```bash
ls -td openvla/openvla-adapter-tmp/*v10-lift-immediate-1200eps-30000steps*
```

## 평가 기준

우선순위:

1. target color contact/grasp 성공률
2. wrong-color touch 발생 여부
3. first close 이후 평균 z action 증가 여부
4. object z 상승량
5. strict lift success

V10의 핵심 검증 포인트는 단순히 contact success가 유지되는지가 아니라, closed gripper phase에서 z action이 V9보다 커졌는지다.

## 클라이언트-서버 실행 방향

현재 서버 측에는 `openvla/openvla_server.py`와 실행 스크립트 `scripts/10_start_openvla_server.sh`를 준비했다.

서버 실행 예:

```bash
ADAPTER_PATH=<ADAPTER_PATH> \
HOST=0.0.0.0 \
PORT=8000 \
bash scripts/10_start_openvla_server.sh
```

서버 endpoint:

```text
GET  /health
POST /predict
```

`/predict` 입력:

```json
{
  "instruction": "lift the red cylinder",
  "image_b64": "<base64 encoded RGB image>",
  "unnorm_key": "raccoon_pick_place",
  "do_sample": false
}
```

`/predict` 출력:

```json
{
  "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
  "unnorm_key": "raccoon_pick_place",
  "prompt": "..."
}
```

중요한 점:

- 서버는 image와 instruction만 policy 입력으로 사용한다.
- object pose, target pose, 색상 좌표는 policy 입력으로 제공하지 않는다.
- adapter-only 실행이 기본이다. `MERGE_ADAPTER=1`을 명시할 때만 merge한다.

클라이언트 쪽은 기존 v8 client 코드가 있으면 그 코드의 image capture, MuJoCo/real robot command execution, logging 구조를 유지하고 `/predict` 호출부만 현재 서버 endpoint에 맞추는 방식이 가장 안전하다.

보류 중인 클라이언트 구현 항목:

- v8 client 코드의 image format 확인
- action scaling 및 gripper command mapping 확인
- MuJoCo client와 real robot client의 공통 action interface 분리
- server action log와 client execution log를 같은 timestamp로 저장
