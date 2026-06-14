# RaccoonBot OpenVLA 색상 조건 Lift 및 Stack 확장

## 최종 구성

1. **V11 Lift**
   - 자연어로 지정된 색상 실린더를 잡고 들어올리는 모델.

2. **V11+Stack**
   - V11 Lift를 기반으로 `stack the red cylinder on the blue cylinder` 같은 쌓기 명령을 추가 학습한 확장 모델.

## 레퍼런스 대비 주요 변경점

- 색상별 자연어 명령 템플릿을 확장.
- 하나의 4색 씬에서 red, blue, green, yellow 목표 에피소드를 모두 생성하도록 구성.
- grasp-only trajectory를 grasp-and-lift trajectory로 변경.
- FK 기반 end-effector command space에서 `command_delta` action label을 만들도록 변환.
- gripper close 직전 transition을 보강하는 `promote_pre_close_steps=3` 처리를 추가.
- VLA 출력만 사용하는 rollout 평가, CSV/JSON 진단, GIF 시각화 저장 루틴 추가.


## 주요 결과

### V11 Lift

```text
openvla/openvla-adapter-tmp/...--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
```

MuJoCo closed-loop rollout 결과입니다.

| 평가 항목 | 결과 |
|---|---:|
| 100 rollout strict lift 성공 | 80/100 |
| 100 rollout wrong-color touch | 9/100 |
| 32 rollout strict lift 성공 | 25/32 |
| 32 rollout wrong-color touch | 2/32 |

대표 시각화 자료는 다음 위치에 남겼습니다.

```text
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/
reports/v11_final_baseline_assets/
```

### V11+Stack

로컬에서 사용한 stack 확장 adapter는 다음 형태의 경로에 있습니다.

```text
openvla/openvla-adapter-tmp/...--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt
```

MuJoCo closed-loop rollout 결과입니다.

| 평가 항목 | 결과 |
|---|---:|
| lift command strict lift 성공 | 15/32 |
| stack command strict stack 성공 | 4/12 |
| stack command source lift 성공 | 9/12 |
| stack command final gripper open | 7/12 |

대표 시각화 자료는 다음 위치에 남겼습니다.

```text
reports/v11_plus_stack_20000_stack_gif4/
reports/v11_plus_stack_20000_lift_failgifs32/
```

## 실행 방법

### 1. V11 Lift 에피소드 생성

```bash
cd /data/Raccoonbot_Openvla

NUM_EPISODES=1200 \
DATASET_ROOT=Mujoco/raccoon_grasp_v10_lift_immediate_1200 \
scripts/05_generate_v11_lift_dataset.sh
```

### 2. V11 Lift RLDS 변환 및 TFDS 빌드

```bash
scripts/06_convert_v11_close_stable.sh
scripts/07_build_tfds_v11.sh
```

### 3. V11 Lift LoRA 학습

```bash
RUN_ID_NOTE=v11-initial-lift-close-1200eps-15000steps-b8ga2 \
MAX_STEPS=15000 \
SAVE_STEPS=2500 \
BATCH_SIZE=8 \
GRAD_ACCUMULATION_STEPS=2 \
scripts/08_train_lora_v11.sh
```

### 4. Stack 에피소드 생성 및 병합

```bash
NUM_EPISODES=120 \
DATASET_ROOT=Mujoco/raccoon_stack_v11_extension_120 \
scripts/05_generate_v11_stack_dataset.sh

scripts/06_merge_v11_stack_raw.sh
```

### 5. V11+Stack 변환 및 학습

```bash
OUT_ROOT=Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta \
PROMOTE_PRE_CLOSE_STEPS=4 \
INITIAL_CLOSE_MIN_Z_ACTION=0.012 \
STACK_RELEASE_OPEN_REPEAT=0 \
scripts/06_convert_v11_plus_stack.sh

RACCOON_RLDS_INTERMEDIATE_ROOT=Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta \
scripts/07_build_tfds_v11_plus_stack.sh

RUN_ID_NOTE=v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000 \
MAX_STEPS=20000 \
SAVE_STEPS=5000 \
BATCH_SIZE=4 \
GRAD_ACCUMULATION_STEPS=4 \
LEARNING_RATE=5e-4 \
scripts/08_train_lora_v11_plus_stack.sh
```

## 평가 방법

V11 Lift 평가:

```bash
python scripts/09_eval_v11_rollout.py \
  --adapter_path /path/to/v11_lift_adapter \
  --num_rollouts 32 \
  --run_name v11_lift_eval32
```

V11+Stack 평가:

```bash
python scripts/09_eval_v11_stack_rollout.py \
  --adapter_path /path/to/v11_plus_stack_adapter \
  --num_rollouts 12 \
  --run_name v11_plus_stack_eval12
```

## 서버 및 클라이언트 실행

```bash
ADAPTER_PATH=/path/to/adapter \
HOST=0.0.0.0 \
PORT=8000 \
DEVICE=cuda \
MERGE_ADAPTER=0 \
scripts/10_start_openvla_server.sh
```


```bash
cd ~/client

python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the red cylinder" \
  --target_color red \
  --max_steps 50 \
  --max_delta_xyz 0.12 \
  --use_viewer
```
