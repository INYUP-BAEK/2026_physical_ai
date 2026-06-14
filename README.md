# RaccoonBot OpenVLA 색상 조건 Lift 및 Stack 확장

## 최종 구성

이번 정리에서 최종적으로 남긴 핵심 모델은 두 가지입니다.

1. **V11 Lift**
   - 자연어로 지정된 색상 실린더를 잡고 들어올리는 기본 최종 모델입니다.
   - 현재 프로젝트에서 가장 안정적인 모델입니다.

2. **V11+Stack**
   - V11 Lift를 기반으로 `stack the red cylinder on the blue cylinder` 같은 쌓기 명령을 추가 학습한 창의성 확장 모델입니다.
   - 쌓기 성공률은 아직 낮지만, lift 능력을 placement 동작으로 확장했다는 점이 차별점입니다.

## 레퍼런스 대비 주요 변경점

- 색상별 자연어 명령 템플릿을 확장했습니다.
- 하나의 4색 씬에서 red, blue, green, yellow 목표 에피소드를 모두 생성하도록 구성했습니다.
- grasp-only trajectory를 grasp-and-lift trajectory로 바꿨습니다.
- FK 기반 end-effector command space에서 `command_delta` action label을 만들도록 변환했습니다.
- gripper close 직전 transition을 보강하는 `promote_pre_close_steps=3` 처리를 추가했습니다.
- 첫 close frame부터 lift target action이 나오도록 `initial_close_min_z_action=0.004`를 적용했습니다.
- stack dataset generator와 stack rollout evaluator를 추가했습니다.
- V11+Stack은 V11 Lift LoRA adapter에서 이어서 학습하도록 수정했습니다.
- VLA 출력만 사용하는 rollout 평가, CSV/JSON 진단, GIF 시각화 저장 루틴을 추가했습니다.
- 서버 추론 코드와 클라이언트 MuJoCo/실제 RaccoonBot 실행 절차를 정리했습니다.


## 저장소 구조

```text
Mujoco/                         MuJoCo 환경, 데이터 생성기, RLDS 변환 코드
scripts/                        에피소드 생성, 변환, 학습, 평가, 서버 실행 스크립트
openvla/                        수정한 OpenVLA 학습 및 추론 서버 코드
dlimp_openvla/                  로컬 RLDS/dlimp 유틸리티
docs/                           한글 분석 문서와 실행 절차
reports/                        선택된 결과 이미지, GIF 시각화
diagnostics/                    rollout CSV/JSON/MD 진단 결과
logs/                           선택된 학습/변환/평가 로그
report.md                       제출용 보고서 원문
README.md                       저장소 실행 안내서
```

## 주요 결과

### V11 Lift

로컬에서 사용한 최종 adapter는 다음 형태의 경로에 있습니다. 실제 체크포인트는 GitHub에 업로드하지 않습니다.

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

아래 명령은 서버 로컬에 대용량 데이터셋과 체크포인트가 존재한다고 가정합니다. 이 파일들은 GitHub에는 포함하지 않습니다.

### 1. V11 Lift 에피소드 생성

```bash
cd /data/biy/Raccoonbot_Openvla

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

평가 결과는 `reports/`와 `diagnostics/` 아래에 저장됩니다.

## 서버 및 클라이언트 실행

서버에서 OpenVLA 추론 서버를 실행합니다.

```bash
ADAPTER_PATH=/path/to/adapter \
HOST=0.0.0.0 \
PORT=8000 \
DEVICE=cuda \
MERGE_ADAPTER=0 \
scripts/10_start_openvla_server.sh
```

클라이언트에서 MuJoCo 테스트를 실행합니다.

```bash
cd /data/biy/client

python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the red cylinder" \
  --target_color red \
  --max_steps 50 \
  --max_delta_xyz 0.12 \
  --use_viewer
```

실제 RaccoonBot 연결 절차는 [client_real_robot_test_plan_ko.md](docs/client_real_robot_test_plan_ko.md)에 정리했습니다.

## 제출 전 확인

```bash
cd /data/biy/Raccoonbot_Openvla
git status --ignored
git add .
git status --short
```
