# V9 1200ep + 30000-step 최종 학습 결과 정리

작성일: 2026-06-09

## 목적

이번 V9 최종 실험의 목적은 자연어 명령에 포함된 색상 정보를 바탕으로 4색 실린더 중 지정된 target을 선택하고, 4축 RaccoonBot 로봇팔로 파지한 뒤 lift까지 수행하는 OpenVLA policy를 학습하는 것이다.

최종 사용 파이프라인:

- raw dataset: `final_align_lift_deep`
- action label: `command_delta`
- EE pose source: FK command-space endpoint
- dataset size: 1200 episode
- train/val split: scene 기준 1080/120
- training: OpenVLA 7B LoRA adapter-only
- max steps: 30000

## 최종 raw dataset

경로:

```text
Mujoco/raccoon_grasp_v9_deep_lift_1200
```

생성 설정:

- `NUM_EPISODES=1200`
- `TRAJECTORY_MODE=final_align_lift_deep`
- `SCENE_REUSE_ALL_COLORS=1`
- `SCENE_COLOR_MAX_FAILURES=3`
- `MAX_CLOSE_EE_Z=0.025`
- `MAX_CLOSE_XY_ERROR=0.006`
- `MIN_OBJECT_DISTANCE=0.042`
- `OBJECT_Y_MAX=0.195`
- `SEED=20260608`

최종 raw health:

- episode: 1200
- episode ID: `000001~001200`, 누락 0개
- scene: 300
- scene당 target color: 4개
- partial scene: 0개
- 색상별 episode: red/blue/green/yellow 각 300개
- success: 1200/1200
- strict lift success: 1200/1200
- close quality success: 1200/1200
- raw step length: 53
- first gripper close step: 29
- instruction templates: 15종
- trajectory mode: `final_align_lift_deep`

최종 raw 분포:

- `target_x min/mean/max = -0.09997 / 0.00069 / 0.09984m`
- `target_y min/mean/max = 0.16000 / 0.17708 / 0.19500m`
- `target_y > 0.200m = 0/1200`
- `close_ee_z min/mean/max = 0.01558 / 0.01603 / 0.01855m`
- `close_xy_error min/mean/max = 0.00008 / 0.00039 / 0.00596m`
- `target_lift_delta min/mean/max = 0.02519 / 0.03564 / 0.03913m`

raw 시각화:

- `reports/v9_deep_lift_1200_final_health/visualization_index.md`
- `reports/v9_deep_lift_1200_final_health/raw/target_color_count.png`
- `reports/v9_deep_lift_1200_final_health/raw/target_xy_scatter.png`
- `reports/v9_deep_lift_1200_final_health/raw/target_y_hist.png`
- `reports/v9_deep_lift_1200_final_health/raw/close_xy_error_hist.png`
- `reports/v9_deep_lift_1200_final_health/raw/target_lift_delta_hist.png`

## Intermediate 변환 결과

경로:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v9_deep_lift_1200_fk_command_delta
```

변환 설정:

- `ee_pose_source = fk`
- `action_label_source = command_delta`
- `drop_idle_steps = True`
- `split_by_scene = True`
- `min_joint_delta_norm = 0.01`
- `min_gripper_delta = 0.0001`
- `min_ee_delta_norm = 0.0005`

변환 결과:

- episode JSON: 1200개
- train episode: 1080
- val episode: 120
- train scene: 270
- val scene: 30
- train/val scene overlap: 0
- raw length: 전부 53
- converted length min/mean/max: `40 / 43.21 / 53`
- total converted transitions: 51851
- action label source: `command_delta`
- EE pose source: `fk`

action 분포:

- `action_xyz_norm min/mean/max = 0.000085 / 0.018818 / 0.109565m`
- `action_xyz_norm < 0.5mm = 3480/51851 = 0.067`
- `action_xyz_norm < 1.0mm = 4212/51851 = 0.081`
- `action_xyz_norm < 2.0mm = 5537/51851 = 0.107`
- `gripper_close_ratio = 21620/51851 = 0.417`

close/lift label 방향성:

- `first_close_z mean = -0.000034m`
- `after_close_2_5_z mean = 0.010158m`
- `post_close_z mean = 0.009597m`

해석:

- offline label 기준으로는 close 이후 lift 방향 z action이 존재한다.
- 즉 raw/intermediate 자체는 lift label을 포함하고 있다.
- 그러나 최종 rollout에서는 post-contact z action이 거의 0으로 붕괴되었으므로, 문제는 closed-loop policy의 stage 전환 학습 실패로 해석된다.

intermediate 포함 시각화:

- `reports/v9_deep_lift_1200_final_intermediate_health/visualization_index.md`
- `reports/v9_deep_lift_1200_final_intermediate_health/intermediate/episode_length_hist.png`
- `reports/v9_deep_lift_1200_final_intermediate_health/intermediate/action_xyz_norm_hist.png`
- `reports/v9_deep_lift_1200_final_intermediate_health/intermediate/z_action_by_phase_boxplot.png`
- `reports/v9_deep_lift_1200_final_intermediate_health/intermediate/instruction_count.png`

## TFDS build 결과

경로:

```text
tensorflow_datasets/raccoon_pick_place/1.0.0
```

결과:

- dataset size: 약 1.17GiB
- 실제 디렉터리 용량: 약 1.2GB
- train split: 1080 episode, 16 shard
- val split: 120 episode, 1 shard

feature schema:

- `episode_metadata.episode_id`: int32
- `episode_metadata.success`: bool
- `episode_metadata.goal_xy`: float32 `(2,)`
- `episode_metadata.box_init_xy`: float32 `(2,)`
- `steps.observation.image`: uint8 `(256, 256, 3)`
- `steps.observation.state`: float32 `(8,)`
- `steps.action`: float32 `(7,)`
- `steps.language_instruction`: string
- `steps.reward`, `discount`, `is_first`, `is_last`, `is_terminal`

loader 검증:

- `tfds.builder('raccoon_pick_place', data_dir='/data/biy/Raccoonbot_Openvla/tensorflow_datasets')` 로드 성공
- sample image shape: `(256, 256, 3)`
- sample state shape: `(8,)`
- sample action shape: `(7,)`
- sample instruction 예: `pick up the cylinder that is blue`

## 30000-step LoRA 학습

학습 명령:

```bash
cd /data/biy/Raccoonbot_Openvla

RUN_ID_NOTE=v9-deep-command-delta-1200eps-30000steps \
MAX_STEPS=30000 \
SAVE_STEPS=10000 \
MERGE_LORA_CHECKPOINT=0 \
REFRESH_DATASET_STATS=1 \
bash scripts/08_train_lora_v9.sh
```

학습 설정:

- base model: local OpenVLA 7B snapshot
- dataset: `raccoon_pick_place`
- LoRA rank: 32
- batch size: 8
- grad accumulation: 2
- effective batch size: 16
- learning rate: `5e-4`
- max steps: 30000
- save steps: 10000
- full merge: disabled
- dataset statistics refresh: enabled

학습 결과:

- `30000/30000` step 정상 완료
- final LoRA adapter 저장 완료
- full fused model은 저장하지 않음

최종 산출물:

```text
openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v9-deep-command-delta-1200eps-30000steps--image_aug
```

processor/stat run dir:

```text
openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v9-deep-command-delta-1200eps-30000steps--image_aug
```

보존 파일:

- `adapter_model.safetensors`: 약 484MB
- `adapter_config.json`
- `README.md`
- `dataset_statistics.json`
- tokenizer/processor files

dataset statistics:

- `num_trajectories = 1200`
- `num_transitions = 51851`
- action mean: `[0.000039, 0.002675, -0.004346, 0, 0, 0, 0.416964]`
- action std: `[0.013878, 0.008575, 0.020272, 0, 0, 0, 0.493011]`

중요한 주의:

- 이번 학습의 10000/20000 adapter는 별도 보존되지 않았다.
- 원인은 기존 `finetune.py`가 LoRA adapter-only 저장에서도 같은 adapter directory를 덮어썼기 때문이다.
- 이후 재학습에서는 `SAVE_LATEST_CHECKPOINT_ONLY=0`을 추가하면 step별 adapter가 보존되도록 코드 수정이 완료되어 있다.

## 8-rollout 평가 결과

평가 대상:

- final 30000-step adapter

평가 명령 요약:

```bash
python scripts/09_eval_v9_rollout.py \
  --adapter_path openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v9-deep-command-delta-1200eps-30000steps--image_aug \
  --run_name v9_deep_command_delta_1200_30000_rollout8_20260609 \
  --num_rollouts 8 \
  --max_steps 50 \
  --step_seconds 0.10 \
  --no_center_crop \
  --no_save_frames
```

결과 파일:

- `diagnostics/v9_deep_command_delta_1200_30000_rollout8_20260609.json`
- `diagnostics/v9_deep_command_delta_1200_30000_rollout8_20260609.csv`
- `diagnostics/v9_deep_command_delta_1200_30000_rollout8_20260609.md`
- `reports/v9_deep_command_delta_1200_30000_rollout8_20260609/summary.json`

요약 결과:

- rollout count: 8
- ever contact success: `8/8 = 1.000`
- final contact success: `8/8 = 1.000`
- ever strict lift success: `0/8 = 0.000`
- final strict lift success: `0/8 = 0.000`
- transient wrong-color touch: `1/8 = 0.125`
- mean final lift delta: `0.000081m`
- mean steps: 50

색상별 결과:

- blue: contact 2/2, strict lift 0/2, wrong touch 0/2
- green: contact 2/2, strict lift 0/2, wrong touch 0/2
- red: contact 2/2, strict lift 0/2, wrong touch 0/2
- yellow: contact 2/2, strict lift 0/2, transient wrong touch 1/2

wrong-color touch 해석:

- `wrong-color touch`는 최종적으로 잘못된 색을 잡았다는 뜻이 아니다.
- 평가 코드는 rollout 중 한 번이라도 target 외 색상과 robot body contact가 기록되면 `ever_wrong_color_touch=True`로 둔다.
- GIF용 yellow rollout에서는 step 16에서 green contact가 1프레임 기록되었고, final touching color는 yellow였다.
- 따라서 보고서 표현은 `transient wrong-color contact`가 더 정확하다.

## Action phase 분석

8-rollout diagnostics에서 phase별 action을 분석했다.

pre-contact:

- z action mean: `-0.013294m`
- z action median: `-0.003425m`
- gripper command mean: `0.000`
- EE z mean: `0.06102m`

post-contact:

- z action mean: `0.000166m`
- z action median: `0.000016m`
- z action max: `0.000329m`
- gripper command mean: `0.996`
- EE z mean: `0.01608m`
- lift delta mean: 약 `0.000075m`

해석:

- 모델은 목표 색상 실린더로 접근하고 접촉하는 정책은 학습했다.
- 접촉 이후 gripper close command도 거의 항상 출력한다.
- 그러나 close 이후 z 방향 lift action이 거의 0으로 붕괴되어 실제 lift가 발생하지 않는다.
- 즉 현재 모델은 color-conditioned grasp/contact policy로는 유효하지만, lift manipulation policy로는 실패했다.

## GIF 시각화

1개 rollout을 프레임 저장 모드로 다시 실행하고 GIF를 생성했다.

run:

```text
v9_deep_command_delta_1200_30000_rollout1_gif_20260609
```

대상:

- target color: yellow
- instruction: `grasp only the yellow cylinder`
- result: contact success, strict lift failure, transient green contact at step 16
- final touching color: yellow
- final lift delta: `0.000052m`
- steps: 50

시각화 파일:

- GIF: `reports/v9_deep_command_delta_1200_30000_rollout1_gif_20260609/v9_deep_command_delta_1200_30000_rollout1.gif`
- frame dir: `reports/v9_deep_command_delta_1200_30000_rollout1_gif_20260609/episode_0001_yellow/`
- contact sheet: `reports/v9_deep_command_delta_1200_30000_rollout1_gif_20260609/wrong_touch_window_contact_sheet.png`
- diagnostics:
  - `diagnostics/v9_deep_command_delta_1200_30000_rollout1_gif_20260609.json`
  - `diagnostics/v9_deep_command_delta_1200_30000_rollout1_gif_20260609.csv`
  - `diagnostics/v9_deep_command_delta_1200_30000_rollout1_gif_20260609.md`

GIF 용도:

- 모델이 목표 실린더 근처로 접근하고 접촉하는 것은 확인 가능
- 접촉 후 실린더를 들어올리지 못하고 낮은 z 위치에 머무르는 실패 양상을 시각적으로 보여줌
- 다음 lift 모델 재학습의 실패 원인 설명 자료로 사용

## VLA-only 동작 여부 검증

질문:

- 현재 30000-step V9 모델의 contact/grasp 성공이 정말 VLA 출력의 결과인가?
- 중간에 외부 controller가 target 위치를 보고 성공률을 높이는 식의 보정이나 꼼수가 들어갔는가?

검증 대상:

- `scripts/09_eval_v9_rollout.py`
- `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`
- diagnostics: `diagnostics/v9_deep_command_delta_1200_30000_rollout8_20260609.json`

### 모델 입력

rollout loop에서 모델에 들어가는 값은 다음 두 가지다.

```python
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
```

즉 VLA policy 입력은 다음뿐이다.

- RGB image
- natural language prompt

`object_pose`, `target_pose`, `target_body_name`, `object_specs`는 모델 입력으로 들어가지 않는다.

`env.get_observation(target_body_name)`은 다음 값을 반환하지만,

- image
- joint angles
- gripper state
- object pose
- EE pose

rollout policy call에서는 이 중 image만 processor에 전달된다. object pose와 target body 정보는 diagnostics, success 판정, lift delta 계산에만 쓰인다.

### 외부 controller 역할

평가에는 deterministic low-level controller가 존재한다.

- VLA output: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- 실행: 현재 EE pose + VLA delta를 `env.move_to()`의 IK target으로 전달
- gripper: VLA gripper output이 0.5 이상이면 close, 아니면 open

따라서 엄밀히 말하면 이 시스템은 torque-level pure VLA가 아니라, OpenVLA가 EE delta command를 내고 RaccoonBot의 기본 IK/move_to controller가 이를 실행하는 구조다.

하지만 이 low-level controller는 다음을 하지 않는다.

- target cylinder 위치로 자동 이동
- 목표 색상을 보고 경로 보정
- grasp 성공을 위해 손을 target으로 snap
- lift 성공을 위해 z action을 외부에서 추가
- wrong color를 피하도록 경로를 재계획

### workspace clip / IK retry 영향 검증

`execute_delta_action()`에는 다음 안정장치가 있다.

- delta scale
- max delta clipping
- workspace xyz clipping
- IK 실패 시 delta shrink retry

이 장치가 성공률을 올리는 꼼수로 작동했는지 8-rollout diagnostics의 전체 400 step을 검사했다.

검사 결과:

- total rollout steps: 400
- raw VLA delta와 executed delta가 달라진 step: 0
- IK retry 발생 step: 0
- max delta clipping 발생 step: 0
- raw/executed delta 최대 차이: 0

즉 이번 8-rollout 평가에서 실제 실행된 이동 delta는 VLA가 출력한 delta와 동일했다. contact success 8/8은 workspace clipping이나 IK retry가 action을 고쳐서 만든 결과가 아니다.

### target 정보 사용 여부

`target_body_name`은 다음 용도로 사용된다.

- target object 초기 z 저장
- target object final z 계산
- target contact/lift success 판정
- diagnostics에 target pose 기록

정책 action 계산에는 사용되지 않는다.

`object_specs`도 scene reset과 diagnostics 저장에 쓰이며, policy에는 전달되지 않는다. 정책이 받는 target 정보는 자연어 instruction 안의 색상 단어뿐이다.

### 결론

현재 V9 30000-step rollout 결과는 "외부 oracle controller가 target을 잡아준 결과"로 보기는 어렵다.

검증상:

- VLA가 prompt와 image만 보고 action을 출력한다.
- object pose나 target pose는 model input에 들어가지 않는다.
- low-level IK controller는 VLA delta를 그대로 실행한다.
- 8-rollout 전체 400 step에서 clipping/retry/shrink 보정은 0회였다.
- gripper close도 VLA action[6]에 대한 threshold 실행일 뿐이다.

따라서 현재 contact/grasp 성능은 VLA 출력으로 얻은 결과라고 정리할 수 있다.

단, 보고서 표현에서 주의할 점:

- "오직 VLA만으로 작동한다"라고 쓰면 torque-level controller까지 VLA가 한다는 오해가 생길 수 있다.
- 더 정확한 표현은 "VLA가 RGB image와 language instruction만으로 EE delta와 gripper command를 출력하고, 별도 target oracle 없이 기본 IK controller가 해당 action을 그대로 실행했다"이다.

추천 보고서 문장:

> V9 30000-step 모델의 closed-loop rollout은 RGB image와 language instruction만을 OpenVLA 입력으로 사용했으며, object pose나 target pose는 policy 입력에 제공되지 않았다. 실행부는 VLA가 출력한 EE delta와 gripper command를 기본 IK/move_to controller로 변환해 수행했으며, 8-rollout 전체 400 step에서 workspace clipping, IK retry, delta shrink 보정은 발생하지 않았다. 따라서 관찰된 contact/grasp 성공은 외부 oracle controller가 target으로 보정한 결과가 아니라 VLA policy 출력에 의해 달성된 것으로 볼 수 있다.

## 최종 결론

V9 1200ep + 30000-step 모델은 색상 기반 target 선택과 contact/grasp 단계에서는 높은 성공률을 보였다.

정량적으로는 8-rollout에서 contact success가 8/8이었다. 따라서 "입력된 자연어 색상 명령에 맞는 실린더를 찾아가 잡는 모델"로는 의미 있는 성과가 있다.

하지만 strict lift success는 0/8이었다. 실패 원인은 접촉 이후 lift 단계에서 z action이 거의 0으로 붕괴되는 것이다. gripper close는 출력되지만, closed-loop policy가 close 이후 lift stage로 전환하지 못한다.

따라서 현재 모델은 최종 manipulation 모델이 아니라, grasp/contact까지 성공한 중간 성과 모델로 기록한다. 다음 lift 모델 재학습에서는 학습 step을 더 늘리는 것보다 post-close/lift label 구조를 수정하는 것이 우선이다.

## 다음 lift 재학습 전 수정 방향

우선순위:

1. post-close hold transition 제거 또는 downweight
2. first close 이후 첫 closed-frame부터 lift target action이 나오도록 trajectory/converter 수정
3. closed gripper 상태에서 z action이 작은 transition을 학습셋에서 제거
4. 40~120ep smoke dataset으로 rollout을 먼저 확인
5. 성공 시 1200ep 재생성 및 10000/20000/30000 step별 adapter 저장 학습

다음 학습 명령에는 반드시 다음 옵션을 포함한다.

```bash
SAVE_LATEST_CHECKPOINT_ONLY=0
```

그래야 10000/20000/30000 adapter를 비교 평가할 수 있다.
