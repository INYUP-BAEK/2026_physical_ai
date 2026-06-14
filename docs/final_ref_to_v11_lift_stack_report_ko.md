# RaccoonBot OpenVLA 최종 정리: Reference 대비 V11 Lift 및 V11+Stack

작성일: 2026-06-13

## 1. 정리 기준

이 문서는 원본 레퍼런스 코드인 `/data/biy/Raccoonbot_Openvla_ref`와 현재 프로젝트 `/data/biy/Raccoonbot_Openvla`의 차이를 기준으로 작성했다. 최종 보존 대상은 다음 두 축이다.

1. `V11 lift` 모델: 자연어로 지정된 색상의 실린더를 집고 들어올리는 기본 최종 모델
2. `V11+Stack` 모델: 기존 lift 능력 위에 "A 색 실린더를 B 색 실린더 위에 쌓기" task를 추가한 창의성 확장 모델

이후 close/release 강화를 더 시도했지만 5000 step 중간 평가에서 성능이 낮아 최종 후보에서 제외했다. 따라서 최종 보고서와 제출 자료는 `V11 lift 15000`과 `V11+Stack 20000` 두 모델만 기준으로 정리한다.

## 2. Reference 기본 구조

레퍼런스는 4색 실린더 중 자연어로 지정된 색상을 4축 RaccoonBot으로 grasp하는 기본 실습 코드였다.

- MuJoCo 환경에서 4색 실린더와 RaccoonBot 로봇팔을 구성한다.
- 단순 색상 instruction으로 demonstration을 생성한다.
- raw episode를 RLDS intermediate로 변환한다.
- TFDS `raccoon_pick_place`를 빌드한다.
- OpenVLA 7B에 LoRA fine-tuning을 수행한다.
- 서버에서 OpenVLA action을 계산하고, 클라이언트 또는 rollout 코드가 MuJoCo에서 실행한다.

레퍼런스의 핵심 task는 "grasp the {color} cylinder"에 가까운 단순 grasp였다. 현재 프로젝트는 이 구조를 유지하되, 데이터 생성, action label, close/lift supervision, 평가/시각화, stack task를 확장했다.

## 3. 전체 발전 흐름

초기 확장 방향은 세 가지였다.

- 입력 자연어 다양화
- 단순 grasp에서 lift까지 task 확장
- 실제 로봇 확장 시 motion 안정성을 위한 action/trajectory 정리

버전 흐름은 다음과 같이 수렴했다.

| 단계 | 핵심 변경 | 결과/판단 |
|---|---|---|
| V9 | 자연어 확장, 4색 균형 scene, FK 기반 `command_delta`, lift trajectory 도입 | 색상 접근은 학습됐지만 close/lift 안정성이 부족했다. |
| V10 | close 직후 hold를 줄이고 immediate lift label을 강화 | close가 아예 나오지 않는 `no-close` 실패가 주요 병목임을 확인했다. |
| V11 lift | close 직전 frame을 close label로 승격하고, 첫 close z-action을 양수 lift로 보정 | 100 rollout 기준 strict lift `80/100`으로 기본 최종 모델이 됐다. |
| V12 pitch 검토 | EE pitch를 활용한 vertical grasp 시나리오 검토 | 원통 실린더가 미끄러지는 물리적 한계가 커서 최종 학습에서 제외했다. |
| V11+Stack | V11 lift를 기반으로 stack task를 추가하고 LoRA adapter continuation 적용 | stack 성공 사례를 만들었지만, lift 기본 성능이 희석되어 후속 개선 대상이 됐다. |

## 4. Reference 대비 주요 코드 변경

### 4.1 MuJoCo 데이터 생성

주요 파일:

- `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`
- `Mujoco/raccoon_stack_dataset.py`
- `Mujoco/raccoon_env.py`

변경 사항:

- 한 scene에서 red/blue/green/yellow 네 색을 모두 target으로 수집하는 `scene_reuse_all_colors` 구조를 도입했다.
- target 색상별 episode 수를 균형 있게 유지하도록 생성기를 수정했다.
- 실패 episode와 성공 episode가 섞이면서 번호가 비는 문제를 고쳤고, 실패 scene은 재시도/폐기하도록 정리했다.
- instruction template을 확장했다. V11 lift는 "grasp", "pick up", "lift", "raise", "take hold" 등 다양한 표현을 사용한다.
- grasp-only가 아니라 close 이후 target cylinder를 실제로 들어올리는 trajectory를 생성한다.
- stack task 전용 생성기 `raccoon_stack_dataset.py`를 추가했다.
- stack instruction은 안정적 task 분리를 위해 단일 템플릿을 사용한다.

Stack instruction:

```text
stack the {source_color} cylinder on the {base_color} cylinder
```

### 4.2 RLDS intermediate 변환

주요 파일:

- `Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`
- `Mujoco/raccoon_dataset/merge_raw_datasets.py`
- `Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py`

변경 사항:

- action label을 joint target이 아니라 command-space EE delta로 바꿨다.

```text
action = [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
```

- EE pose는 raw log의 target만 믿지 않고 FK 기반 command-space endpoint로 재계산한다.
- rotation action은 최종 V11/V11+Stack에서는 0으로 유지한다. pitch 실험 코드는 검토했지만 최종 task에서는 끄는 방향으로 정리했다.
- scene 기준 train/val split을 적용해 같은 scene의 색상 episode가 train/val에 섞이지 않도록 했다.
- idle step, post-close hold, closed 상태의 작은 z-action transition을 필터링했다.
- close 직전 open frame을 close label로 승격하는 `promote_pre_close_steps`를 추가했다.
- 첫 close frame의 z-action을 최소 양수 lift action으로 보정하는 `initial_close_min_z_action`을 추가했다.
- stack episode에서 release/open supervision이 약해지는 문제를 줄이기 위해 `stack_release_open_repeat`를 추가했다.

### 4.3 OpenVLA 학습 코드

주요 파일:

- `openvla/vla-scripts/finetune.py`
- `scripts/08_train_lora_v11.sh`
- `scripts/08_train_lora_v11_plus_stack.sh`

변경 사항:

- LoRA adapter-only 학습을 기본으로 사용한다.
- 디스크와 메모리 절약을 위해 `MERGE_LORA_CHECKPOINT=0`으로 adapter checkpoint만 남길 수 있게 했다.
- V11+Stack 학습에서 base OpenVLA에서 새 LoRA를 시작하지 않고, V11 lift 최종 adapter를 초기값으로 이어서 학습할 수 있게 `--init_lora_adapter_path`를 추가했다.
- dataset statistics cache를 매번 갱신해 이전 TFDS 통계가 새 데이터셋에 남아 영향을 주지 않도록 했다.

### 4.4 Rollout 평가 및 시각화

주요 파일:

- `scripts/09_eval_v11_rollout.py`
- `scripts/09_eval_v11_rollout_core.py`
- `scripts/09_eval_v11_stack_rollout.py`
- `scripts/analyze_raccoon_dataset_health.py`
- `scripts/visualize_raccoon_dataset_health.py`

변경 사항:

- VLA 출력 action만으로 closed-loop rollout을 수행한다.
- 별도 외부 controller가 목표를 보정해 성공률을 올리는 방식은 최종 평가에 사용하지 않는다.
- workspace clipping, max delta 제한, z min/max 제한은 물리 안전장치이며 target으로 끌어주는 보조 controller가 아니다.
- lift 평가는 contact 기반 strict metric과 pose 기반 lift metric을 함께 확인할 수 있게 했다.
- stack 평가는 source lift, stack xy/z, final gripper open, non-source touch를 분리해 기록한다.
- `summary.json`, diagnostics CSV/MD, GIF를 남겨 보고서 시각화 자료로 쓸 수 있게 했다.

## 5. 최종 보존 모델 1: V11 Lift

### 5.1 목적

자연어 명령에 포함된 색상 정보를 보고 4색 실린더 중 target만 선택한 뒤, 로봇팔이 close 후 lift까지 수행하는 모델이다.

### 5.2 데이터

Raw episode:

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200
```

Intermediate:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
```

주요 설정:

| 항목 | 값 |
|---|---:|
| raw episode | 1200 |
| train / val | 1080 / 120 |
| target color | red/green/blue/yellow 각 300 |
| action label | `command_delta` |
| EE pose source | `fk` |
| `promote_pre_close_steps` | 3 |
| `initial_close_min_z_action` | 0.004 |

### 5.3 최종 checkpoint

Adapter:

```text
openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
```

Run directory:

```text
openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
```

### 5.4 평가 결과

100 rollout 평가:

| 항목 | 결과 |
|---|---:|
| ever strict lift | 80/100 |
| final strict lift | 80/100 |
| wrong-color touch | 9/100 |
| exception | 0 |

32 rollout 평가:

| 항목 | 결과 |
|---|---:|
| ever strict lift | 25/32 |
| final strict lift | 25/32 |
| wrong-color touch | 2/32 |
| exception | 0 |

시각화/평가 자료:

```text
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4
reports/v11_initial_lift_close_1200_b8ga2_15000_rollout32
reports/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs
reports/v11_final_baseline_assets
diagnostics/v11_initial_lift_close_1200_b8ga2_15000_*
```

### 5.5 해석

V11 lift는 본 프로젝트의 기본 최종 모델이다. 이전 버전에서 가장 큰 실패였던 `no-close`와 `late-close`를 converter label 보정으로 줄였고, closed-loop rollout에서 목표 실린더를 실제로 lift하는 성능을 확인했다. 실패가 남는 경우도 대부분 색상 이해 실패보다는 close trigger, 배치상 접근 난이도, contact 판정의 엄격함에 의한 것이다.

## 6. 최종 보존 모델 2: V11+Stack

### 6.1 목적

V11 lift의 grasp/lift 능력을 유지하면서, 자연어 stack 명령을 통해 source cylinder를 base cylinder 위로 이동시키는 task를 추가했다. 과제의 창의성 요소를 보여주기 위한 확장 모델이다.

### 6.2 데이터

Stack raw episode:

```text
Mujoco/raccoon_stack_v11_extension_120
```

V11 lift 1200ep와 stack 120ep를 병합한 raw dataset:

```text
Mujoco/raccoon_grasp_v11_plus_stack_raw
```

V11+Stack 20000 step 모델에 사용된 intermediate:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta
```

V11+Stack 20000 intermediate 주요 설정:

| 항목 | 값 |
|---|---:|
| total episode | 1320 |
| train / val | 1191 / 129 |
| base lift raw | 1200 |
| stack raw | 120 |
| action label | `command_delta` |
| EE pose source | `fk` |
| `promote_pre_close_steps` | 4 |
| `initial_close_min_z_action` | 0.012 |
| `stack_release_open_repeat` | 미적용 |

### 6.3 최종 checkpoint

Adapter:

```text
openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt
```

Run directory:

```text
openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt
```

### 6.4 평가 결과

Lift command 평가:

| 항목 | 결과 |
|---|---:|
| ever strict lift | 15/32 |
| final strict lift | 15/32 |
| wrong-color touch | 3/32 |
| exception | 0 |

Stack command 평가:

| 항목 | 결과 |
|---|---:|
| final strict stack | 4/12 |
| ever source lift | 9/12 |
| final gripper open | 7/12 |
| non-source robot touch | 0/12 |
| base robot touch | 0/12 |

Stack 실패 분류:

| 유형 | 개수 |
|---|---:|
| strict_stack_success_clean | 4 |
| on_top_but_not_released | 5 |
| no_source_grasp_or_lift | 3 |

시각화/평가 자료:

```text
reports/v11_plus_stack_20000_rollout32
reports/v11_plus_stack_20000_lift_failgifs32
reports/v11_plus_stack_20000_stack_rollout12_pairs
reports/v11_plus_stack_20000_stack_gif4
diagnostics/v11_plus_stack_20000_*
```

### 6.5 해석

V11+Stack은 task 확장 자체는 가능함을 보여줬다. source cylinder를 들어올려 base 위로 이동시키는 중간 단계는 12 rollout 중 9회 관찰됐고, strict stack 성공도 4회 있었다. 다만 lift-only 성능은 V11 baseline보다 낮아졌다. 이는 stack task가 추가되면서 close/release/open 문맥이 복잡해지고, 모델이 close mode 전환을 더 불안정하게 배운 것으로 해석된다.

이후 close/release supervision을 더 강화한 5000 step 중간 모델도 시험했지만, lift 8 rollout에서 `1/8` 성공에 그쳐 최종 후보에서 제외했다. 따라서 최종 제출에서는 안정성이 높은 V11 Lift와 창의성 확장 근거가 남아 있는 V11+Stack 20000을 중심 결과로 사용한다.

## 7. 최종 남긴 실행 스크립트

정리 후 남긴 scripts:

```text
scripts/05_generate_v11_lift_dataset.sh
scripts/05_generate_v11_stack_dataset.sh
scripts/06_convert_v11_close_stable.sh
scripts/06_convert_v11_plus_stack.sh
scripts/06_merge_v11_stack_raw.sh
scripts/07_build_tfds_v11.sh
scripts/07_build_tfds_v11_plus_stack.sh
scripts/08_train_lora_v11.sh
scripts/08_train_lora_v11_plus_stack.sh
scripts/09_eval_v11_rollout.py
scripts/09_eval_v11_rollout_core.py
scripts/09_eval_v11_stack_rollout.py
scripts/10_start_openvla_server.sh
scripts/analyze_raccoon_dataset_health.py
scripts/visualize_raccoon_dataset_health.py
```

이전 `v9` 이름으로 남아 있던 평가 core와 생성 스크립트는 현재 용도에 맞게 rename했다.

- `05_generate_v9_stable_lift.sh` -> `05_generate_v11_lift_dataset.sh`
- `05_generate_stack_smoke.sh` -> `05_generate_v11_stack_dataset.sh`
- `09_eval_v9_rollout.py` -> `09_eval_v11_rollout_core.py`

삭제한 구버전 스크립트:

```text
scripts/06_convert_v9_fk_to_rlds.sh
scripts/07_build_tfds_v9.sh
scripts/08_train_lora_v9.sh
scripts/09_wait_and_train_v11_plus_stack_20000.sh
```

## 8. 정리 후 남긴 주요 자료

### 데이터

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200
Mujoco/raccoon_stack_v11_extension_120
Mujoco/raccoon_grasp_v11_plus_stack_raw
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta
```

주의: `tensorflow_datasets/raccoon_pick_place/1.0.0`은 GitHub 제출 대상이 아니며, 재현 시 위 intermediate에서 다시 빌드한다.

### 모델

```text
openvla/openvla-adapter-tmp/...--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
openvla/openvla-runs/...--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
openvla/openvla-adapter-tmp/...--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt
openvla/openvla-runs/...--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt
```

## 9. 삭제한 자료

정리하면서 다음 자료를 삭제했다.

- stack smoke raw data: `raccoon_stack_v11_extension_smoke2`, `raccoon_stack_v11_extension_smoke40`
- V10 rollout reports 및 diagnostics
- V11 lift 중간 checkpoint 평가 reports: 2500, 5000, 7500, 10000, 12500, 15000 rollout8
- V11+Stack smoke rollout reports
- V11 lift 중간 adapter checkpoints: 2500, 5000, 7500, 10000, 12500
- V11+Stack 중간 adapter checkpoints: 5000, 10000, 15000
- 실패한 quantization/noquant 초기 run 골격
- 최종 후보에서 제외한 close004 5000 step checkpoint, intermediate, TFDS, 부분 rollout 산출물
- Python `__pycache__`

보존 기준은 "최종 보고서에 필요한 v11 lift와 v11+stack의 데이터, checkpoint, 핵심 평가/시각화 자료"다.

## 10. 최종 판단

제출/보고서 관점에서 가장 안정적인 핵심 결과는 V11 lift다. 레퍼런스의 단순 grasp task를 넘어 자연어 다양화, target lift, close supervision 개선, VLA-only rollout 평가까지 포함하므로 기본 과제 요구사항과 성능 개선 근거가 명확하다.

V11+Stack은 성공률 자체는 아직 낮지만, 기존 pick/lift 능력을 조합해 새로운 조작 task를 시도했다는 점에서 창의성 요소로 가치가 있다. 특히 실패가 무작위가 아니라 `release/open` 및 `close mode 전환`으로 좁혀졌기 때문에, 최종 보고서에서는 완성 모델이 아니라 창의적 확장 실험과 한계 분석으로 제시하는 것이 가장 정확하다.

## 11. Client 및 실제 로봇 실행 계획

서버 inference, client MuJoCo viewer, 실제 RaccoonBot 동시 실행 절차는 별도 문서에 정리했다.

```text
docs/client_real_robot_test_plan_ko.md
```

이 절차는 GPU가 비어 있을 때 서버를 띄운 뒤, client에서 MuJoCo only smoke test를 먼저 수행하고, 마지막에 `--use_real_robot`를 켜 실제 로봇이 같은 VLA action 흐름을 따라가도록 하는 순서다.
