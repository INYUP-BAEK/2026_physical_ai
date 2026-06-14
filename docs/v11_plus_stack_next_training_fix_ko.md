# v11 plus stack 다음 학습 수정 기록

작성일: 2026-06-13

## 결론

기존 `v11-plus-stack 20000 step` 모델의 lift 실패 대부분은 gripper close 값이 애매하게 낮은 것이 아니라, rollout 중 close 명령이 끝까지 `0.0`으로 남는 모드 전환 실패였다. 따라서 다음 학습은 close 라벨을 더 과격하게 만드는 방향이 아니라, v11에서 안정적이었던 close/lift 문맥을 보존하면서 stack의 release/open 구간만 보강하는 방향으로 수정했다.

## 확인된 문제

- `v11 baseline`은 lift rollout 32개 기준 strict lift `25/32 = 78.1%`.
- `v11-plus-stack 20000`은 lift rollout 32개 기준 strict lift `15/32 = 46.9%`.
- 다만 pose 기준으로 보면 `ep03`, `ep26`처럼 물체가 정상적으로 올라갔지만 gripper finger contact sensor가 잡히지 않아 strict 실패로 분류된 사례가 있었다.
- 진짜 주요 실패는 `first_close=None`, 즉 gripper close action 자체가 나오지 않는 케이스였다.
- plus-stack 데이터의 close 비율은 부족하지 않았다. train split 기준:
  - v11 grasp close ratio: `0.478`
  - plus-stack grasp close ratio: `0.490`
  - plus-stack stack close ratio: `0.586`
- 문제는 close 수량 부족이 아니라, close가 나와야 하는 시각/상태 문맥이 흐려진 것이다.

## 수정 사항

### 1. close boost 복원

`scripts/06_convert_v11_plus_stack.sh`의 기본 변환 설정을 v11 안정 설정에 가깝게 되돌렸다.

- 기존 plus-stack:
  - `PROMOTE_PRE_CLOSE_STEPS=4`
  - `INITIAL_CLOSE_MIN_Z_ACTION=0.012`
- 수정 후:
  - `PROMOTE_PRE_CLOSE_STEPS=3`
  - `INITIAL_CLOSE_MIN_Z_ACTION=0.004`

`0.012m` z boost는 close와 lift를 강하게 묶는 효과가 있지만, close 전후 상태에서 action 문맥을 오염시킬 가능성이 컸다. 다음 학습에서는 v11에서 검증된 `0.004m`로 되돌린다.

### 2. stack release/open 보강

`Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`에 `--stack_release_open_repeat` 옵션을 추가했다.

- stack episode에서만 적용된다.
- 첫 close 이후 나오는 open/release frame만 반복한다.
- lift episode에는 영향을 주지 않는다.

기본값은 `scripts/06_convert_v11_plus_stack.sh`에서 `STACK_RELEASE_OPEN_REPEAT=2`로 설정했다. smoke2 변환 검증에서는 93 raw step이 95/98 converted step으로만 증가했고, first close z는 `0.004`로 정상 반영됐다.

### 3. v11 adapter에서 이어서 학습

`openvla/vla-scripts/finetune.py`에 `--init_lora_adapter_path` 옵션을 추가했다.

이제 plus-stack 학습은 base OpenVLA에서 새 LoRA를 시작하지 않고, v11 최종 adapter를 초기값으로 사용한다.

기본 adapter:

`openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt`

`scripts/08_train_lora_v11_plus_stack.sh` 기본값:

- `INIT_LORA_ADAPTER_PATH`: v11 15000 adapter
- `LEARNING_RATE=1e-4`
- `MAX_STEPS=10000`
- `SAVE_STEPS=5000`
- `RUN_ID_NOTE=v11-plus-stack-from-v11-close004-promote3-release2`

### 4. 평가 지표 보완

`scripts/09_eval_v9_rollout.py`에 pose 기반 lift 성공률을 추가했다.

새 지표:

- `ever_pose_lift_success`
- `final_pose_lift_success`
- `first_pose_lift_step`

기존 strict-contact 지표는 유지한다. 앞으로 보고서에는 strict-contact와 pose-lift를 같이 적어야 한다. contact sensor가 놓치는 성공 케이스를 과소평가하지 않기 위해서다.

## 다음 학습 명령 순서

프로젝트 루트:

```bash
cd /data/biy/Raccoonbot_Openvla
```

raw merge:

```bash
scripts/06_merge_v11_stack_raw.sh
```

RLDS intermediate 변환:

```bash
scripts/06_convert_v11_plus_stack.sh
```

TFDS build:

```bash
scripts/07_build_tfds_v11_plus_stack.sh
```

학습:

```bash
RUN_ID_NOTE=v11-plus-stack-from-v11-close004-promote3-release2-10000 \
MAX_STEPS=10000 \
SAVE_STEPS=5000 \
LEARNING_RATE=1e-4 \
scripts/08_train_lora_v11_plus_stack.sh
```

## 기대 효과

- v11의 lift/close 정책을 초기값으로 유지한다.
- stack 추가로 인한 lift 성능 붕괴를 줄인다.
- close 라벨을 과격한 z action으로 왜곡하지 않는다.
- stack의 주요 후반 실패인 `on_top_but_not_released`를 release/open frame 보강으로 직접 겨냥한다.
- 평가에서는 contact sensor false negative를 pose-lift 지표로 분리해서 해석할 수 있다.

## 학습 후 확인 포인트

1. lift rollout 32개에서 `final_pose_lift_success`와 `final_strict_lift_success`를 모두 확인한다.
2. strict 실패 중 pose 성공인 케이스는 GIF로 확인 후 별도 분류한다.
3. stack rollout 12-pair에서 `on_top_but_not_released`가 줄었는지 확인한다.
4. lift 성공률이 v11 baseline 근처로 유지되지 않으면 stack step 수 또는 학습 step 수를 더 줄여야 한다.
