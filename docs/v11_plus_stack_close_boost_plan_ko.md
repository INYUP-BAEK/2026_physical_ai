# V11 Plus Stack 및 Close 명령 보강 계획

작성일: 2026-06-12

## 목적

최종 모델의 기본 성공률은 V11 수평 grasp/lift를 기반으로 유지하되, 과제 창의성 요소를 위해 별도 stack task를 추가한다. 동시에 V11 rollout에서 관찰된 `close 명령이 충분히 강하게 나오지 않는 문제`를 데이터 변환 단계에서 보강한다.

## 핵심 방향

### 1. V11 grasp/lift 유지

기존 1200 episode는 그대로 사용한다.

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200
```

이 데이터는 기본 색상 선택, 접근, deep close, immediate lift를 학습시키는 주력 데이터다.

### 2. Stack episode 추가

새 stack task는 기존 V11 trajectory를 확장한다.

```text
stack the {source_color} cylinder on the {base_color} cylinder
```

템플릿은 1개로 고정한다. 자연어 다양성보다 task 구분 안정성을 우선한다.

trajectory:

```text
source 위 safe approach
-> V11 deep grasp
-> close
-> immediate lift
-> base cylinder 위로 이동
-> 천천히 하강
-> gripper open
-> retreat
```

초기 smoke 결과:

```text
Mujoco/raccoon_stack_v11_extension_smoke2
```

2개 episode 모두 성공했다.

| episode | instruction | close z | stack xy error | stack z delta |
|---|---|---:|---:|---:|
| 000001 | stack the red cylinder on the blue cylinder | 0.0167m | 0.0006m | 0.0204m |
| 000002 | stack the red cylinder on the yellow cylinder | 0.0160m | 0.0005m | 0.0200m |

40 episode smoke 결과:

```text
Mujoco/raccoon_stack_v11_extension_smoke40
```

- 저장 episode: 40개
- 생성 시도: 42회
- raw episode 길이: 93 frame 고정
- 성공 episode: 40/40
- close 품질: 40/40 통과
- strict stack 성공: 40/40

품질 통계:

| 항목 | min | mean | max |
|---|---:|---:|---:|
| close ee z | 0.01585m | 0.01616m | 0.01674m |
| close xy error | 0.000001m | 0.000182m | 0.001089m |
| target lift delta | 0.01966m | 0.02009m | 0.02051m |
| final stack xy error | 0.000478m | 0.000528m | 0.000647m |
| final stack z delta | 0.02000m | 0.02004m | 0.02043m |

변환 smoke 점검:

- 40/40 episode 변환 성공
- 변환 후 길이: 77~81 step
- close action: 평균 46.9 step
- close 상태 upward z action: 평균 18 step
- close 상태 수평 이동 action: 평균 3.6 step
- promoted pre-close step: 모든 episode 4 step

해석:

- stack task 자체는 물리적으로 안정적이다.
- close 명령 보강이 실제 변환 action에 반영됐다.
- closed gripper 상태의 수평 이동이 필터에 의해 제거되지 않고 보존됐다.
- stack episode는 최종 학습에 추가해도 raw 품질 측면에서는 문제가 작다.

## 120 Episode 최종 Stack 추가 데이터 검증

최종 추가용 stack raw:

```text
Mujoco/raccoon_stack_v11_extension_120
```

생성 결과:

- 저장 episode: 120개
- 생성 시도: 122회
- 실패 시도: 2회
- 누락 episode 번호: 없음
- `success=True`: 120/120
- `strict_stack_success=True`: 120/120
- `close_quality_success=True`: 120/120
- raw episode 길이: 93 frame 고정

색상 분포:

- source color: red/blue/green/yellow 각각 30개
- base color: red/blue/green/yellow 각각 30개
- ordered source/base pair 12개가 각각 10개씩 포함됨

품질 통계:

| 항목 | min | mean | p95 | max |
|---|---:|---:|---:|---:|
| close ee z | 0.01591m | 0.01614m | 0.01630m | 0.01655m |
| close xy error | 0.000019m | 0.000171m | 0.000291m | 0.003699m |
| target lift delta | 0.01817m | 0.02010m | 0.02029m | 0.02301m |
| final stack xy error | 0.000411m | 0.000553m | 0.000636m | 0.003178m |
| final stack z delta | 0.020001m | 0.020029m | 0.020183m | 0.020372m |

outlier:

- `episode_000109`가 close xy error `0.0037m`, stack xy error `0.00318m`로 가장 크다.
- 기준치 `0.006m` 이내이며, 최종 stack 상태도 프레임 확인 결과 정상이다.
- 전체 120개 중 1개 outlier라 학습 분포를 흔들 수준은 아니므로 유지한다.

변환 검증:

임시 intermediate 변환으로 120/120 episode가 성공적으로 변환되는 것을 확인했다.

- 변환 후 step 수: 77~81 step
- close action: 평균 46.8 step
- open action: 평균 33.0 step
- closed 상태 upward z action: 평균 18.0 step
- closed 상태 수평 이동 action: 평균 3.58 step
- promoted pre-close step: 모든 episode 4 step

해석:

- close 명령이 충분히 자주 보존되어 rollout에서 close 누락을 줄이는 방향이다.
- stack 후반의 closed-horizontal transport가 필터에 의해 제거되지 않았다.
- release/open 구간도 task 구분에 필요한 만큼 포함되어 있다.

병합 검증:

기존 v11 1200ep와 stack 120ep를 hardlink 기반 임시 병합해 검사했다.

```text
grasp: 1200 episode
stack: 120 episode
total: 1320 episode
```

검증 결과:

- 누락 episode 번호 없음
- 성공 episode 1320/1320
- stack pair 균형 유지
- 병합 후 stack episode는 뒤쪽 번호 `episode_001201`부터 들어감
- v11과 stack의 scene_id 충돌 방지를 위해 stack scene_id는 offset 적용됨

최종 판단:

`Mujoco/raccoon_stack_v11_extension_120`은 기존 1200ep와 병합해 최종 학습에 사용할 수 있다.

## Close 명령 보강

기존 변환 단계의 문제:

- close command frame이 episode 안에서 상대적으로 희소하다.
- V11에서는 close 직후 lift를 바로 수행하므로, 모델이 close action을 강하게 학습하지 못하면 rollout에서 그립 실패가 난다.
- stack task에서는 closed 상태로 수평 이동하는 프레임이 생기므로, 기존 `closed gripper + small z action` 필터가 핵심 이동 프레임을 잘못 제거할 수 있다.

반영한 수정:

1. `PROMOTE_PRE_CLOSE_STEPS=4`
   - 첫 raw close 직전의 deep-z open frame 4개를 close command로 재라벨링한다.
   - 모델이 “충분히 내려간 뒤 닫기”를 더 자주 본다.

2. `INITIAL_CLOSE_MIN_Z_ACTION=0.012`
   - promoted close frame과 첫 raw close frame의 z action이 최소 12mm 위쪽을 가리키도록 한다.
   - close 후 바로 lift로 이어지는 라벨을 강화한다.

3. closed-frame 필터 보정
   - 기존에는 closed 상태에서 z action이 작으면 제거했다.
   - 이제는 z action도 작고 xy action도 작을 때만 제거한다.
   - 따라서 stack에서 물체를 잡은 채 base 위로 수평 이동하는 프레임은 보존된다.

4. stack source/base pair 균형 생성
   - 최종 stack 추가 생성에서는 12개 ordered color pair를 균등하게 채운다.
   - 120ep 생성 시 각 pair가 10개씩 들어간다.
   - 실패 retry가 있어도 최종 성공 episode 분포가 균형을 유지하도록 수정했다.

5. merge 시 scene id offset 적용
   - 기존 v11 raw와 stack raw는 각각 scene_id가 1부터 시작할 수 있다.
   - 병합 후 `split_by_scene` 검증 분리에서 서로 다른 데이터셋의 scene_id가 충돌하지 않도록 input root별 scene_id offset을 적용한다.

## 추가 파일

```text
Mujoco/raccoon_stack_dataset.py
Mujoco/raccoon_dataset/merge_raw_datasets.py
scripts/05_generate_stack_smoke.sh
scripts/06_merge_v11_stack_raw.sh
scripts/06_convert_v11_plus_stack.sh
scripts/07_build_tfds_v11_plus_stack.sh
scripts/08_train_lora_v11_plus_stack.sh
```

## 권장 실행 순서

### 1. Stack smoke 생성

```bash
MUJOCO_GL=egl \
DATASET_ROOT=Mujoco/raccoon_stack_v11_extension_smoke40 \
NUM_EPISODES=40 \
MAX_ATTEMPTS=800 \
SEED=20260614 \
scripts/05_generate_stack_smoke.sh
```

### 2. 기존 1200ep와 stack raw 병합

```bash
BASE_RAW_ROOT=Mujoco/raccoon_grasp_v10_lift_immediate_1200 \
STACK_RAW_ROOT=Mujoco/raccoon_stack_v11_extension_smoke40 \
OUT_ROOT=Mujoco/raccoon_grasp_v11_plus_stack_raw \
scripts/06_merge_v11_stack_raw.sh
```

### 3. RLDS intermediate 변환

```bash
RAW_ROOT=Mujoco/raccoon_grasp_v11_plus_stack_raw \
OUT_ROOT=Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta \
scripts/06_convert_v11_plus_stack.sh
```

### 4. TFDS 빌드

```bash
RACCOON_RLDS_INTERMEDIATE_ROOT=Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_plus_stack_fk_command_delta \
scripts/07_build_tfds_v11_plus_stack.sh
```

### 5. LoRA 학습

```bash
RUN_ID_NOTE=v11-plus-stack-close-boost \
MAX_STEPS=15000 \
SAVE_STEPS=5000 \
BATCH_SIZE=4 \
GRAD_ACCUMULATION_STEPS=4 \
scripts/08_train_lora_v11_plus_stack.sh
```

## 주의

stack episode는 창의성 및 다단계 조작 확장 목적이다. 최종 기본 grasp 성공률을 망치지 않기 위해 stack 비율은 처음에는 작게 유지한다.

추천:

```text
v11 grasp/lift: 1200ep
stack: 40~160ep
```

40ep smoke 후 성공률과 변환 결과가 안정적이면 120~160ep로 늘리는 것이 좋다.
