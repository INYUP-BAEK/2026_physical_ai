# V11 Final Close-Stable 학습 계획

작성일: 2026-06-11

## 배경

V10 1200ep / 30000step 학습 결과를 10000, 20000, 30000 checkpoint로 비교했다.

| Checkpoint | Ever strict lift | Final strict lift | Wrong-color touch | Mean final lift |
| --- | ---: | ---: | ---: | ---: |
| 10000 | 4/8 = 0.500 | 4/8 = 0.500 | 0/8 = 0.000 | 0.006381m |
| 20000 | 3/8 = 0.375 | 3/8 = 0.375 | 1/8 = 0.125 | 0.006308m |
| 30000 | 2/8 = 0.250 | 2/8 = 0.250 | 1/8 = 0.125 | 0.003270m |

추가로 10000 checkpoint를 32 rollout으로 확장 평가했다.

| 항목 | 결과 |
| --- | ---: |
| ever contact success | 18/32 = 0.562 |
| ever strict lift success | 17/32 = 0.531 |
| final strict lift success | 17/32 = 0.531 |
| wrong-color touch | 5/32 = 0.156 |
| mean final lift delta | 0.006939m |

핵심 실패 유형은 다음과 같았다.

- close가 전혀 발생하지 않음: 13/32
- close가 35 step 이후 너무 늦게 발생: 5/32
- close는 발생했지만 target contact 실패: 1/32

성공 episode에서는 close 이후 z-action이 양수로 이어지고 lift까지 도달했다. 따라서 마지막 개선의 핵심은 lift target 강화가 아니라 close 전환 안정화다.

## V11 수정 방향

V10 raw episode는 1200/1200 성공했고 물리 궤적 자체는 유효하다. 그래서 raw episode를 새로 생성하지 않고, 기존 성공 raw 1200ep를 close-stable 라벨로 재변환했다.

수정 사항:

- converter에 `--promote_pre_close_steps` 옵션 추가
- 첫 raw close 직전의 open-gripper frame 3개를 close command로 relabel
- promoted frame은 idle/small-z 필터에 의해 삭제되지 않도록 보호
- raw waypoint debug 값도 promoted frame에서는 gripper=1로 기록
- 기존 기본 동작은 `promote_pre_close_steps=0`이면 그대로 유지

의도:

- close 직전 2~3프레임은 이미 EE z가 약 1.6~1.7cm라 물리적으로 닫아도 되는 높이다.
- V10에서는 이 구간이 open으로 남아 있어서 rollout 중 모델이 계속 open을 유지하는 실패가 많았다.
- V11은 이 애매한 하강 말단 구간을 close supervision으로 바꿔 close 타이밍을 앞당기는 것을 목표로 한다.

## V11 데이터 상태

생성된 intermediate:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
```

TFDS:

```text
tensorflow_datasets/raccoon_pick_place/1.0.0
```

검증 결과:

| 항목 | 값 |
| --- | ---: |
| raw episodes | 1200 |
| train episodes | 1080 |
| val episodes | 120 |
| raw steps | 49 고정 |
| converted steps 평균 | 47.975 |
| promoted pre-close frames | 3600 |
| promoted frames per episode | 3.0 |
| close frame ratio | 0.478 |
| first close index 평균 | 25.02 |
| post-close first 8 z-action 평균 | 0.007326 |

색상 균형:

| split | red | green | blue | yellow |
| --- | ---: | ---: | ---: | ---: |
| train | 270 | 270 | 270 | 270 |
| val | 30 | 30 | 30 | 30 |

## 마지막 학습 전략

V10에서는 10000 step 이후 20000/30000에서 성능이 떨어졌다. 따라서 마지막 학습은 30000 step까지 밀지 않고, `15000 step`까지만 진행하며 `2500 step` 간격으로 checkpoint를 저장한다.

이유:

- 최적점이 10000 근처였으므로 30000은 과학습/action drift 위험이 크다.
- V11은 close supervision만 바꾼 비교적 작은 수정이라 15000 step이면 충분히 수렴 여부를 볼 수 있다.
- 2500 간격 저장으로 7500, 10000, 12500 구간을 놓치지 않는다.

권장 학습 명령:

```bash
cd /data/biy/Raccoonbot_Openvla

RUN_ID_NOTE=v11-close-stable-promote3-1200eps-15000steps-b4ga4 \
MAX_STEPS=15000 \
SAVE_STEPS=2500 \
BATCH_SIZE=4 \
GRAD_ACCUMULATION_STEPS=4 \
SAVE_LATEST_CHECKPOINT_ONLY=0 \
scripts/08_train_lora_v11.sh
```

## 학습 후 평가 계획

우선 5000, 7500, 10000, 12500, 15000 checkpoint를 8 rollout으로 빠르게 비교한다.

```bash
cd /data/biy/Raccoonbot_Openvla

for STEP in 5000 7500 10000 12500 15000; do
  ADAPTER=$(ls -td openvla/openvla-adapter-tmp/*v11-close-stable-promote3-1200eps-15000steps-b4ga4*${STEP}_chkpt | head -1)
  python scripts/09_eval_v11_rollout.py \
    --adapter_path "$ADAPTER" \
    --run_name v11_close_stable_1200_${STEP}_rollout8 \
    --num_rollouts 8 \
    --max_steps 50 \
    --no_save_frames
done
```

비교:

```bash
for f in diagnostics/v11_close_stable_1200_*_rollout8.md; do
  echo "===== $f"
  rg "ever contact success|ever strict lift success|final strict lift success|wrong-color touch|mean final lift delta" "$f"
done
```

가장 좋은 checkpoint만 32 rollout으로 확장한다.

```bash
BEST_ADAPTER=/path/to/best_adapter

python scripts/09_eval_v11_rollout.py \
  --adapter_path "$BEST_ADAPTER" \
  --run_name v11_close_stable_1200_best_rollout32 \
  --num_rollouts 32 \
  --max_steps 50 \
  --no_save_frames
```

최종 시각화:

```bash
python scripts/09_eval_v11_rollout.py \
  --adapter_path "$BEST_ADAPTER" \
  --run_name v11_close_stable_1200_best_gif4 \
  --num_rollouts 4 \
  --max_steps 50 \
  --save_frames \
  --make_gif
```

## 성공 판단 기준

V10 10000/32 rollout 기준선:

- strict lift: 17/32 = 0.531
- wrong-color touch: 5/32 = 0.156

V11이 성공적인 마지막 개선으로 보이려면 다음 중 하나 이상을 만족해야 한다.

- strict lift가 20/32 이상으로 상승
- no-close 실패가 명확히 감소
- wrong-color touch가 5/32 이하로 유지
- close first step 평균이 V10 rollout의 32.68보다 앞당겨짐

V11의 핵심 가설은 “close를 놓치는 실패를 줄이면 전체 lift 성공률이 오른다”이다.
