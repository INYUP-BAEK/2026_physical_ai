# V10 1200ep / 30000step Rollout 검증 분석

작성일: 2026-06-10

## 검증 대상

- 데이터셋: `Mujoco/raccoon_grasp_v10_lift_immediate_1200`
- RLDS intermediate: `Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_1200_fk_command_delta`
- TFDS: `tensorflow_datasets/raccoon_pick_place/1.0.0`
- 학습 run note: `v10-lift-immediate-1200eps-30000steps-b4ga4`
- 저장 체크포인트:
  - `10000_chkpt`
  - `20000_chkpt`
  - `30000_chkpt`

## Rollout 요약

8 rollout 기준으로는 `10000 step` 체크포인트가 가장 좋았다.

| Checkpoint | Ever contact | Ever strict lift | Final strict lift | Wrong-color touch | Mean final lift delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10000 | 4/8 = 0.500 | 4/8 = 0.500 | 4/8 = 0.500 | 0/8 = 0.000 | 0.006381m |
| 20000 | 3/8 = 0.375 | 3/8 = 0.375 | 3/8 = 0.375 | 1/8 = 0.125 | 0.006308m |
| 30000 | 2/8 = 0.250 | 2/8 = 0.250 | 2/8 = 0.250 | 1/8 = 0.125 | 0.003270m |

결론적으로 현재 결과에서는 `30000 step`이 최종 후보가 아니라, `10000 step`을 우선 후보로 두고 더 큰 rollout 평가를 해야 한다.

## 주요 실패 패턴

실패의 주원인은 lift 단계가 약한 것이 아니라 `close gripper` 전환이 나오지 않는 것이다.

- 성공 episode에서는 close 이후 z-action이 양수로 이어지고 strict lift까지 도달한다.
- 실패 episode 대부분은 전체 50 step 동안 gripper command가 0에 머물러 close가 아예 발생하지 않는다.
- `30000 step`에서는 일부 episode에서 close 타이밍이 늦거나, 한 episode는 너무 이른 close가 나와서 안정성이 더 나빠졌다.

체크포인트별 close 관찰:

- `10000`: 성공 episode 4개는 close가 27~29 step 부근에서 발생했고, 실패 episode 4개는 close가 전혀 발생하지 않았다.
- `20000`: 성공 episode 3개는 close가 26~33 step 부근에서 발생했다. 한 red episode는 44 step에 너무 늦게 close되어 lift 성공으로 이어지지 못했다.
- `30000`: 성공 episode 2개만 정상 close/lift를 수행했다. 한 green episode는 16 step에 이른 close가 발생했지만 접촉/lift에는 실패했다.

## 데이터셋 상태

데이터셋 자체의 색상 균형과 split은 정상이다.

- 총 episode: 1200
- 색상별 episode: red 300, green 300, blue 300, yellow 300
- train: 색상별 270
- val: 색상별 30
- converted total steps: 54199
- converted 평균 step: 약 45.17
- gripper close/action=1 step 비율: 약 44.18%

즉, close 라벨이 데이터셋에서 희소한 문제는 아니다. 문제는 closed 상태로 전환해야 하는 관찰 구간을 모델이 rollout 중 안정적으로 인식하지 못하는 쪽에 가깝다.

## 해석

현재 V10 방향성은 V9보다 lift target을 더 명확하게 만든 점에서는 개선이다. 실제로 close가 발생한 rollout에서는 lift action이 잘 따라온다. 하지만 최종 성공률은 close 전환 안정성에 의해 제한되고 있다.

`10000 -> 20000 -> 30000`으로 갈수록 성능이 좋아지지 않고 오히려 하락했기 때문에, 이 설정에서는 장시간 학습이 정책을 더 안정화하지 못했다. 특히 `30000 step`은 wrong-color touch가 생기고 평균 lift delta도 낮아져 최종 후보로 쓰기 어렵다.

가능한 원인은 다음과 같다.

- 동일 데이터셋 반복 학습이 길어지면서 특정 시각/언어 패턴에 과적합 또는 action drift가 발생했다.
- close 직전 관찰 상태가 충분히 다양하지 않아 rollout 중 작은 위치 오차가 누적되면 close 전환을 놓친다.
- 현재 평가 8개는 scene 1, 2만 사용하므로 통계적으로 작지만, 같은 scene 안에서도 색/문장에 따라 close 여부가 갈리는 점은 정책 출력 안정성 문제를 강하게 시사한다.

## 다음 권장 검증

최종 모델 후보는 `10000 step`으로 두고, 먼저 32 rollout 이상으로 재평가한다.

```bash
cd /data/biy/Raccoonbot_Openvla

ADAPTER_10000=$(ls -td openvla/openvla-adapter-tmp/*v10-lift-immediate-1200eps-30000steps-b4ga4*10000_chkpt | head -1)

python scripts/09_eval_v9_rollout.py \
  --adapter_path "$ADAPTER_10000" \
  --run_name v10_lift_immediate_1200_10000_rollout32 \
  --num_rollouts 32 \
  --max_steps 50 \
  --no_save_frames
```

32 rollout에서 `10000 step`이 유지되면 해당 체크포인트를 최종 후보로 두고 GIF/보고서 자료를 생성한다.

```bash
python scripts/09_eval_v9_rollout.py \
  --adapter_path "$ADAPTER_10000" \
  --run_name v10_lift_immediate_1200_10000_gif4 \
  --num_rollouts 4 \
  --max_steps 50 \
  --save_frames \
  --make_gif
```

## 10000 Step 32 Rollout 추가 검증

`10000 step` 체크포인트를 32 rollout으로 확장 평가한 결과, 8 rollout에서 보였던 우위가 유지되었다.

| 평가 항목 | 결과 |
| --- | ---: |
| ever contact success | 18/32 = 0.562 |
| ever strict lift success | 17/32 = 0.531 |
| final strict lift success | 17/32 = 0.531 |
| wrong-color touch | 5/32 = 0.156 |
| mean final lift delta | 0.006939m |
| mean steps | 44.03 |

색상별 결과는 다음과 같다.

| 색상 | strict-ever | strict-final | wrong-touch | mean final lift |
| --- | ---: | ---: | ---: | ---: |
| blue | 4/8 = 0.500 | 4/8 = 0.500 | 2/8 = 0.250 | 0.007248m |
| green | 4/8 = 0.500 | 4/8 = 0.500 | 2/8 = 0.250 | 0.006389m |
| red | 3/8 = 0.375 | 3/8 = 0.375 | 0/8 = 0.000 | 0.004580m |
| yellow | 6/8 = 0.750 | 6/8 = 0.750 | 1/8 = 0.125 | 0.009538m |

실패 유형을 분해하면 다음과 같다.

- close가 전혀 발생하지 않은 episode: 13/32
- close가 35 step 이후 너무 늦게 발생한 episode: 5/32
- close는 발생했지만 target contact가 없었던 episode: 1/32
- wrong-color touch가 있으면서도 strict lift는 성공한 episode: 2/32

정책 출력 검증:

- IK retry: 0회
- raw action과 executed action 차이: 0/1409 step
- close 발생 episode의 첫 close step 평균: 32.68
- close 이후 첫 8 step z-action 평균: 0.008074

따라서 현재 rollout은 외부 컨트롤러 보정이나 IK retry로 성공한 것이 아니라, VLA action이 그대로 실행된 결과다. 다만 실패 episode 상당수는 VLA가 close 명령을 내지 않거나 너무 늦게 내는 패턴이다.

이 결과 기준 최종 후보는 `30000 step`이 아니라 `10000 step`이다. 단, 성공률이 약 53%이고 wrong-color touch가 15.6%이므로 “완성 모델”보다는 “현재까지 가장 유망한 checkpoint”로 표현하는 것이 정확하다.

## 다음 학습 개선안

다음 재학습을 한다면 30000 step을 기본 목표로 두지 말고 다음 방향을 우선 적용한다.

1. `10000 step` 전후를 더 촘촘히 저장한다. 예: 5000, 7500, 10000, 12500, 15000.
2. close 전환 직전/직후 상태를 더 다양화한다. 특히 target 위에서 약간의 xy/z 오차가 있는 상태에서도 close로 들어가는 데이터를 추가한다.
3. close 이후 lift 라벨은 현재처럼 유지한다. close가 발생한 경우에는 lift는 비교적 정상적으로 따라간다.
4. 평가는 8 rollout이 아니라 최소 32 rollout 기준으로 checkpoint를 고른다.
5. 최종 보고서에는 10000/20000/30000 비교표와 GIF를 같이 남긴다.

## 관련 산출물

- 요약 md:
  - `diagnostics/v10_lift_immediate_1200_10000_rollout8.md`
  - `diagnostics/v10_lift_immediate_1200_20000_rollout8.md`
  - `diagnostics/v10_lift_immediate_1200_30000_rollout8.md`
- 상세 json:
  - `diagnostics/v10_lift_immediate_1200_10000_rollout8.json`
  - `diagnostics/v10_lift_immediate_1200_20000_rollout8.json`
  - `diagnostics/v10_lift_immediate_1200_30000_rollout8.json`
- 생성 GIF:
  - `reports/v10_lift_immediate_1200_30000_gif1/v10_lift_immediate_1200_30000_gif1_episode_0001_yellow.gif`
