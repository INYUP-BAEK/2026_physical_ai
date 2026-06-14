# V11 Final Baseline 정리

작성일: 2026-06-12

## 결론

V11은 본 프로젝트의 기본 모델 최종본으로 확정한다. 최종 checkpoint는 `15000 step` adapter이며, 100회 closed-loop rollout 기준 목표 실린더 lift 성공률은 `80/100`이다. 다른 실린더 touch는 최종 성공/실패 판정에서 제외하고, 별도의 간섭 품질 지표로 기록한다.

V11의 핵심 성과는 다음과 같다.

- 단순 grasp가 아니라 목표 실린더를 lift threshold 이상 들어올리는 동작까지 학습했다.
- 4색 자연어 지시를 균형 있게 처리한다.
- 실패 대부분이 색상 인식 실패가 아니라 마지막 close trigger 또는 기구 간섭 문제로 좁혀졌다.
- V12에서 다룰 문제가 `gripper close 안정화`와 `EE pitch/vertical grasp 활용`으로 명확해졌다.

## 최종 모델

학습 run:

```text
v11-initial-lift-close-1200eps-15000steps-b8ga2
```

최종 adapter:

```text
openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
```

최종 run directory:

```text
openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt
```

## 데이터셋

Raw episode:

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200
```

V11 intermediate:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
```

TFDS:

```text
tensorflow_datasets/raccoon_pick_place/1.0.0
```

주요 통계:

| 항목 | 값 |
|---|---:|
| raw episodes | 1200 |
| train / val | 1080 / 120 |
| 색상 분포 | red/blue/green/yellow 각 300 |
| instruction 종류 | 60 |
| converted step 수 | min 46 / avg 47.975 / max 49 |
| first close index 평균 | 약 25.0 |
| close ratio | 약 0.478 |

## V11 데이터 라벨링 핵심

V11은 raw trajectory를 새로 만들기보다 converter에서 close/lift supervision을 강화했다.

| 설정 | 값 | 목적 |
|---|---:|---|
| `promote_pre_close_steps` | 3 | 첫 close 직전 open frame을 close label로 승격 |
| `initial_close_min_z_action` | 0.004 | close 직후 lift 방향 z-action 보장 |
| `drop_post_close_hold_steps` | 2 | close 이후 정지성 transition 축소 |
| `drop_closed_gripper_small_z_actions` | true | closed 상태에서 z-action이 작은 transition 제거 |
| `closed_gripper_min_z_action` | 0.002 | closed 상태 lift supervision 최소값 |

이 라벨링의 의도는 V10/V11 이전 실패에서 보였던 `no-close`, `late-close`, `close 이후 lift action 약화`를 줄이는 것이다.

## 학습 설정

최종 학습은 batch size 8, grad accumulation 2로 실행했다. effective batch는 16이다.

| 항목 | 값 |
|---|---:|
| max steps | 15000 |
| save steps | 2500 |
| batch size | 8 |
| grad accumulation | 2 |
| effective batch | 16 |
| learning rate | 5e-4 |
| LoRA rank | 32 |
| LoRA dropout | 0.0 |
| image aug | true |
| shuffle buffer size | 100000 |

남아 있는 checkpoint:

```text
2500, 5000, 7500, 10000, 12500, 15000
```

## Checkpoint Scan

각 checkpoint를 8회 rollout으로 먼저 비교했다.

| checkpoint step | final target lift | wrong-color touch | mean final lift delta |
|---:|---:|---:|---:|
| 2500 | 7/8 (87.5%) | 1/8 (12.5%) | 0.0114 m |
| 5000 | 6/8 (75.0%) | 1/8 (12.5%) | 0.0132 m |
| 7500 | 6/8 (75.0%) | 0/8 (0.0%) | 0.0093 m |
| 10000 | 6/8 (75.0%) | 1/8 (12.5%) | 0.0115 m |
| 12500 | 7/8 (87.5%) | 0/8 (0.0%) | 0.0107 m |
| 15000 | 8/8 (100.0%) | 0/8 (0.0%) | 0.0123 m |

8회 비교에서는 15000 step이 가장 좋았고, 이를 최종 후보로 선택했다.

## 100 Rollout 최종 평가

평가 기준:

- 성공: 목표 실린더가 `lift_threshold=0.01m` 이상 들어올려짐
- wrong-color touch: 성공/실패 판정에는 넣지 않고 별도 품질 지표로 기록
- exception: rollout 중 실행 오류

최종 결과:

| 항목 | 결과 |
|---|---:|
| final target lift success | 80/100 (80.0%) |
| ever contact success | 82/100 (82.0%) |
| wrong-color touch | 9/100 (9.0%) |
| exception | 0/100 |
| mean final lift delta | 0.010306 m |
| mean steps | 40.63 |

색상별 결과:

| color | final target lift | wrong-color touch | mean final lift delta |
|---|---:|---:|---:|
| blue | 19/25 (76.0%) | 3/25 (12.0%) | 0.010140 m |
| green | 19/25 (76.0%) | 2/25 (8.0%) | 0.009776 m |
| red | 19/25 (76.0%) | 3/25 (12.0%) | 0.009870 m |
| yellow | 23/25 (92.0%) | 1/25 (4.0%) | 0.011437 m |

## 실패 분석

100회 rollout 중 목표 lift 실패는 20개였다. wrong-color touch가 있었더라도 목표 실린더를 lift한 경우는 성공으로 처리했다.

| 분류 | episode 수 | 해석 |
|---|---:|---|
| 목표 lift 성공 | 80 | 최종 성공 |
| no contact 또는 no close | 14 | 타겟 근처까지 갔지만 close/contact가 발생하지 않음 |
| contact but lift fail | 2 | 접촉은 했지만 lift threshold까지 못 올림 |
| wrong touch + target lift fail | 4 | 간섭과 target lift 실패가 함께 발생 |

추가 진단:

- 목표 lift 실패 20개 중 `16개`는 gripper close 명령이 한 번도 나오지 않았다.
- 그 16개는 모두 target XY 거리 `15mm` 이내까지 접근한 케이스였다.
- 따라서 V11의 주요 병목은 목표 탐색이나 색상 이해가 아니라, 마지막 grasp trigger와 close/lift transition이다.
- 일부 hard scene에서는 가까운 실린더가 EE 박스/링크 구조와 간섭하여 그리퍼가 충분히 내려가지 못하는 패턴이 관찰됐다.

## 시각화 산출물

보고서용 그림:

```text
reports/v11_final_baseline_assets/v11_checkpoint_scan_rollout8.png
reports/v11_final_baseline_assets/v11_rollout100_by_color.png
reports/v11_final_baseline_assets/v11_rollout100_outcome_categories.png
```

episode별 분류 CSV:

```text
reports/v11_final_baseline_assets/v11_rollout100_episode_classification.csv
```

대표 성공 GIF:

```text
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0001_yellow.gif
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0002_red.gif
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0003_green.gif
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0004_blue.gif
```

실패 GIF:

```text
reports/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs
```

위 폴더에는 최종 성공 기준에서 실패한 20개 GIF만 남겨두었다. PNG frame은 남기지 않았다.

## V11의 의미

V11은 본 프로젝트에서 다음 기준을 만족하는 기본 모델이다.

1. 4색 실린더에 대한 자연어 grounding이 가능하다.
2. 목표 실린더를 단순히 터치하는 것이 아니라 lift까지 수행한다.
3. rollout 평가와 GIF 시각화를 통해 성공/실패 원인을 추적할 수 있다.
4. 실패 원인이 close trigger와 기구 간섭으로 좁혀져 다음 실험 방향을 제시한다.

## V12로 넘길 문제

V12에서는 V11을 기준 모델로 두고, 아래 두 가지를 추가 개선한다.

1. close trigger 강화
   - target 근처 open transition 제거 또는 downweight
   - close 직전 frame 추가 승격
   - close/lift transition oversampling

2. EE pitch 기반 vertical grasp 추가
   - 현재 V11 action execution은 `dx, dy, dz, gripper`만 사용한다.
   - OpenVLA action의 `dpitch`를 gripper orientation mode로 살려야 한다.
   - 간섭 가능 배치에서는 expert trajectory가 `lockv()` vertical mode를 사용하도록 raw episode를 생성한다.
   - rollout에서는 외부 규칙으로 강제 vertical mode를 넣지 않고, VLA가 출력한 pitch/mode action만 실행해야 한다.

V12의 목표는 V11에서 이미 확보한 target approach 능력을 유지하면서, close trigger와 vertical grasp strategy를 추가로 학습시키는 것이다.
