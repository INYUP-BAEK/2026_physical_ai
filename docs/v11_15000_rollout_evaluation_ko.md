# V11 15000 Step Rollout 평가 요약

## 평가 대상

- 학습 run: `v11-initial-lift-close-1200eps-15000steps-b8ga2`
- 최종 adapter: `openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt`
- 평가 스크립트: `scripts/09_eval_v11_rollout.py`
- 평가 조건: closed-loop MuJoCo rollout, `max_steps=50`, `max_delta_xyz=0.12`, `workspace_z_min=0.016`, `lift_threshold=0.01`

## 체크포인트별 8 Rollout 비교

| checkpoint step | final strict lift | wrong-color touch | mean final lift delta |
|---:|---:|---:|---:|
| 2500 | 7/8 (87.5%) | 1/8 (12.5%) | 0.0114 m |
| 5000 | 6/8 (75.0%) | 1/8 (12.5%) | 0.0132 m |
| 7500 | 6/8 (75.0%) | 0/8 (0.0%) | 0.0093 m |
| 10000 | 6/8 (75.0%) | 1/8 (12.5%) | 0.0115 m |
| 12500 | 7/8 (87.5%) | 0/8 (0.0%) | 0.0107 m |
| 15000 | 8/8 (100.0%) | 0/8 (0.0%) | 0.0123 m |

8 rollout 기준으로는 15000 step checkpoint가 가장 좋았다. 특히 12500 step에서 반복적으로 실패하던 scene 2 yellow 케이스가 15000 step에서는 성공했기 때문에 최종 후보로 15000 step을 선택했다.

## 15000 Step 32 Rollout 확장 평가

- final strict lift success: `25/32` (78.125%)
- ever strict lift success: `25/32` (78.125%)
- ever contact success: `26/32` (81.25%)
- wrong-color touch: `2/32` (6.25%)
- exception: `0/32`
- mean final lift delta: `0.010014 m`
- mean steps: `41.03`

색상별 결과:

| color | final strict lift | wrong-color touch | mean final lift delta |
|---|---:|---:|---:|
| blue | 6/8 (75.0%) | 1/8 (12.5%) | 0.0094 m |
| green | 6/8 (75.0%) | 0/8 (0.0%) | 0.0102 m |
| red | 5/8 (62.5%) | 1/8 (12.5%) | 0.0078 m |
| yellow | 8/8 (100.0%) | 0/8 (0.0%) | 0.0126 m |

## 해석

이번 V11 15000 step 모델은 단순 접촉을 넘어 실제 lift까지 수행하는 closed-loop 성능이 확인되었다. 32 rollout 기준 strict lift 성공률은 78.125%이며, wrong-color touch는 6.25%로 낮다. 따라서 현재 모델은 "색상 지시에 따라 목표 실린더를 선택하고 들어올리는 VLA 정책"으로서 이전 grasp 중심 모델보다 명확히 개선되었다.

다만 실패는 특정 scene에서 묶여 나타나는 경향이 있었다. 대표적으로 scene 4와 scene 7에서 여러 색상이 연속 실패했고, 대부분 wrong-touch 없이 접촉 또는 접근 자체가 실패했다. 이는 색상 언어 이해보다는 초기 배치, 접근 경로, 작업공간 한계, 하강-폐쇄 타이밍의 일반화 문제가 남아 있음을 의미한다.

## 100 Rollout 추가 평가

V11을 기본 모델 최종본으로 고정하기 위해 100 rollout을 추가로 수행했다. 이때 성공 기준은 `목표 실린더가 lift_threshold 이상 들어올려졌는가`로 두고, 다른 실린더 touch는 실패가 아니라 별도의 간섭 품질 지표로 기록했다.

결과:

| 항목 | 값 |
|---|---:|
| final target lift success | 80/100 (80.0%) |
| wrong-color touch | 9/100 (9.0%) |
| exception | 0/100 |
| mean final lift delta | 0.010306 m |

색상별 final target lift:

| color | 성공률 |
|---|---:|
| blue | 19/25 (76.0%) |
| green | 19/25 (76.0%) |
| red | 19/25 (76.0%) |
| yellow | 23/25 (92.0%) |

실패 GIF는 최종 성공 기준에서 실패한 20개만 남겼고, PNG frame은 삭제했다.

## 산출물

- 8 rollout 체크포인트 비교 결과:
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_2500_rollout8.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_5000_rollout8.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_7500_rollout8.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_10000_rollout8.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_12500_rollout8.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout8.md`
- 15000 step 32 rollout 확장 평가:
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout32.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout32.csv`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout32.json`
- 15000 step GIF 시각화:
  - `reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0001_yellow.gif`
  - `reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0002_red.gif`
  - `reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0003_green.gif`
  - `reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/v11_initial_lift_close_1200_b8ga2_15000_gif4_episode_0004_blue.gif`
- 15000 step 100 rollout 평가:
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs.md`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs.csv`
  - `diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs.json`
- 100 rollout 실패 GIF:
  - `reports/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs`
- 최종 baseline 보고서와 그림:
  - `docs/v11_final_baseline_report_ko.md`
  - `reports/v11_final_baseline_assets`

## 결론

최종 후보는 15000 step checkpoint로 보는 것이 타당하다. 짧은 8 rollout에서는 100% 성공, 확장 32 rollout에서는 78.125%, 100 rollout에서는 80.0% target lift 성공을 보였다. 실패가 색상 전반에 무작위로 퍼진 것이 아니라 특정 배치와 close trigger 문제에 집중되므로, 다음 개선 여지는 close transition 강화와 EE pitch 기반 vertical grasp 추가에 있다.
