# V11 최종 학습 전 최적화 점검

작성 시점: final V11 학습 직전

## 점검 결론

최종 학습 입력은 기존 V10 raw 1200 episode를 유지하고, converter 단계에서 close/lift supervision을 더 명확하게 만든 V11 데이터셋이다. raw episode는 1200/1200 성공이고 4색이 300개씩 균형을 이룬다.

이번 최종 점검에서 추가로 수정한 핵심은 다음 두 가지다.

1. `promote_pre_close_steps=3`으로 close command를 앞당긴다.
2. `initial_close_min_z_action=0.004`로 promoted close frame과 첫 raw close frame의 z-action을 즉시 lift 방향으로 만든다.

이 수정은 V10/V11 이전 평가에서 주로 보였던 `no-close`, `late-close`, `close 이후 lift 신호 약화`를 줄이기 위한 것이다.

## 최종 데이터 통계

경로:

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
tensorflow_datasets/raccoon_pick_place/1.0.0
```

주요 값:

| 항목 | 값 |
| --- | --- |
| raw episode | 1200 |
| train / val | 1080 / 120 |
| 색상 분포 | red/blue/green/yellow 각 300 |
| instruction 종류 | 60 |
| converted step 수 | min 46 / avg 47.975 / max 49 |
| first close index 평균 | 약 25.0 |
| close ratio | 약 0.478 |
| `promote_pre_close_steps` | 3 |
| `initial_close_min_z_action` | 0.004 |

초기 close 구간 검증:

| 구간 | 결과 |
| --- | --- |
| first closed-frame z-action | min 0.004 / count < 0.002: 0/1200 |
| first 4 closed-frame z-action | min 0.004 / count < 0.002: 0/4800 |
| first 8 closed-frame z-action | min 0.004 / count < 0.002: 0/9600 |

즉, 첫 close 이후 초기 lift target이 데이터셋에 확실히 반영됐다.

## 코드 수정 요약

`Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py`

- `--initial_close_min_z_action` 옵션 추가
- promoted pre-close frame과 첫 raw close frame의 z-action을 최소 양수 lift action으로 보정
- dataset metadata에 `initial_close_min_z_action` 기록

`scripts/06_convert_v11_close_stable.sh`

- 기본값 `INITIAL_CLOSE_MIN_Z_ACTION=0.004` 추가

`openvla/vla-scripts/finetune.py`

- checkpoint 저장과 stop 조건을 micro-batch 기준이 아니라 optimizer step 기준으로 수정
- `grad_accumulation_steps > 1`에서 같은 step checkpoint가 반복 저장되는 문제 제거
- `MAX_STEPS`가 실제 optimizer update 수와 일치하도록 수정

`scripts/08_train_lora_v11.sh`

- `LORA_DROPOUT`, `IMAGE_AUG`, `SHUFFLE_BUFFER_SIZE`를 명시적으로 로그에 남기고 finetune에 전달
- 최종 학습 재현성을 높임

## 최종 학습 권장 세팅

V10에서는 10000 step이 20000/30000보다 나았고, V11은 label-targeted correction이므로 30000을 한 번에 보는 것보다 15000까지 저장점을 촘촘히 남기는 편이 더 안전하다.

권장:

```bash
cd /data/biy/Raccoonbot_Openvla

RUN_ID_NOTE=v11-initial-lift-close-1200eps-15000steps-b4ga4 \
MAX_STEPS=15000 \
SAVE_STEPS=2500 \
BATCH_SIZE=4 \
GRAD_ACCUMULATION_STEPS=4 \
LEARNING_RATE=5e-4 \
LORA_RANK=32 \
LORA_DROPOUT=0.0 \
IMAGE_AUG=true \
SHUFFLE_BUFFER_SIZE=100000 \
SAVE_LATEST_CHECKPOINT_ONLY=0 \
MERGE_LORA_CHECKPOINT=0 \
scripts/08_train_lora_v11.sh
```

이 설정은 2500, 5000, 7500, 10000, 12500, 15000 checkpoint를 남긴다. 최종 선택은 rollout 8개씩 빠르게 비교한 뒤, 가장 좋은 step을 32개 rollout으로 재검증한다.

