# V11 최종 학습 전 프로젝트 구조 정리

작성일: 2026-06-11

## 최종 목표

마지막 학습은 V10에서 확인된 `no-close` 실패를 줄이는 데 집중한다. 기존 V10 raw 1200 episode는 물리적으로 모두 성공한 데이터였으므로 새 raw episode를 만들지 않고, converter 단계에서 close 직전 frame을 close label로 승격한 V11 데이터셋을 최종 학습 입력으로 사용한다.

## 최종 유지 파일

### Raw episode

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200
```

- V10 immediate-lift trajectory로 생성된 성공 raw 1200 episode
- V11은 이 raw를 재사용해 close-stable label로 변환
- raw 물리 궤적은 유지하고, 학습 label만 안정화

### V11 intermediate

```text
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta
```

검증 결과:

| 항목 | 값 |
| --- | ---: |
| episodes | 1200 |
| train | 1080 |
| val | 120 |
| 색상별 총 episode | red/green/blue/yellow 각 300 |
| promoted pre-close frames | 3600 |
| promote 설정 | 3 frames/episode |

### TFDS

```text
tensorflow_datasets/raccoon_pick_place/1.0.0
```

검증 결과:

| split | examples |
| --- | ---: |
| train | 1080 |
| val | 120 |

### 최종 실행 스크립트

최종 실행은 `v11` 이름의 스크립트를 사용한다.

```text
scripts/06_convert_v11_close_stable.sh
scripts/07_build_tfds_v11.sh
scripts/08_train_lora_v11.sh
scripts/09_eval_v11_rollout.py
```

기존 `v9` 이름 스크립트는 과거 명령과의 호환을 위해 남겨두었지만, 최종 학습/평가 명령에서는 사용하지 않는다.

## 핵심 코드 수정

파일:

```text
Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py
```

추가 기능:

```text
--promote_pre_close_steps
```

역할:

- 첫 raw close 직전 N개 open-gripper frame을 close command로 relabel
- V11 기본값은 3
- promoted frame은 idle/small-z filter로 삭제되지 않도록 보호
- `raw_waypoint_action` debug 값도 promoted frame에서는 gripper=1로 기록

의도:

- V10 rollout 실패 대부분은 lift 실패가 아니라 close command 미발생 또는 지연이었다.
- close 직전 frame들은 이미 EE가 충분히 낮은 위치에 있으므로, 이 구간을 close label로 바꾸면 close 전환 학습 신호가 강해진다.

## 정리한 이전 파일

삭제한 대용량/이전 버전 산출물:

```text
Mujoco/raccoon_grasp_v9_deep_lift_1200
Mujoco/raccoon_grasp_v10_lift_immediate_smoke40
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v9_deep_lift_1200_fk_command_delta
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_1200_fk_command_delta
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v10_lift_immediate_smoke40_fk_command_delta
openvla/openvla-adapter-tmp/*v9*
openvla/openvla-adapter-tmp/*v10*
openvla/openvla-runs/*v9*
openvla/openvla-runs/*v10*
```

삭제한 보조 산출물:

```text
reports/v9_*
reports/v10_lift_immediate_smoke40_5000_rollout8
logs/v9_*.log
logs/v10_*.log
diagnostics/v9_*
diagnostics/v10_lift_immediate_smoke40_*
```

남긴 비교 근거:

```text
diagnostics/v10_lift_immediate_1200_*rollout*
reports/v10_lift_immediate_1200_10000_gif4
docs/archive
```

V10 10000/20000/30000 비교와 10000 checkpoint 32 rollout 분석은 최종 보고서의 baseline 근거로 남겼다.

## 현재 용량 상태

정리 후 프로젝트 크기:

```text
약 4.4G
```

주요 용량:

```text
Mujoco/raccoon_grasp_v10_lift_immediate_1200: 약 1.5G
Mujoco/raccoon_dataset/openvla_rlds_intermediate_v11_close_stable_1200_fk_command_delta: 약 1.6G
tensorflow_datasets: 약 1.4G
```

기존 프로젝트 크기는 약 12G였고, 이전 adapter/raw/intermediate 제거로 약 7G 이상을 줄였다.

## 마지막 학습 명령

GPU가 비면 아래 명령으로 최종 학습을 실행한다.

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

V10에서 10000 step 이후 성능이 떨어졌기 때문에 30000 step까지 밀지 않는다. 2500 step 간격 저장으로 5000, 7500, 10000, 12500, 15000 checkpoint를 비교한다.

## 학습 후 평가 명령

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

최종 후보는 32 rollout으로 확장 검증한다.
