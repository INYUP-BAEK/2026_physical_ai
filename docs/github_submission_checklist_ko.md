# GitHub 제출 전 체크리스트

작성일: 2026-06-13

## 과제 요구사항 확인

| 요구사항 | 현재 상태 | 위치 |
|---|---|---|
| GitHub repository URL 제출 | 로컬 정리 완료, GitHub 업로드 전 상태 | `/data/biy/Raccoonbot_Openvla` |
| `README.md` 포함 | 완료 | `README.md` |
| modified code 포함 | 완료 | `Mujoco/`, `scripts/`, `openvla/`, `dlimp_openvla/` |
| logs 포함 | 완료 | `logs/` |
| screenshots / episode visualizations 포함 | 완료 | `reports/` |
| short report 포함 | 완료 | `report.pdf`, `report.md` |
| 변경점 설명 | 완료 | `README.md`, `docs/final_ref_to_v11_lift_stack_report_ko.md` |
| 실행 방법 설명 | 완료 | `README.md`, `docs/client_real_robot_test_plan_ko.md` |
| 결과 설명 | 완료 | `README.md`, `report.pdf`, `docs/final_ref_to_v11_lift_stack_report_ko.md` |
| large dataset 미업로드 | `.gitignore` 반영 완료 | `.gitignore` |
| model checkpoint 미업로드 | `.gitignore` 반영 완료 | `.gitignore` |

## `.gitignore` 검증 결과

다음 대용량 항목이 ignore되는 것을 확인했다.

```text
tensorflow_datasets/
Mujoco/raccoon_grasp_v*/
Mujoco/raccoon_stack_v*/
Mujoco/raccoon_dataset/openvla_rlds_intermediate*/
openvla/openvla-adapter-tmp/
openvla/openvla-runs/
.hf_cache/
*.safetensors
*.bin
*.pt
*.ckpt
```

추가로 `find . -type f -size +10M` 기준, ignore되지 않은 10MB 초과 파일은 없었다.

## GitHub에 올려야 하는 핵심 파일/폴더

```text
README.md
.gitignore
report.pdf
report.md
docs/
scripts/
Mujoco/*.py
Mujoco/*.xml
Mujoco/assets/
Mujoco/rlds_dataset_builder/
openvla/openvla_server.py
openvla/vla-scripts/finetune.py
openvla/prismatic/
dlimp_openvla/
logs/
reports/
diagnostics/
```

## GitHub에 올리면 안 되는 항목

```text
tensorflow_datasets/
Mujoco/raccoon_grasp_v10_lift_immediate_1200/
Mujoco/raccoon_stack_v11_extension_120/
Mujoco/raccoon_grasp_v11_plus_stack_raw/
Mujoco/raccoon_dataset/openvla_rlds_intermediate*/
openvla/openvla-adapter-tmp/
openvla/openvla-runs/
.hf_cache/
```

## 업로드 전 권장 명령

현재 폴더가 아직 git repository가 아니라면:

```bash
cd /data/biy/Raccoonbot_Openvla
git init
git status --ignored
```

대용량 파일이 stage되지 않는지 확인:

```bash
git add .
git status --short
```

만약 `tensorflow_datasets`, `openvla-adapter-tmp`, `openvla-runs`, `Mujoco/raccoon_grasp_*` 같은 대용량 경로가 보이면 `git reset` 후 `.gitignore`를 다시 확인한다.

정상이라면:

```bash
git commit -m "Finalize RaccoonBot OpenVLA lift and stack project"
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```
