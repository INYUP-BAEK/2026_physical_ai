# Client 및 실제 RaccoonBot 실행 테스트 절차

작성일: 2026-06-13

## 1. 목적

이 문서는 서버에서 OpenVLA inference를 수행하고, 클라이언트에서 MuJoCo 시각화 및 실제 RaccoonBot 동작을 실행하는 테스트 절차를 정리한다. 현재 GPU가 학습에 사용 중이므로 이 문서는 실행 결과가 아니라, 학습 완료 후 그대로 따라갈 수 있는 순서와 방법을 정리한 것이다.

기본 흐름:

```text
client MuJoCo image
-> HTTP /predict
-> server OpenVLA action
-> client MuJoCo execute_delta_action7
-> 같은 target xyz/gripper 명령을 실제 RaccoonBot IK로 변환
-> 실제 로봇 동작
```

이 구조에서 OpenVLA 연산은 서버 GPU에서 수행하고, MuJoCo viewer와 실제 로봇 제어는 client 쪽에서 수행한다.

## 2. 현재 client 코드 상태

Client 경로:

```text
/data/biy/client
```

주요 파일:

| 파일 | 역할 |
|---|---|
| `openvla_multicolor_client.py` | MuJoCo client rollout |
| `openvla_multicolor_client_real_robot.py` | MuJoCo rollout과 실제 RaccoonBot 동시 실행 |
| `batch_eval_openvla_client.py` | 여러 색상/seed batch 평가 |
| `raccoon_env.py` | MuJoCo RaccoonBot 환경 및 7D action 실행 |
| `Raccoon_colored_cylinder.xml` | 4색 cylinder scene |
| `requirements.txt` | client dependency |

실제 로봇 실행은 `openvla_multicolor_client_real_robot.py`에서만 켠다. 기본값은 MuJoCo만 실행이며, `--use_real_robot`를 줘야 실제 RaccoonBot으로 명령이 전송된다.

## 3. 서버 실행

### 3.1 V11 lift 모델 서버

GPU가 비어 있을 때 서버에서 실행한다.

```bash
cd /data/biy/Raccoonbot_Openvla

ADAPTER_PATH="/data/biy/Raccoonbot_Openvla/openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt" \
HOST=0.0.0.0 \
PORT=8000 \
DEVICE=cuda \
MERGE_ADAPTER=0 \
scripts/10_start_openvla_server.sh
```

### 3.2 V11+Stack 20000 모델 서버

```bash
cd /data/biy/Raccoonbot_Openvla

ADAPTER_PATH="/data/biy/Raccoonbot_Openvla/openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt" \
HOST=0.0.0.0 \
PORT=8000 \
DEVICE=cuda \
MERGE_ADAPTER=0 \
scripts/10_start_openvla_server.sh
```

서버 script는 `ADAPTER_PATH`가 주어지면 같은 이름의 `openvla-runs` 디렉토리에서 processor/statistics를 찾는다. `MERGE_ADAPTER=0`이면 adapter를 병합하지 않고 PEFT adapter 상태로 추론한다.

서버 확인:

```bash
curl http://SERVER_IP:8000/health
```

정상 응답:

```json
{"ok":true}
```

## 4. SSH tunnel 연결

클라이언트 PC에서 서버 포트가 직접 열려 있지 않으면 SSH local forwarding을 사용한다.

```bash
ssh -L 8000:127.0.0.1:8000 root@qlak315.iptime.org -p 23000
```

터널 터미널은 닫지 않는다. 다른 터미널에서 확인한다.

```bash
curl http://127.0.0.1:8000/health
```

정상 응답이 나오면 client는 `--server_url http://127.0.0.1:8000`으로 접속하면 된다.

Client script의 `--use_ssh_tunnel` 옵션을 사용할 수도 있지만, 보고서 재현성 측면에서는 수동 SSH tunnel을 먼저 열고 `--server_url`을 명시하는 방식이 더 단순하다.

## 5. Client 환경 설치

클라이언트 PC 또는 client 실행 환경에서:

```bash
cd /data/biy/client
pip install -r requirements.txt
```

MuJoCo viewer를 띄울 경우 화면 표시가 가능한 환경에서 실행해야 한다. headless 환경이면 `--use_viewer`를 빼고 frame 저장만 사용한다.

## 6. 1단계: 서버-클라이언트 연결 smoke test

실제 로봇을 연결하기 전에 반드시 MuJoCo only로 먼저 확인한다.

```bash
cd /data/biy/client

python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the red cylinder" \
  --target_color red \
  --max_steps 5 \
  --max_delta_xyz 0.12 \
  --settle_seconds_per_action 0.1 \
  --initial_settle_seconds 0.1 \
  --output_dir rollout_outputs/server_client_smoke \
  --no_save_images
```

확인할 것:

- 서버 로그에 `[PREDICT] instruction=...`가 출력되는지
- client가 action을 받아 MuJoCo에서 `OK` step을 찍는지
- `/health`는 되는데 `/predict`가 실패하면 server model/statistics 경로를 확인한다.

## 7. 2단계: MuJoCo viewer로 VLA-only 동작 확인

VLA-only 검증에서는 보조 controller flag를 켜지 않는다. 아래 옵션만 사용한다.

```bash
cd /data/biy/client

python openvla_multicolor_client.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the blue cylinder" \
  --target_color blue \
  --seed 1001 \
  --max_steps 50 \
  --max_delta_xyz 0.12 \
  --settle_seconds_per_action 0.1 \
  --initial_settle_seconds 0.1 \
  --stop_on_lift_success \
  --output_dir rollout_outputs/vla_only_lift_blue \
  --use_viewer
```

VLA-only 평가에서 켜면 안 되는 옵션:

```text
--assist_xy_alignment_before_close
--gate_close_by_xy
--preclose_min_z_when_xy_bad
--latch_gripper_after_close
--post_close_clamp_negative_z
--post_close_lift_controller
--post_close_keep_xy
--use_action_smoothing
```

이 옵션들은 성능 개선/디버깅용 client-side 보조 제어다. 과제 평가나 "only VLA" 결과에는 사용하지 않는다.

## 8. 3단계: 실제 로봇 없이 하드웨어 script dry run

실제 로봇 script가 서버/MuJoCo까지 정상 동작하는지 먼저 확인한다. 이때 `--use_real_robot`는 주지 않는다.

```bash
cd /data/biy/client

python openvla_multicolor_client_real_robot.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the yellow cylinder" \
  --target_color yellow \
  --seed 1002 \
  --max_steps 10 \
  --max_delta_xyz 0.12 \
  --settle_seconds_per_action 0.8 \
  --initial_settle_seconds 0.3 \
  --output_dir rollout_outputs/real_script_dry_run \
  --use_viewer
```

이 단계는 실제 로봇 API를 호출하지 않는다. MuJoCo viewer에서 동작이 정상인지, 서버 action이 정상적으로 내려오는지만 확인한다.

## 9. 4단계: 실제 로봇 연결 확인

실제 RaccoonBot 실행 전 체크리스트:

- 로봇 전원 연결
- USB/Bluetooth 연결 확인
- `roboid` 패키지 import 가능 여부 확인
- 로봇 주변 작업공간 비우기
- 실린더 위치가 MuJoCo scene과 최대한 비슷한지 확인
- 손으로 비상 정지할 수 있는 상태 유지

간단한 import 확인:

```bash
cd /data/biy/client

python - <<'PY'
from roboid import Raccoon
r = Raccoon()
print("ready:", bool(getattr(getattr(r, "_roboid", None), "_ready", False)))
PY
```

`ready: True`가 아니면 실제 실행으로 넘어가지 않는다.

## 10. 5단계: 실제 로봇 동시 실행

실제 로봇까지 같은 action을 따라 하게 하려면 `openvla_multicolor_client_real_robot.py`에 `--use_real_robot`를 켠다.

처음에는 step 수를 작게 잡는다.

```bash
cd /data/biy/client

python openvla_multicolor_client_real_robot.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the red cylinder" \
  --target_color red \
  --seed 1003 \
  --max_steps 8 \
  --max_delta_xyz 0.06 \
  --speed 50 \
  --settle_seconds_per_action 0.8 \
  --real_settle_seconds 0.8 \
  --initial_settle_seconds 0.3 \
  --real_initial_wait_seconds 5.0 \
  --real_go_home_on_exit \
  --output_dir rollout_outputs/real_robot_red_step8 \
  --use_viewer \
  --use_real_robot
```

문제가 없으면 step 수와 delta 제한을 늘린다.

```bash
cd /data/biy/client

python openvla_multicolor_client_real_robot.py \
  --server_url http://127.0.0.1:8000 \
  --xml_path Raccoon_colored_cylinder.xml \
  --instruction "grasp and lift the red cylinder" \
  --target_color red \
  --seed 1003 \
  --max_steps 50 \
  --max_delta_xyz 0.12 \
  --speed 70 \
  --settle_seconds_per_action 0.8 \
  --real_settle_seconds 0.8 \
  --initial_settle_seconds 0.3 \
  --real_initial_wait_seconds 5.0 \
  --real_go_home_on_exit \
  --output_dir rollout_outputs/real_robot_red_full \
  --use_viewer \
  --use_real_robot
```

실행 중 script는 다음 순서로 동작한다.

1. MuJoCo scene을 reset한다.
2. observation image를 서버로 보낸다.
3. 서버에서 7D action을 받는다.
4. MuJoCo `execute_delta_action7()`로 workspace clipping, IK, target xyz를 계산한다.
5. 실제 로봇 controller가 `exec_info["target_xyz"]`를 cm 단위로 변환한다.
6. 실제 RaccoonBot IK로 4개 joint angle을 계산한다.
7. joint angle과 gripper open/close 명령을 실제 로봇에 전송한다.

## 11. 권장 테스트 순서

실제 제출/시연 전에는 다음 순서가 가장 안전하다.

1. 서버만 실행하고 `/health` 확인
2. `openvla_multicolor_client.py`로 MuJoCo only 5 step smoke
3. `openvla_multicolor_client.py`로 MuJoCo viewer 50 step VLA-only 확인
4. `openvla_multicolor_client_real_robot.py`를 `--use_real_robot` 없이 dry run
5. 실제 로봇 연결 확인
6. `--use_real_robot`, `max_steps=8`, `max_delta_xyz=0.06`, `speed=50`으로 짧은 안전 테스트
7. `max_steps=50`, `max_delta_xyz=0.12`, `speed=70`으로 전체 테스트

## 12. Stack 모델 client 테스트 주의

현재 `/data/biy/client/openvla_multicolor_client.py`와 `openvla_multicolor_client_real_robot.py`는 기본적으로 단일 target color lift/grasp command에 맞춰져 있다.

특히 instruction에서 색상이 두 개 이상 발견되면 target color 동기화 로직이 에러를 낸다.

예:

```text
stack the red cylinder on the blue cylinder
```

이 명령은 `red`, `blue` 두 색상이 있으므로 현재 client의 `resolve_target_color_and_instruction()` 구조와 맞지 않는다. 따라서 stack 모델은 현재 상태에서 다음 방식으로 다룬다.

- 서버 측 rollout/evaluation script로 stack 성능을 검증한다.
- 실제 client stack 시연이 필요하면 client에 `--source_color`, `--base_color`를 추가하고, 두 색상 배치를 명시적으로 고정하는 확장이 필요하다.
- 실제 로봇 stack 시연은 lift보다 위험도가 높으므로, MuJoCo stack client가 먼저 안정화된 뒤 실로봇으로 넘기는 것이 맞다.

즉, 현재 실로봇 동시 실행 문서는 V11 lift 모델을 1차 시연 대상으로 둔다. V11+Stack은 창의성 확장 결과로 보고서에 포함하되, 실로봇 즉시 시연 대상은 아니다.

## 13. 문제 발생 시 확인 포인트

서버 연결 실패:

```bash
curl http://127.0.0.1:8000/health
```

- 응답이 없으면 SSH tunnel 또는 server process를 확인한다.
- `/health`는 되지만 `/predict`가 실패하면 model path, adapter path, dataset statistics를 확인한다.

실제 로봇 연결 실패:

- `roboid` import 실패 여부 확인
- RaccoonBot 전원/연결 확인
- `ready: True` 확인
- `--allow_sim_only_on_hw_fail`은 디버깅용이며, 실제 시연 성공으로 기록하지 않는다.

동작이 너무 빠르거나 위험할 때:

- `--speed 50`
- `--max_delta_xyz 0.04~0.06`
- `--real_settle_seconds 1.0`
- `--max_steps 5~8`

위 설정으로 줄여서 짧게 재확인한다.

## 14. 보고서에 적을 핵심 문장

실제 로봇 확장 구조는 VLA가 직접 출력한 7D action을 client에서 임의로 성공 방향으로 보정하는 방식이 아니다. 서버가 예측한 action을 MuJoCo 실행기에서 동일한 4DOF RaccoonBot workspace target으로 해석하고, 그 target xyz와 gripper command를 실제 RaccoonBot IK로 변환해 전송한다. 따라서 MuJoCo와 실제 로봇은 같은 high-level VLA action 흐름을 공유한다.
