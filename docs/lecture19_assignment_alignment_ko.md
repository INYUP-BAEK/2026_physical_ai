# Lecture19 과제 요구사항 대응 정리

분석 대상: `/data/biy/client/Lecture19_VLA_Lab1.pdf`

## PDF 기준 원본 파이프라인

Lecture19는 RaccoonBot + OpenVLA 실습을 다음 흐름으로 설명한다.

1. MuJoCo에서 4색 실린더 grasp demonstration 생성
2. raw episode를 RLDS intermediate로 변환
3. TFDS `raccoon_pick_place` 빌드
4. OpenVLA LoRA fine-tuning
5. OpenVLA server를 원격 서버에서 실행
6. 로컬 client가 MuJoCo image와 instruction을 `/predict`로 보내고 action을 받아 실행

기본 과제 task는 color-conditioned grasping이며, 원본 instruction은 `"grasp the {color} cylinder"` 단일 템플릿이다.

## 과제 요구사항과 현재 수정 방향

PDF 36p의 Dataset Extension 요구사항:

- task type을 grasp only에서 확장
- language instruction 다양화
- 새 MuJoCo demonstration 생성
- RLDS / TFDS 재빌드
- episode 시각화 및 짧은 LoRA test 실행

현재 프로젝트는 이 요구사항과 일치한다.

- grasp only에서 `grasp + lift` 방향으로 확장했다.
- 색상별 instruction template을 확장했다.
- 4색 전체 target을 균형 있게 포함하는 1200 episode raw dataset을 생성했다.
- RLDS intermediate와 TFDS를 재빌드했다.
- rollout markdown, CSV, GIF/프레임 저장 방식으로 시각화와 로그를 남기도록 했다.

PDF 37p의 Code Improvement 요구사항:

- 7D-to-4DOF action mapping 개선
- inference 또는 motion execution 안정화/속도 개선
- timing/action log 또는 visualization 개선
- before/after evidence와 VLA pipeline 영향 논의

현재 프로젝트는 이 요구사항과도 일치한다.

- OpenVLA 7D action 중 xyz + gripper를 4DOF RaccoonBot에 맞게 실행한다.
- `command_delta` 변환, post-close/lift transition 분석, V11 close-stable relabeling을 적용했다.
- client/server rollout에서 VLA-only 여부를 확인할 수 있도록 보조 controller flag를 명시적으로 분리했다.
- diagnostics markdown, summary JSON/CSV, rollout frame/GIF 저장으로 before/after evidence를 남겼다.

## 최종 방향의 적합성

현재 V11 방향은 과제의 단순 구현을 넘어서 실패 원인을 분석해 학습 supervision을 수정한 형태다. 특히 V10 평가에서 lift 자체보다 `no-close` 및 late-close가 주요 실패였으므로, V11에서는 close 직전 frame을 close command로 승격해 close timing 학습 신호를 강화했다.

이 방향은 과제의 “더 reliable한 VLA pipeline” 요구와 직접 대응된다.

주의할 점:

- 과제 제출 repository에는 대용량 raw dataset, TFDS, model checkpoint를 올리지 않는 것이 PDF 38p 요구와 맞다.
- 대신 code, README, logs, screenshots/visualization, report를 남겨야 한다.
- 성능 검증 시 client-side 보조 controller를 켜면 VLA-only 결과가 아니므로, 기본 평가는 보조 controller flag 없이 수행해야 한다.

## PDF 방식 client 실행 대응

PDF 31~32p는 client 실행을 다음 구조로 설명한다.

1. 서버에서 OpenVLA inference server 실행
2. 로컬 client PC에서 SSH tunnel 열기
3. client는 `http://127.0.0.1:8000`으로 접속
4. `openvla_multicolor_client.py` 실행

현재 `/data/biy/client`는 이 방식에 맞게 정리했다.

- `assets/Link2.STL` 포함
- `Raccoon_colored_cylinder.xml`, `RaccoonBot_S.xml` 포함
- `openvla_multicolor_client.py`, `raccoon_env.py`, `requirements.txt` 포함
- `sshtunnel`은 `--use_ssh_tunnel` 사용 시에만 필요하도록 수정
- 기본 object y range와 z workspace를 V11 평가 조건에 맞춤
- 기본 `max_delta_xyz`를 서버 rollout 평가 조건과 맞춰 VLA action이 과도하게 잘리지 않도록 수정

