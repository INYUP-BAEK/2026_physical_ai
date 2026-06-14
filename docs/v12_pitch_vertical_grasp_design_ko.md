# V12 Pitch / Vertical Grasp 설계 점검

작성일: 2026-06-12

> 상태: 최종 학습 방향에서 제외됨.
> `docs/v12_pitch_review_and_v11_rollback_ko.md`에 최종 판단과 V11 롤백 근거를 정리했다.

## 목표

V11의 주요 실패는 목표 실린더 근처까지 접근한 뒤 close 명령이 나오지 않거나, 특정 배치에서 EE 박스/링크 구조가 앞쪽 실린더와 간섭하는 문제였다. V12에서는 4축 로봇의 EE pitch 자유도를 활용해, 같은 radial line에 가까운 실린더와 먼 실린더가 놓인 hard scene에서 먼 실린더 접근 시 gripper orientation을 바꾸는 전략을 학습시키고자 한다.

## 현재 반영한 코드 변경

### 1. Pitch action 활성화

OpenVLA action 형식은 기존과 동일하게 7D를 유지한다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

V12에서는 `dpitch`를 `pitch_alpha`로 해석한다.

```text
pitch_alpha = 0.0  -> horizontal grasp
pitch_alpha = 1.0  -> vertical grasp
```

수정 파일:

```text
Mujoco/raccoon_grasp_multicolor_scene_dataset.py
Mujoco/raccoon_dataset/convert_raw_to_openvla_rlds_intermediate.py
Mujoco/rlds_dataset_builder/raccoon_pick_place/raccoon_pick_place_dataset_builder.py
scripts/09_eval_v9_rollout.py
```

기본 v11 경로에서는 `include_pitch_action` / `use_pitch_action`을 켜지 않으면 기존처럼 rotation action이 0으로 유지된다.

### 2. 실제 로봇 적용을 고려한 판단

threshold 방식으로 `dpitch > threshold -> lockv()`만 적용하면 action 한 step에서 목표 orientation이 바뀐다. 다만 시뮬레이터의 joint4는 순간 이동하지 않고 속도 제한으로 따라가므로, 문제는 1프레임 명령 자체가 아니라 `orientation이 완전히 바뀌기 전에 하강/close가 이어지는 것`이다.

따라서 V12 raw trajectory에는 다음이 필요하다.

- safe height에서 pitch command를 먼저 준다.
- orientation settle frame을 충분히 둔다.
- 그 다음 하강한다.
- rollout에서도 `--pitch_settle_seconds_on_change`로 실제 로봇의 모터 이동 시간을 반영할 수 있게 한다.

### 3. 간섭 scene 생성

`--v12_interference_scenes` 옵션을 추가했다. 이 옵션은 같은 robot-radial line에 near/far cylinder pair를 배치한다.

의도:

- near cylinder target: horizontal grasp 유지
- far cylinder target: vertical/pitch-aware grasp label 사용

이를 통해 hard-coded controller가 아니라, 이미지와 자연어 지시를 보고 VLA가 필요한 target에서 pitch action을 출력하도록 학습시키는 구조를 만들 수 있다.

## 디버그 결과

작은 debug scene에서 같은 배치의 네 색을 시도했다.

```text
Mujoco/raccoon_grasp_v12_pitch_debug4
```

결과:

| target | pitch_alpha | 결과 |
|---|---:|---|
| yellow | 0.0 | 성공 |
| green | 0.0 | 성공 |
| red | 0.0 | 성공 |
| blue far target | 1.0 | 실패 |

blue는 green 뒤쪽에 있는 far target이며, vertical grasp가 필요한 케이스로 분류됐다.

실패 분석:

- 처음 구현 방향을 반대로 바꾸는 실험은 잘못된 방향이었다. vertical 방향은 기존 방향이 맞다.
- horizontal grasp에서는 gripper finger가 약 7.7~8cm 앞으로 튀어나온 구조이므로, 기존 command point convention은 이 전방 offset을 포함한다.
- vertical grasp에서는 실린더 좌표와 grasp 기준점이 일치해야 하므로 XY에 추가 offset을 주면 안 된다.
- full vertical 상태에서 기존 horizontal close 높이까지 내려가려 하면 gripper/finger가 바닥과 충돌하거나 관절/접촉 제약 때문에 EE가 충분히 내려가지 못한다.
- vertical close z를 `0.030m`로 올려도 full vertical blue far target은 lift에 실패했다.

즉 현재 XML/기구 구조 기준으로는 `full 90도 vertical close`를 그대로 학습 데이터에 넣기 어렵다. 이 상태로 40~120 episode smoke dataset을 만들면 far vertical target이 반복 실패하면서 scene rollback이 발생한다.

## 수정된 이해

초기 가정:

```text
간섭 scene에서는 gripper를 full vertical로 세워서 내려가면 성공할 것이다.
```

수정된 판단:

```text
full vertical orientation은 링크 박스 간섭은 줄일 수 있지만,
짧은 실린더를 잡는 close 높이에서는 gripper/finger와 바닥 충돌 문제가 커진다.
```

따라서 V12에서 바로 full vertical grasp를 학습시키기보다, 아래 중 하나로 방향을 정해야 한다.

## 권장 V12 방향

### A안: pitch-aware approach + horizontal close

가장 현실적인 방향이다.

1. safe height에서 pitch를 vertical 또는 semi-vertical로 바꾼다.
2. near blocker를 지나 far target 위로 접근한다.
3. close 직전에는 horizontal 또는 partial pitch로 되돌린다.
4. 기존 V11과 같은 안정적인 close/lift를 수행한다.

장점:

- V11의 검증된 horizontal close 성공률을 유지할 수 있다.
- pitch는 간섭 회피용으로만 쓰므로 ground collision 위험이 작다.
- 실제 로봇에서도 motor settle 시간을 넣기 쉽다.

### B안: partial pitch grasp

full vertical `pitch_alpha=1.0` 대신 `0.25~0.5` 범위의 partial pitch를 사용한다.

장점:

- 전방 박스 간섭을 일부 줄이면서 손가락이 바닥을 향해 완전히 내려가는 문제를 줄일 수 있다.

단점:

- 어떤 alpha가 실제 lift에 가장 좋은지 sweep이 필요하다.
- 단순 2-mode보다 라벨 분포가 복잡해진다.

### C안: full vertical grasp를 위한 EE collision/geometry 수정

실제 과제에서 full vertical grasp를 반드시 보여주려면 XML/실제 gripper geometry를 함께 고려해야 한다.

필요 조건:

- vertical 상태에서 손가락 끝과 바닥 간 clearance 확보
- vertical close 시 실린더를 양쪽 finger contact로 잡을 수 있는 높이와 폭 보정
- success sensor/contact 기준 재검증

이 방향은 가장 창의적이지만, 학습 전에 기구/시뮬레이션 설계 검증이 먼저 필요하다.

## 현재 상태

V12 코드 기반은 준비됐다.

- pitch action label 저장 가능
- rollout에서 pitch action 실행 가능
- hard radial interference scene sampling 가능
- smoke generation script 추가됨

추가 파일:

```text
scripts/05_generate_v12_pitch_smoke.sh
scripts/06_convert_v12_pitch_smoke.sh
```

하지만 full vertical close raw episode는 아직 안정적으로 성공하지 못했다. 따라서 학습으로 넘어가기 전, 다음 단계는 `A안: pitch-aware approach + horizontal close` 또는 `B안: partial pitch grasp` 중 하나를 선택해 smoke raw trajectory를 다시 설계하는 것이다.
