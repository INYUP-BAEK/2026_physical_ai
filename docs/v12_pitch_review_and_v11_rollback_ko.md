# V12 Pitch 실험 검토 및 V11 롤백 결정

작성일: 2026-06-12

## 결론

V12에서 검토한 `gripper pitch` 기반 수직 grasp는 최종 학습 방향에서 제외한다.

가장 큰 이유는 위치 정렬 문제가 아니라 접촉 안정성 문제다. 수직 pitch 상태에서 그리퍼가 원통형 실린더를 시각적으로 정확히 감싸도, 실린더가 둥글고 접촉면이 작아 lift 중 미끄러져 빠지는 현상이 확인됐다. 이 실패는 데이터셋을 늘리거나 close z/xy를 조금 조정해서 해결할 성격이 아니며, 그대로 학습하면 모델이 불안정한 grasp label을 배우게 된다.

따라서 최종 전략은 V11 계열의 수평 close/lift trajectory로 롤백하고, 목표를 `수평 grasp 성공률 99%에 최대한 근접`으로 재설정한다.

## 검토한 pitch 활용 시나리오

### 1. Full vertical grasp

내용:

- 같은 radial line에서 앞쪽 실린더가 링크/그리퍼 바디와 간섭할 때, 그리퍼를 수직으로 세워 먼 실린더를 직접 잡는 방식
- safe z에서 pitch를 먼저 내리고, 이후 z 하강 및 close 수행

판단:

- 탈락
- close z를 높이면 바닥 충돌은 줄지만 실린더를 충분히 물지 못한다.
- close z와 xy를 맞춰도 원통형 실린더가 lift 중 미끄러져 빠진다.
- 성공 조건이 불안정해 학습 데이터로 쓰기 어렵다.

### 2. Partial pitch grasp

내용:

- 완전 수직 대신 중간 pitch로 접근 및 close 수행

판단:

- 최종 학습에는 부적합
- full vertical보다 바닥 충돌은 줄일 수 있지만, 원통 실린더의 미끄러짐 문제는 여전히 남는다.
- 적정 pitch, close z, radial compensation을 추가 sweep해야 하며, 마지막 학습 기회에 쓰기에는 위험하다.

### 3. Pitch-aware approach + horizontal close

내용:

- pitch는 간섭 회피용 접근에만 사용하고, close 직전에는 다시 V11 수평 grasp로 전환

판단:

- 이론적으로는 가능
- 하지만 학습 관점에서는 `pitch up/down`, `xy 이동`, `close 전 수평 복귀`가 추가되어 action sequence가 길고 복잡해진다.
- 현재 V11 실패의 핵심은 대부분 close/lift 안정화 문제이므로, 이 시나리오가 최종 성공률을 크게 올린다고 보기 어렵다.

### 4. Grasp 이후 pitch 시연

내용:

- 수평 grasp로 잡은 뒤 pitch를 움직여 로봇이 물체를 제어한다는 점을 보여주는 방식

판단:

- 보고서의 창의성 시연으로는 가능하지만, 자연어 지시에 맞게 특정 색 실린더를 집고 들어올리는 핵심 성공률에는 직접 기여하지 않는다.
- 최종 모델 성공률 최적화 단계에서는 제외한다.

## 코드 정리

정리한 항목:

- V12 전용 생성 스크립트 제거
  - `scripts/05_generate_v12_pitch_smoke.sh`
- V12 전용 변환 스크립트 제거
  - `scripts/06_convert_v12_pitch_smoke.sh`
- V12 실험 raw 산출물 제거
  - `Mujoco/raccoon_grasp_v12_pitch_alpha_sweep`
  - `Mujoco/raccoon_grasp_v12_pitch_debug4`
  - `Mujoco/raccoon_grasp_v12_pitch_pilot8`
  - `Mujoco/raccoon_grasp_v12_vertical_closez_sweep`
- `make_pitch_grasp_plan()`의 full vertical branch는 fail-fast 처리
  - 실수로 v12 vertical trajectory를 다시 생성하지 않도록 막음

남겨둔 항목:

- converter/evaluator의 optional pitch action 인자는 유지
- 기본값이 꺼져 있어 V11 데이터 생성, 변환, 학습, rollout에는 영향 없음
- 추후 실제 gripper 형상 변경이나 다른 물체 과제가 생기면 재사용 가능

## 최종 권장 방향

V11 수평 grasp/lift 계열에서 성공률을 올리는 것이 최선이다.

우선순위:

1. 수평 close가 확실히 들어가는 trajectory 유지
2. first close 직후 lift action이 바로 나오도록 유지
3. closed gripper 상태에서 z action이 작은 transition 제거 유지
4. raw dataset은 성공 episode만 사용
5. rollout 실패를 `wrong color`, `grip miss`, `lift slip`, `workspace/interference`로 분류해 마지막 데이터 분포 조정

최종 목표는 pitch 제어를 억지로 포함하는 것이 아니라, 과제 핵심인 자연어 기반 색상 선택 grasp/lift 성공률을 최대화하는 것이다.
