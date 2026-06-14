# RaccoonBot OpenVLA 최종 보고서

## 1. 프로젝트 목표

레퍼런스 코드는 4축 RaccoonBot MuJoCo 환경에서 자연어로 지정된 색상의 실린더를 집는 OpenVLA 학습 예제이다. 본 프로젝트의 목표는 이 기본 동작을 더 안정적이고 확장 가능한 조작 정책으로 발전시키는 것이다.

최종적으로 두 가지 모델을 중심으로 결과를 정리했다.

1. **V11 Lift 모델**
   - 자연어 명령에 맞는 색상 실린더를 선택하고, gripper close 이후 실제로 들어올리는 동작까지 수행한다.
   - 현재 가장 성공률이 높은 기본 최종 모델이다.

2. **V11+Stack 모델**
   - V11 Lift 정책을 기반으로 한 색상 실린더를 다른 색상 실린더 위에 올리는 stack 명령을 추가했다.
   - 성공률은 lift-only 모델보다 낮지만, 단순 grasp/lift를 넘어 placement 성격의 창의적 과제를 추가했다는 의미가 있다.

## 2. 레퍼런스 코드 대비 변경점

### 2.1 자연어 명령 확장

레퍼런스는 제한된 형태의 색상 grasp 명령을 사용했다. 최종 코드에서는 색상별 pick/lift 명령 템플릿을 확장하고, stack 모델에서는 다음과 같은 2색 조합 명령을 추가했다.

```text
stack the red cylinder on the blue cylinder
```

이 변화로 모델이 단순 위치 모방만 하는 것이 아니라 이미지와 언어 조건을 함께 사용해야 하는 구조를 유지했다.

### 2.2 4색 균형 데이터 생성

기존 데이터 생성 흐름에서는 seed나 배치에 따라 특정 색상 또는 특정 배치가 과하게 반영될 수 있었다. 최종 데이터 생성에서는 하나의 씬에서 red, blue, green, yellow 목표 에피소드를 모두 생성하는 방식으로 색상 균형을 맞췄다.

또한 물리적으로 성공이 어려운 배치가 반복될 때는 실패 에피소드 번호가 비지 않도록 재시도 및 삭제 로직을 정리했다. 이로 인해 학습 데이터의 episode index와 실제 성공 episode 수가 안정적으로 맞춰졌다.

### 2.3 Grasp-only에서 Lift까지 확장

레퍼런스의 핵심은 목표 실린더를 잡는 것이었다. 본 프로젝트에서는 gripper close 이후 z 방향으로 들어올리는 trajectory를 추가했다. 이는 실제 로봇 제어 관점에서 단순 접촉보다 더 명확한 성공 기준을 제공한다.

초기 실험에서는 gripper close 명령이 충분히 강하게 학습되지 않아 물체를 건드리거나 위치까지 접근한 뒤 놓치는 실패가 많았다. 이를 보완하기 위해 close transition 주변의 action label을 강화했다.

### 2.4 Action label 개선

최종 변환 파이프라인은 FK 기반 end-effector command space에서 `command_delta`를 생성한다. 즉, joint target을 그대로 학습하기보다 end-effector의 명령 변화량을 action으로 사용한다.

적용한 주요 처리:

- `promote_pre_close_steps=3`
  - close 직전 프레임에도 close 신호가 더 일찍 보이도록 보강했다.
- `initial_close_min_z_action=0.004`
  - first close 이후 첫 closed frame부터 위로 들어올리는 action이 나오도록 했다.
- post-close hold 또는 z action이 거의 없는 transition은 줄이거나 제거했다.
  - gripper가 닫힌 상태에서 멈추는 행동을 학습하지 않도록 하기 위한 조치이다.

### 2.5 VLA-only rollout 검증

최종 평가는 외부 하드코딩 컨트롤러가 성공률을 보정하지 않는지 확인하는 방향으로 진행했다. rollout은 OpenVLA가 출력한 action을 그대로 MuJoCo에 적용하는 방식이며, 성공 판정만 별도로 계산한다.

따라서 V11 Lift의 성공률은 외부 controller가 목표 위치를 보정한 결과가 아니라, 학습된 VLA 출력의 closed-loop 결과로 해석할 수 있다.

### 2.6 Stack 확장

lift동작 후 추가 동작에 대한 테스트를 위해 stack task를 추가했다. Stack trajectory는 V11 Lift 동작을 확장하는 방식으로 구성했다.

기본 흐름은 다음과 같다.

1. source 색상 실린더 접근
2. gripper close
3. source 실린더 lift
4. base 색상 실린더 위로 이동
5. 내려놓기 및 gripper open

V11+Stack은 처음부터 새로 학습하지 않고 V11 Lift adapter에서 이어서 학습하도록 구성했다. 이는 lift 동작을 유지하면서 placement skill을 추가하기 위한 선택이다.

## 3. 최종 모델별 결과

### 3.1 V11 Lift

V11 Lift는 1200개 lift raw episode를 사용했다. 변환 후 학습/검증 분할은 1080/120 episode 기준으로 구성했다.

대표 학습 설정:

```text
MAX_STEPS=15000
SAVE_STEPS=2500
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=2
LoRA r=32
```

MuJoCo closed-loop rollout 결과:

| 평가 항목 | 결과 |
|---|---:|
| 100 rollout strict lift 성공 | 80/100 |
| 100 rollout wrong-color touch | 9/100 |
| 32 rollout strict lift 성공 | 25/32 |
| 32 rollout wrong-color touch | 2/32 |

V11 Lift는 최종 기본 모델로 볼 수 있다. 대부분의 실패는 목표 물체 직전까지 접근한 뒤 gripper close가 약하거나 타이밍이 어긋나서 물체가 충분히 잡히지 않는 경우였다. 반대로 목표 실린더 위치까지 접근하는 능력은 상당히 안정적이었다.

관련 시각화:

```text
reports/v11_initial_lift_close_1200_b8ga2_15000_gif4/
reports/v11_final_baseline_assets/
diagnostics/v11_initial_lift_close_1200_b8ga2_15000_rollout100_failgifs.md
```

### 3.2 V11+Stack

V11+Stack은 V11 Lift 1200 episode에 stack 120 episode를 추가한 1320 episode 구성을 사용했다.

대표 학습 설정:

```text
MAX_STEPS=20000
SAVE_STEPS=5000
LoRA continuation from V11 Lift adapter
```

MuJoCo closed-loop rollout 결과:

| 평가 항목 | 결과 |
|---|---:|
| lift command strict lift 성공 | 15/32 |
| stack command strict stack 성공 | 4/12 |
| stack command source lift 성공 | 9/12 |
| stack command final gripper open | 7/12 |

Stack failure 분석:

| 유형 | 개수 |
|---|---:|
| strict stack success clean | 4 |
| on top but not released | 5 |
| no source grasp or lift | 3 |

Stack 모델은 일부 episode에서 source 실린더를 들어올리고 base 실린더 위로 이동하는 동작을 보여줬다. 다만 release가 안정적으로 되지 않거나, lift 단계에서 gripper close가 충분히 강하게 나오지 않는 문제가 있었다.

관련 시각화:

```text
reports/v11_plus_stack_20000_stack_gif4/
reports/v11_plus_stack_20000_stack_rollout12_pairs/
reports/v11_plus_stack_20000_lift_failgifs32/
diagnostics/v11_plus_stack_20000_stack_rollout12_pairs.md
```

## 4. 결과 해석

### 4.1 가장 안정적인 성과

가장 안정적인 성과는 V11 Lift 모델이다. 레퍼런스의 grasp-only 과제에서 다음 요소를 추가로 달성했다.

- 자연어 색상 명령 확장
- 목표 색상 선택
- gripper close
- 물체 lift
- VLA-only closed-loop rollout 평가
- GIF/진단 파일 기반 실패 분석

100 rollout에서 80% strict lift 성공률을 보였으므로, 최종 기본 정책으로 제시하기에 가장 적합하다.

### 4.2 Stack 확장의 의미

V11+Stack은 성공률만 보면 V11 Lift보다 낮지만, 과제의 창의성 항목을 설명하기 좋은 확장이다. 단순히 목표 물체를 잡는 것을 넘어, 두 물체의 관계를 포함하는 자연어 명령과 placement 동작을 시도했기 때문이다.

특히 stack 명령의 초반부는 lift와 유사하므로, lift 학습에서 얻은 접근 및 grasp skill이 stack 명령에도 부분적으로 전이될 수 있다. 실제 결과에서도 source 실린더를 들어올리는 단계까지는 성공하는 사례가 있었다.

### 4.3 주요 한계

최종 실험에서 가장 큰 병목은 gripper close action의 안정성이다. 실패 장면을 확인하면 end-effector가 목표 실린더 근처까지는 잘 이동하지만, 닫는 명령이 약하거나 늦게 나와서 lift 또는 stack이 실패하는 경우가 반복되었다.

따라서 다음 개선 방향은 다음과 같다.

- close 명령이 나오는 frame 비율을 더 높이기
- first close 이후 lift action이 바로 연결되도록 더 강하게 relabeling하기
- stack 데이터가 lift 정책을 약화하지 않도록 sampling ratio 조절하기
- release 단계와 close 단계의 action label이 서로 섞이지 않도록 task별 action 분포를 분리해서 점검하기

## 5. 서버, 클라이언트, 실제 로봇 확장

서버는 GPU에서 OpenVLA inference를 수행하고, 클라이언트는 MuJoCo 카메라 이미지를 서버의 `/predict` endpoint로 전송한다. 서버가 반환한 7D action은 MuJoCo에서 먼저 실행하고, 실제 로봇 테스트 시에는 같은 target xyz와 gripper command를 RaccoonBot IK 및 gripper 명령으로 변환한다.

실제 로봇 테스트는 안전을 위해 다음 순서로 진행하도록 정리했다.

1. 서버 health check
2. 클라이언트 MuJoCo-only smoke test
3. viewer 기반 VLA-only 동작 확인
4. 실제 로봇 연결 후 짧은 step 수로 dry-run
5. `--use_real_robot` 옵션으로 제한된 범위에서 실제 명령 전송

자세한 실행 절차는 다음 문서에 정리했다.

```text
docs/client_real_robot_test_plan_ko.md
```

## 6. 결론

본 프로젝트는 레퍼런스의 색상 조건 grasp 예제를 lift 가능한 조작 정책으로 확장했고, 추가로 stack task를 통해 더 창의적인 조작 시나리오를 실험했다.

최종 제출에서 중심으로 제시할 모델은 V11 Lift이다. V11 Lift는 성공률과 안정성이 가장 좋으며, 레퍼런스 대비 변경점도 명확하다. V11+Stack은 아직 완성된 정책이라기보다는 창의성 확장 실험에 가깝지만, 자연어 기반 다단계 조작으로 발전할 가능성을 보여주는 결과로 정리할 수 있다.

최종적으로 제출 자료에는 수정 코드, README, 본 report, 로그, GIF/이미지 시각화, 그리고 사용자가 직접 생성한 `report.pdf`를 포함하면 된다.
