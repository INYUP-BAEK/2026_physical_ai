# RaccoonBot OpenVLA 최종 보고서

**학번 / 이름:** 2021741027 백인엽

## 1. 프로젝트 목표

레퍼런스 코드는 4축 RaccoonBot MuJoCo 환경에서 자연어로 지정된 색상의 실린더를 집는 OpenVLA 학습 예제이다. 본 프로젝트의 목표는 이 기본 동작을 더 안정적이고 확장 가능한 조작 정책으로 발전시키는 것이다.

최종적으로는 다음 두 가지 모델을 중심으로 프로젝트를 진행하였다.

### 1.1 V11 Lift 모델

- 자연어 명령에 맞는 색상 실린더를 선택한다.
- gripper close 이후 실제로 실린더를 들어올리는 동작까지 수행한다.

### 1.2 V11+Stack 모델

- V11 Lift 정책을 기반으로 한다.
- 한 색상 실린더를 다른 색상 실린더 위에 올리는 stack 명령을 추가했다.

## 2. 레퍼런스 코드 대비 변경점

### 2.1 자연어 명령 확장

레퍼런스는 제한된 형태의 색상 grasp 명령을 사용했다. 최종 코드에서는 색상별 pick/lift 명령 템플릿을 확장하고, stack 모델에서는 2색 조합 명령을 추가했다.

추가된 주요 명령 템플릿은 다음과 같다.

- `grasp the {color} cylinder`
- `pick up the {color} cylinder`
- `lift the {color} cylinder`
- `raise the {color} cylinder`
- `grab and lift the {color} cylinder`
- `grasp and lift the {color} cylinder`
- `pick up only the {color} cylinder`
- `grasp only the {color} cylinder`
- `lift the {color} cylinder without touching the others`

### 2.2 4색 균형 데이터 생성

기존 데이터 생성에서는 하나의 seed 배치에 따라 하나의 목표에 대한 동작만 수행했다. 이 방식은 목표 색상에 대해서는 데이터가 골고루 수집될 수 있지만, 물체 위치에 대해서는 편향이 발생할 가능성이 있다.

이를 보완하기 위해 하나의 배치에서 4가지 색상의 실린더를 모두 잡도록 데이터 생성 방식을 변경하였다. 이를 통해 색상과 배치에 대한 균형을 맞추고, 특정 색상이나 위치에 과적합되는 문제를 줄이고자 했다.

### 2.3 Grasp-only에서 Lift까지 확장

기존 코드는 물체에 그리퍼가 닿는 것까지를 주요 목표로 삼았다. 이번 개선에서는 물체를 잡은 이후 z축 방향으로 들어올리는 동작까지 추가하였다.

이를 통해 단순 접촉이 아니라, 로봇팔이 물체를 올바르게 통제할 수 있음을 보여주는 방향으로 task를 확장하였다.

### 2.4 Action Label 개선

최종 변환 파이프라인은 FK 기반 end-effector command space에서 `command_delta`를 생성한다. 즉, joint target을 그대로 학습하기보다 end-effector의 명령 변화량을 action으로 사용한다.

이 변경은 모델이 로봇 관절값 자체보다 실제 end-effector 이동 방향과 변화량을 학습하도록 하기 위한 것이다.

### 2.5 VLA-only Rollout 검증

최종 평가는 외부 하드코딩 컨트롤러가 성공률을 보정하지 않는지 확인하는 방향으로 진행했다.

Rollout은 OpenVLA가 출력한 action을 그대로 MuJoCo에 적용하는 방식으로 수행했으며, 성공 여부는 별도의 평가 로직으로만 계산하였다. 따라서 최종 성공률은 외부 제어기 보정이 아니라 VLA 출력 기반의 결과로 해석할 수 있다.

### 2.6 Stack 확장

Lift 동작 이후 추가 조작 task를 테스트하기 위해 stack task를 추가했다.

Stack task의 기본 흐름은 다음과 같다.

1. Source 색상 실린더에 접근한다.
2. Gripper를 close한다.
3. Source 실린더를 lift한다.
4. Base 색상 실린더 위로 이동한다.
5. 내려놓기 및 gripper open을 수행한다.

## 3. 최종 모델별 결과

### 3.1 V11 Lift

V11 Lift는 1200개의 lift raw episode를 사용했다. 변환 후 학습/검증 분할은 1080/120 episode 기준으로 구성했다. 총 학습 스텝은 15000 step으로 설정하였다.

MuJoCo closed-loop rollout 평가 결과는 다음과 같다.

| 평가 항목 | 결과 |
| --- | ---: |
| 100 rollout strict lift 성공 | 80/100 |
| 100 rollout wrong-color touch | 9/100 |
| 32 rollout strict lift 성공 | 25/32 |
| 32 rollout wrong-color touch | 2/32 |

총 100번의 rollout 평가 결과 성공 케이스는 80개, 실패 케이스는 20개로 약 80%의 성공률을 보였다. 실패 케이스의 원인은 잘못된 색상 터치가 9건, gripper close 동작을 충분히 수행하지 못한 경우가 11건이었다.

결론적으로 대부분의 실패는 목표 물체 직전까지 접근한 뒤 gripper close가 약하거나 타이밍이 어긋나서 물체가 충분히 잡히지 않는 경우였다. 반대로 목표 실린더 위치까지 접근하는 능력은 상당히 안정적이었다.

### 3.2 V11+Stack

V11+Stack은 V11 Lift 1200 episode에 stack 120 episode를 추가한 1320 episode 구성을 사용했다. 총 학습 스텝은 20000 step으로 설정하였다.

평가 결과는 다음과 같다.

| 평가 항목 | 결과 |
| --- | ---: |
| Lift command strict lift 성공 | 15/32 |
| Stack command strict stack 성공 | 4/12 |
| Stack command source lift 성공 | 9/12 |
| Stack command final gripper open | 7/12 |

총 32번의 lift command 평가에서는 약 50%의 성공률을 보였다. 실패 케이스의 경우 실린더 색상 및 배치 자체보다는 V11에서도 발생했던 gripper close 실패가 가장 큰 원인이었다.

Stack command에서는 source 실린더를 들어올리는 단계까지는 비교적 성공하는 경우가 있었지만, 최종 stack 위치에 도달한 뒤 gripper release를 수행하지 못하는 케이스가 많았다.

결론적으로 그리퍼를 목표 위치까지 이동시키는 방식은 어느 정도 학습되었으나, gripper open/close 타이밍을 안정적으로 학습하는 데에는 한계가 있었다.

## 4. 결론

본 프로젝트에서는 레퍼런스의 단순 색상 grasp task를 기반으로 자연어 명령 확장, lift 동작 추가, 4색 균형 데이터 생성, `command_delta` 기반 action label 변환, stack task 확장까지 진행하였다.

V11 Lift 모델은 목표 색상 실린더에 접근하고 들어올리는 동작에서 비교적 안정적인 성능을 보였다. V11+Stack 모델은 stack이라는 확장 task를 시도했다는 점에서 의미가 있었지만, gripper close/open 타이밍이 성공률을 제한하는 주요 요인으로 남았다.

따라서 최종 결과는 V11 Lift를 안정적인 기본 조작 모델로, V11+Stack을 창의적 확장 실험 모델로 정리할 수 있다.
