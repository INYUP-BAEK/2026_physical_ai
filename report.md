RaccoonBot OpenVLA 최종 보고서
2021741027 백인엽

1. 프로젝트 목표

레퍼런스 코드는 4축 RaccoonBot MuJoCo 환경에서 자연어로 지정된 색상의 실린더를 집는 OpenVLA 학습 예제이다. 본 프로젝트의 목표는 이 기본 동작을 더 안정적이고 확장 가능한 조작 정책으로 발전시키는 것이다.


최종적으로 두 가지 모델을 중심으로 프로젝트를 진행하였다.

1. V11 Lift 모델
- 자연어 명령에 맞는 색상 실린더를 선택하고, gripper close 이후 실제로 들어올리는 동작까지 수행한다.


2. V11+Stack 모델
- V11 Lift 정책을 기반으로 한 색상 실린더를 다른 색상 실린더 위에 올리는 stack 명령을 추가했다.


2. 레퍼런스 코드 대비 변경점
   2.1 자연어 명령 확장

레퍼런스는 제한된 형태의 색상 grasp 명령을 사용했다. 최종 코드에서는 색상별 pick/lift 명령 템플릿을 확장하고, stack 모델에서는 다음과 같은 2색 조합 명령을 추가했다.

추가된 명령 템플릿
- `grasp the {color} cylinder`
- `pick up the {color} cylinder`
- `lift the {color} cylinder`
- `raise the {color} cylinder`
- `grab and lift the {color} cylinder`
- `grasp and lift the {color} cylinder`
- `pick up only the {color} cylinder`
- `grasp only the {color} cylinder`
- `lift the {color} cylinder without touching the others`

  2.2 4색 균형 데이터 생성

기존 데이터 생성에서는 하나의 seed 배치에 따라 하나의 목표에 대한 동작만 수행했는데 이는 목표 색깔에 대해서는 데이터가 골고루 수집될 수 있지만 물체 위치에 대해서는 편향이 일어날 가능성이 있다. 따라서 하나의 배치에서 4가지 색의 실린더를 모두 잡도록 하여 색상 및 배치의 균형을 맞췄다.

  2.3 Grasp-only에서 Lift까지 확장

기존 코드의 경우 물체에 그리퍼가 닿는 것까지가 목표였지만 이번 개선 사항에서는 물체 그립 이후 z축 방향으로 들어올리는 동작까지 추가하여 단순히 접촉만 된것이 아닌 물체를 올바르게 통제할 수 있는 정확성을 보이도록 하였다.

  2.4 Action label 개선

최종 변환 파이프라인은 FK 기반 end-effector command space에서 `command_delta`를 생성한다. 즉, joint target을 그대로 학습하기보다 end-effector의 명령 변화량을 action으로 사용한다.

  2.5 VLA-only rollout 검증

최종 평가는 외부 하드코딩 컨트롤러가 성공률을 보정하지 않는지 확인하는 방향으로 진행했다. rollout은 OpenVLA가 출력한 action을 그대로 MuJoCo에 적용하는 방식이며, 성공 판정만 별도로 계산한다.

  2.6 Stack 확장

lift동작 후 추가 동작에 대한 테스트를 위해 stack task를 추가했다.

기본 흐름은 다음과 같다.

1. source 색상 실린더 접근
2. gripper close
3. source 실린더 lift
4. base 색상 실린더 위로 이동
5. 내려놓기 및 gripper open

3. 최종 모델별 결과
  3.1 V11 Lift

V11 Lift는 1200개 lift raw episode를 사용했다. 변환 후 학습/검증 분할은 1080/120 episode 기준으로 구성했다. 총 학습 스텝은 15000스텝으로 학습을 진행했다.

평가 결과는 아래와 같다.

MuJoCo closed-loop rollout 결과:

| 평가 항목 | 결과 |
|---|---:|
| 100 rollout strict lift 성공 | 80/100 |
| 100 rollout wrong-color touch | 9/100 |
| 32 rollout strict lift 성공 | 25/32 |
| 32 rollout wrong-color touch | 2/32 |

총 100번의 rollout 평가 결과 성공 케이스는 80개이고 실패 케이스는 20개로 약 80%의 성공률을 보였다. 실패 케이스의 실패 이유는 잘못된 색상의 터치가 9건, 그리퍼 close 동작을 수행하지 못한것이 11건이었다.

결론적으로 대부분의 실패는 목표 물체 직전까지 접근한 뒤 gripper close가 약하거나 타이밍이 어긋나서 물체가 충분히 잡히지 않는 경우였다. 반대로 목표 실린더 위치까지 접근하는 능력은 상당히 안정적이었다.

  3.2 V11+Stack

V11+Stack은 V11 Lift 1200 episode에 stack 120 episode를 추가한 1320 episode 구성을 사용했다. 학습 스텝으로는 20000스텝으로 학습을 진행했다.

평가 결과는 아래와 같다. 

| 평가 항목 | 결과 |
|---|---:|
| lift command strict lift 성공 | 15/32 |
| stack command strict stack 성공 | 4/12 |
| stack command source lift 성공 | 9/12 |
| stack command final gripper open | 7/12 |

총 32번의 평가 결과 약 50%의 성공률을 보여줬다. 실패케이스의 경우 실린더 색상 및 배치에 영향을 받지는 않았으나 기존 V11의 데이터 그대로에 stack 에피소드를 섞어 학습을 진행한 결과인지 V11에서 발생했던 실패 케이스인 그리퍼 close실패로 인한 실패케이스가 가장 많았다. 두번째로는 목표 stack 위치에 도달하여 그리퍼를 release하지 못하는 케이스가 많았다.

결론적으로 그리퍼를 이동시키는 방식에 대해서는 학습이 잘 되었으나 그리퍼를 열고 닫는 타이밍에 대해서 잘 배우지 못한 모습을 보였다.
