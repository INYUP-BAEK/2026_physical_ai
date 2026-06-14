# OpenVLA Closed-loop Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt`
- rollouts: `100`
- max steps: `50`
- step seconds: `0.1`
- center crop: `True`
- max delta xyz: `0.12`
- workspace z min/max: `0.016` / `0.1`
- lift threshold: `0.01`

## Summary

- ever contact success: `82/100` (0.820)
- ever strict lift success: `80/100` (0.800)
- final strict lift success: `80/100` (0.800)
- wrong-color touch: `9/100` (0.090)
- exceptions: `0`
- mean final lift delta: `0.010306m`
- mean steps: `40.63`

## By Color

- blue: n `25`, strict-ever `0.760`, strict-final `0.760`, wrong-touch `0.120`, mean final lift `0.010140m`
- green: n `25`, strict-ever `0.760`, strict-final `0.760`, wrong-touch `0.080`, mean final lift `0.009776m`
- red: n `25`, strict-ever `0.760`, strict-final `0.760`, wrong-touch `0.120`, mean final lift `0.009870m`
- yellow: n `25`, strict-ever `0.920`, strict-final `0.920`, wrong-touch `0.040`, mean final lift `0.011437m`
