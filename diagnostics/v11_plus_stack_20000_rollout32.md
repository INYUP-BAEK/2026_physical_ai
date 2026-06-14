# OpenVLA Closed-loop Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt`
- rollouts: `32`
- max steps: `50`
- step seconds: `0.1`
- center crop: `True`
- max delta xyz: `0.12`
- workspace z min/max: `0.016` / `0.1`
- lift threshold: `0.01`

## Summary

- ever contact success: `16/32` (0.500)
- ever strict lift success: `15/32` (0.469)
- final strict lift success: `15/32` (0.469)
- wrong-color touch: `3/32` (0.094)
- exceptions: `0`
- mean final lift delta: `0.007969m`
- mean steps: `44.56`

## By Color

- blue: n `8`, strict-ever `0.625`, strict-final `0.625`, wrong-touch `0.125`, mean final lift `0.008103m`
- green: n `8`, strict-ever `0.125`, strict-final `0.125`, wrong-touch `0.000`, mean final lift `0.009653m`
- red: n `8`, strict-ever `0.375`, strict-final `0.375`, wrong-touch `0.000`, mean final lift `0.004575m`
- yellow: n `8`, strict-ever `0.750`, strict-final `0.750`, wrong-touch `0.250`, mean final lift `0.009544m`
