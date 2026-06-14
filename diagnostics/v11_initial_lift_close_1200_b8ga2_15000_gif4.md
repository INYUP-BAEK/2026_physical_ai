# OpenVLA Closed-loop Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt`
- rollouts: `4`
- max steps: `50`
- step seconds: `0.1`
- center crop: `True`
- max delta xyz: `0.12`
- workspace z min/max: `0.016` / `0.1`
- lift threshold: `0.01`

## Summary

- ever contact success: `4/4` (1.000)
- ever strict lift success: `4/4` (1.000)
- final strict lift success: `4/4` (1.000)
- wrong-color touch: `0/4` (0.000)
- exceptions: `0`
- mean final lift delta: `0.012191m`
- mean steps: `36.50`

## By Color

- blue: n `1`, strict-ever `1.000`, strict-final `1.000`, wrong-touch `0.000`, mean final lift `0.012157m`
- green: n `1`, strict-ever `1.000`, strict-final `1.000`, wrong-touch `0.000`, mean final lift `0.012120m`
- red: n `1`, strict-ever `1.000`, strict-final `1.000`, wrong-touch `0.000`, mean final lift `0.012235m`
- yellow: n `1`, strict-ever `1.000`, strict-final `1.000`, wrong-touch `0.000`, mean final lift `0.012249m`
