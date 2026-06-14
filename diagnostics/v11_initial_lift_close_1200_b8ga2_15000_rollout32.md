# OpenVLA Closed-loop Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-initial-lift-close-1200eps-15000steps-b8ga2--image_aug--15000_chkpt`
- rollouts: `32`
- max steps: `50`
- step seconds: `0.1`
- center crop: `True`
- max delta xyz: `0.12`
- workspace z min/max: `0.016` / `0.1`
- lift threshold: `0.01`

## Summary

- ever contact success: `26/32` (0.812)
- ever strict lift success: `25/32` (0.781)
- final strict lift success: `25/32` (0.781)
- wrong-color touch: `2/32` (0.062)
- exceptions: `0`
- mean final lift delta: `0.010014m`
- mean steps: `41.03`

## By Color

- blue: n `8`, strict-ever `0.750`, strict-final `0.750`, wrong-touch `0.125`, mean final lift `0.009440m`
- green: n `8`, strict-ever `0.750`, strict-final `0.750`, wrong-touch `0.000`, mean final lift `0.010232m`
- red: n `8`, strict-ever `0.625`, strict-final `0.625`, wrong-touch `0.125`, mean final lift `0.007835m`
- yellow: n `8`, strict-ever `1.000`, strict-final `1.000`, wrong-touch `0.000`, mean final lift `0.012551m`
