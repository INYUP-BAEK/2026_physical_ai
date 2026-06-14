# OpenVLA Stack Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt`
- adapter: `/data/biy/Raccoonbot_Openvla/openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt`
- rollouts: `12`
- max steps: `100`
- step seconds: `0.1`
- stack xy threshold: `0.02`
- stack z threshold: `0.014`
- object x range: `-0.1` / `0.1`
- object y range: `0.135` / `0.18`

## Summary

- final strict stack success: `4/12` (0.333)
- ever source contact: `9/12` (0.750)
- ever source lift: `9/12` (0.750)
- final gripper open: `7/12` (0.583)
- non-source robot touch: `0/12` (0.000)
- base robot touch: `0/12` (0.000)
- exceptions: `0`
- mean final stack xy distance: `0.020776m`
- mean final stack z delta: `0.028310m`
- mean source lift delta final: `0.028521m`
- mean steps: `96.67`

## Failure Classes

- on_top_but_not_released: `5`
- strict_stack_success_clean: `4`
- no_source_grasp_or_lift: `3`

## By Source Color

- blue: n `3`, stack `0.333`, source-lift `1.000`, xy `0.002802m`, dz `0.035524m`
- green: n `3`, stack `0.333`, source-lift `0.333`, xy `0.047538m`, dz `0.005704m`
- red: n `3`, stack `0.667`, source-lift `0.667`, xy `0.025155m`, dz `0.013374m`
- yellow: n `3`, stack `0.000`, source-lift `1.000`, xy `0.007609m`, dz `0.058637m`

## By Base Color

- blue: n `3`, stack `0.333`, source-lift `0.667`, xy `0.025581m`, dz `0.023428m`
- green: n `3`, stack `0.333`, source-lift `1.000`, xy `0.004798m`, dz `0.048987m`
- red: n `3`, stack `0.000`, source-lift `0.667`, xy `0.028270m`, dz `0.027448m`
- yellow: n `3`, stack `0.667`, source-lift `0.667`, xy `0.024455m`, dz `0.013376m`
