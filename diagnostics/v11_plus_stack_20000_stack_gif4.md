# OpenVLA Stack Rollout Evaluation

- model: `/data/biy/Raccoonbot_Openvla/openvla/openvla-runs/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt`
- adapter: `/data/biy/Raccoonbot_Openvla/openvla/openvla-adapter-tmp/47a0ec7fc4ec123775a391911046cf33cf9ed83f+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--v11-plus-stack-close-boost-stack120-1320eps-20000steps-save5000--image_aug--20000_chkpt`
- rollouts: `4`
- max steps: `100`
- step seconds: `0.1`
- stack xy threshold: `0.02`
- stack z threshold: `0.014`
- object x range: `-0.1` / `0.1`
- object y range: `0.135` / `0.18`

## Summary

- final strict stack success: `2/4` (0.500)
- ever source contact: `3/4` (0.750)
- ever source lift: `3/4` (0.750)
- final gripper open: `3/4` (0.750)
- non-source robot touch: `0/4` (0.000)
- base robot touch: `0/4` (0.000)
- exceptions: `0`
- mean final stack xy distance: `0.020706m`
- mean final stack z delta: `0.022612m`
- mean source lift delta final: `0.022729m`
- mean steps: `92.25`

## Failure Classes

- strict_stack_success_clean: `2`
- no_source_grasp_or_lift: `1`
- on_top_but_not_released: `1`

## By Source Color

- blue: n `1`, stack `1.000`, source-lift `1.000`, xy `0.004498m`, dz `0.020067m`
- red: n `2`, stack `0.500`, source-lift `0.500`, xy `0.034146m`, dz `0.010030m`
- yellow: n `1`, stack `0.000`, source-lift `1.000`, xy `0.010034m`, dz `0.050320m`

## By Base Color

- blue: n `2`, stack `0.000`, source-lift `0.500`, xy `0.034492m`, dz `0.025160m`
- yellow: n `2`, stack `1.000`, source-lift `1.000`, xy `0.006920m`, dz `0.020064m`
