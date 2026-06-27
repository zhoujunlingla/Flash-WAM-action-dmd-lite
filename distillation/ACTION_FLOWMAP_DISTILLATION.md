# Action Flow-Map Distillation

This branch keeps Flash-WAM's video consistency distillation, but replaces the
action consistency target with direct teacher action trajectory transitions.
It does not train a fake score, distribution critic, discriminator, or extra
student model.

## What Changes

Original Flash-WAM action distillation predicts a clean action endpoint from a
noisy action state and matches an EMA consistency target.

This branch instead trains the action student as a flow-map:

```text
teacher action trajectory:  x_s -> x_e^T
student transition:         Phi_a^S(x_s, sigma_s, sigma_e, C) -> x_e^T
```

The action student learns multiple source-to-target transitions, such as:

```text
sigma_s -> sigma_e at 0.5 stride
sigma_s -> sigma_e at 1.0 stride
```

The main loss is:

```text
L_action_flowmap =
Huber(Phi_a^S(x_s, sigma_s, sigma_e, C), stopgrad(x_e^T))
```

where:

- `x_s` is the source noisy action state
- `sigma_s` is the source action noise level
- `sigma_e` is the target action noise level
- `x_e^T` is the teacher rollout endpoint from `sigma_s` to `sigma_e`
- `C` is the language/observation context and the student video cache

## WAM-Specific Detail

The action branch reads the student video cache, not the teacher video cache:

```text
student video consistency -> student video cache
student action flow-map reads student video cache
teacher action rollout also uses that student video cache as condition
```

This keeps training and inference aligned.  The action loss cannot rely on a
clean teacher video context that will be unavailable at runtime.

## Target-Time Conditioning

No DiT architecture change is required.  The existing action
`cond_timesteps` field is reused to carry the target timestep `sigma_e`, while
the normal action `timesteps` field carries the source timestep `sigma_s`.

## Loss

The first implementation uses:

```text
L_total =
  L_video_consistency
+ lambda_flow * L_action_flowmap
+ lambda_aware * L_action_fm_regularizer
+ lambda_ep * L_action_endpoint
+ lambda_sc * L_action_self_consistency
```

`L_action_flowmap` is the main action loss.  `L_action_endpoint` anchors the
final clean executable action to the dataset action.  `L_action_self_consistency`
encourages the direct long transition and the composed short-to-long transition
to agree, which is useful for future multi-level routers.

## Config

Use `distillation/config_robotwin_1v2a.py` for RoboTwin v1/a2:

```bash
ACTION_FLOWMAP_ENABLE=1
ACTION_FLOWMAP_STRIDE_RATIOS=0.5,1.0
ACTION_FLOWMAP_LOSS_WEIGHTS=0.5,1.0
ACTION_FLOWMAP_TEACHER_MIN_SUBSTEPS=1
ACTION_FLOWMAP_TEACHER_MAX_SUBSTEPS=4
ACTION_FLOWMAP_ENDPOINT_WEIGHT=0.05
ACTION_FLOWMAP_SELF_CONSISTENCY_WEIGHT=0.05
```

Teacher action rollout substeps scale with stride.  With the default ratios,
the shorter target uses one teacher substep and the deploy-scale target uses two
teacher substeps, capped by `ACTION_FLOWMAP_TEACHER_MAX_SUBSTEPS`.
