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

The action student can learn multiple source-to-target transitions, such as:

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

The normal action `timesteps` field carries the source timestep `sigma_s`.
The clean action condition keeps its original `cond_timesteps=0`; it is not
used to carry the target timestep.  The target timestep `sigma_e` is passed
through a separate zero-initialized target-time embedder:

```text
train path:      action_dict["action_target_timesteps"]
action sampler:  input_dict["target_timesteps"]
```

The target-time projection is added only to noisy action tokens.  This keeps the
clean action condition clean and makes the training path match the action-only
inference path used by future routers.

Sanity check before a long run:

```text
fix x_s, sigma_s, C
compare Phi_a(x_s, sigma_s, sigma_e=0.5, C)
     vs Phi_a(x_s, sigma_s, sigma_e=0.0, C)
```

The two predictions should differ; otherwise target-time conditioning is not
being used.

## Source Timestep Sampling

When action flow-map is enabled, action source timesteps are resampled from the
legal range implied by the largest configured stride:

```text
max_start = num_train_timesteps - 1 - max_stride
```

This avoids clipped transitions such as `start + stride > 999`, which would
otherwise turn a large-stride target into a shorter or near-identity target.

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
ACTION_FLOWMAP_STRIDE_RATIOS=1.0
ACTION_FLOWMAP_LOSS_WEIGHTS=1.0
ACTION_FLOWMAP_TEACHER_MIN_SUBSTEPS=8
ACTION_FLOWMAP_TEACHER_MAX_SUBSTEPS=16
ACTION_FLOWMAP_ENDPOINT_WEIGHT=0.05
ACTION_FLOWMAP_SELF_CONSISTENCY_WEIGHT=0.0
```

Teacher action rollout substeps scale with stride and are capped by
`ACTION_FLOWMAP_TEACHER_MAX_SUBSTEPS`.  The first stable experiment should use a
single deploy-scale stride and keep self-consistency off.  After the endpoint
and flow-map losses are stable, add shorter ratios and optionally enable
self-consistency with a warmup.
