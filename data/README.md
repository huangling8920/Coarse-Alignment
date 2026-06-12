# Data Format

The repository ships with a generator for a minimal toy dataset. Real synthetic polar sequences exported from Unity, vehicle-test processed data, or third-party VIO outputs can replace the toy files if they follow the schemas below.

## Required Files

```text
dataset_root/
├── calibration.yaml
├── truth.csv
├── imu.csv
├── gps.csv
├── vi_measurements.csv
├── quality_train.csv
└── images/
    ├── frame_000000.png
    └── ...
```

## `calibration.yaml`

```yaml
camera:
  width: 752
  height: 480
  fx: 376.0
  fy: 376.0
  cx: 376.0
  cy: 240.0
  distortion: [0, 0, 0, 0]
extrinsic:
  C_c_b: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  p_c_b: [0.05, 0.00, 0.02]
  time_offset_s: 0.0
  monocular_scale: 1.0
```

## `truth.csv`

```text
t,roll_deg,pitch_deg,heading_deg,bgx_dph,bgy_dph,bgz_dph
```

## `imu.csv`

```text
t,wx,wy,wz,ax,ay,az
```

Angular velocity is in rad/s and acceleration is in m/s^2.

## `gps.csv`

```text
t,vx,vy,vz,valid,sigma_v
```

Velocity is in m/s in the transverse frame. `valid=0` denotes an outage.

## `vi_measurements.csv`

```text
t_i,t_j,dp_x,dp_y,dp_z,dv_x,dv_y,dv_z,dq_w,dq_x,dq_y,dq_z,
omega_i_x,omega_i_y,omega_i_z,omega_j_x,omega_j_y,omega_j_z,
feature_count,track_success,inlier_ratio,feature_dispersion,
reproj_error_px,blur_indicator,reflection_indicator,preint_residual
```

## `quality_train.csv`

```text
n_feat_ratio,track_success,inlier_ratio,feature_dispersion,
reproj_norm,blur_indicator,reflection_indicator,preint_norm,rho_target
```

The target construction is a replaceable default because the paper does not fully specify the training-label generation.

