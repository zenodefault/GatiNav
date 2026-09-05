# Reference Equation and Porting Audit

## Scope and blocking status

`specs/02_ekf.md` still contains six empty HUMAN equation blocks, and
`specs/03_ml_noise_adaptation.md` contains one empty HUMAN loss block. No
paper equations were pasted into the specs, so there is nothing there to
transcribe.

The paper file `reference/brossard_ai_imu_dr.pdf` exists. The current
environment has no PDF text-extraction command available, so a paper-versus-
code discrepancy audit cannot be completed from the paper text. The
equations below are transcribed only from the executable reference source.

## 1. Equations present in the reference code

The Brossard repository is a provenance reference, not an unrestricted
transcription target. Its executable filter uses a 21-state covariance model,
car-to-IMU extrinsic states, and a two-row body-frame lateral/vertical
velocity update. Those structures are not part of this project's 15-state
mobile filter.

### Nominal propagation

From `reference/ai-imu-dr/src/utils_numpy_filter.py:169-181`:

```latex
\begin{aligned}
\mathbf{a}_n &= \mathbf{R}_{n-1}
  \left(\mathbf{u}_{a,n}-\mathbf{b}_{a,n-1}\right)+\mathbf{g},\\
\mathbf{v}_n &= \mathbf{v}_{n-1}+\mathbf{a}_n\Delta t,\\
\mathbf{p}_n &= \mathbf{p}_{n-1}+\mathbf{v}_{n-1}\Delta t
  +\frac{1}{2}\mathbf{a}_n\Delta t^2,\\
\boldsymbol{\omega}_n &= \mathbf{u}_{\omega,n}
  -\mathbf{b}_{\omega,n-1},\\
\mathbf{R}_n &= \mathbf{R}_{n-1}
  \operatorname{SO3Exp}\!\left(\boldsymbol{\omega}_n\Delta t\right).
\end{aligned}
```

The code holds both bias vectors and the vehicle-to-IMU extrinsics constant
during this step.

### Error covariance propagation

From `utils_numpy_filter.py:184-211`:

```latex
\begin{aligned}
\mathbf{F} &\leftarrow \Delta t\,\mathbf{F},&
\mathbf{G} &\leftarrow \Delta t\,\mathbf{G},\\
\mathbf{\Phi} &=
\mathbf{I}_{21}+\mathbf{F}
+\frac{1}{2}\mathbf{F}^{2}
+\frac{1}{6}\mathbf{F}^{3},\\
\mathbf{P}_n &=
\mathbf{\Phi}\left(
\mathbf{P}_{n-1}+\mathbf{G}\mathbf{Q}\mathbf{G}^{T}
\right)\mathbf{\Phi}^{T}.
\end{aligned}
```

The nonzero code assignments are:

```latex
\begin{aligned}
\mathbf{F}_{v,\theta} &= [\mathbf{g}]_\times,&
\mathbf{F}_{p,v} &= \mathbf{I}_3,\\
\mathbf{F}_{v,b_a} &= -\mathbf{R},&
\mathbf{F}_{\theta,b_\omega} &= -\mathbf{R},\\
\mathbf{F}_{v,b_\omega} &= -[\mathbf{v}]_\times\mathbf{R},&
\mathbf{F}_{p,b_\omega} &= -[\mathbf{p}]_\times\mathbf{R},\\
\mathbf{G}_{\theta,\omega} &= \mathbf{R},&
\mathbf{G}_{v,\omega} &= [\mathbf{v}]_\times\mathbf{R},\\
\mathbf{G}_{p,\omega} &= [\mathbf{p}]_\times\mathbf{R},&
\mathbf{G}_{v,a} &= \mathbf{R},\\
\mathbf{G}_{b_\omega,\cdot} &= \mathbf{I}_6,&
\mathbf{G}_{R_{ci},\cdot} &= \mathbf{I}_3,\\
\mathbf{G}_{t_{ci},\cdot} &= \mathbf{I}_3.
\end{aligned}
```

The final four assignments above are a compact description of the code
slices; their exact process-noise-column interpretation must be checked before
porting.

### Generic update

From `utils_numpy_filter.py:239-260`:

```latex
\begin{aligned}
\mathbf{S} &= \mathbf{H}\mathbf{P}\mathbf{H}^{T}+\mathbf{R},\\
\mathbf{K} &= \mathbf{P}\mathbf{H}^{T}\mathbf{S}^{-1},\\
\delta\mathbf{x} &= \mathbf{K}\mathbf{r},\\
\mathbf{P}^{+} &=
(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}
(\mathbf{I}-\mathbf{K}\mathbf{H})^{T}
+\mathbf{K}\mathbf{R}\mathbf{K}^{T}.
\end{aligned}
```

The implementation computes the gain through a linear solve rather than an
explicit inverse. It injects `δx` through `sen3exp` for the first nine
components, adds the 3-vector bias corrections, and uses `so3exp` for the
vehicle-to-IMU orientation correction.

### ZUPT/NHC-like update actually present in code

The reference update is not a generic zero-velocity 3-vector update. It
constructs a two-component residual from body-frame lateral and vertical
velocity:

```latex
\begin{aligned}
\mathbf{R}_{body} &= \mathbf{R}\mathbf{R}_{ci},\\
\mathbf{v}_{imu} &= \mathbf{R}^{T}\mathbf{v},\\
\mathbf{v}_{body} &=
\mathbf{R}_{ci}^{T}\mathbf{v}_{imu}
 +[\mathbf{t}_{ci}]_\times
(\mathbf{u}_{\omega}-\mathbf{b}_{\omega}),\\
\mathbf{r} &= -\mathbf{v}_{body}[1:3],\\
\mathbf{R}_{meas} &= \operatorname{diag}(\texttt{measurement\_cov}).
\end{aligned}
```

The code uses `H` with two rows, `H[:, 3:6] = R_body^T[1:]`, and additional
extrinsic/bias-related blocks; see `utils_numpy_filter.py:213-237`.

This is retained as reference evidence only. The project NHC is instead the
single scalar ENU road-perpendicular measurement specified in
`specs/04_map_matching_nhc.md`.

### SO(3) exponential used by the code

For `\boldsymbol{\phi}\neq0`, `so3exp` implements Rodrigues' formula:

```latex
\begin{aligned}
\theta &= \|\boldsymbol{\phi}\|,\qquad
\mathbf{a}=\boldsymbol{\phi}/\theta,\\
\operatorname{SO3Exp}(\boldsymbol{\phi}) &=
\cos\theta\,\mathbf{I}
+(1-\cos\theta)\mathbf{a}\mathbf{a}^{T}
+\sin\theta[\mathbf{a}]_\times.
\end{aligned}
```

For small angle, the code uses `I + [φ]x`. The left Jacobian used by
`sen3exp` is also implemented in the same source.

### Learned covariance code

From `reference/ai-imu-dr/src/utils_torch_filter.py:47-68`:

```latex
\begin{aligned}
\mathbf{y} &= \operatorname{CovNet}(\mathbf{u}),\\
\mathbf{z} &= \tanh(\operatorname{Linear}(\mathbf{y})),\\
\mathbf{z}_{net} &= \boldsymbol{\beta}_{measurement}
\odot\mathbf{z},\\
\mathbf{R}_{measurements} &=
\mathbf{R}_{measurement,0}\odot10^{\mathbf{z}_{net}}.
\end{aligned}
```

From `utils_torch_filter.py:30-38`:

```latex
\begin{aligned}
\boldsymbol{\alpha}_{0} &= \operatorname{Linear}_{0}(1),&
\boldsymbol{\beta}_{0} &= 10^{\tanh(\boldsymbol{\alpha}_{0})},\\
\boldsymbol{\alpha}_{Q} &= \operatorname{Linear}_{Q}(1),&
\boldsymbol{\beta}_{Q} &= 10^{\tanh(\boldsymbol{\alpha}_{Q})}.
\end{aligned}
```

The exact CNN layer sequence is visible in source but is intentionally not
copied into `specs/03_ml_noise_adaptation.md` until the HUMAN architecture
block is filled.

### Training loss code

From `reference/ai-imu-dr/src/train_torch_filter.py:191-201` and
`251-270`, the executable loss is normalized relative displacement error:

```latex
\begin{aligned}
\Delta\mathbf{p}_{est} &=
\mathbf{R}_{0}^{T}
\left(\mathbf{p}_{end}-\mathbf{p}_{0}\right),\\
\widehat{\Delta\mathbf{p}}_{est} &=
\frac{\Delta\mathbf{p}_{est}}
{\|\Delta\mathbf{p}_{gt}\|},\\
\widehat{\Delta\mathbf{p}}_{gt} &=
\frac{\Delta\mathbf{p}_{gt}}
{\|\Delta\mathbf{p}_{gt}\|},\\
\mathcal{L} &= \operatorname{MSELoss}_{sum}
\left(\widehat{\Delta\mathbf{p}}_{est},
\widehat{\Delta\mathbf{p}}_{gt}\right).
\end{aligned}
```

The reference code computes ground-truth relative displacements at 1 Hz over
distance targets `[100,200,300,400,500,600,700,800]` in
`train_torch_filter.py:25-56`. This is a code observation, not a confirmation
of the paper's stated loss.

## 2. Symbol table and dimensions

The reference implementation's covariance state is 21-dimensional, not the
15-element project state. `P_dim = 21` and `Q_dim = 18` are explicit in
`utils_numpy_filter.py:52-56`.

| Symbol | Meaning in reference code | Units | Shape |
|---|---|---:|---:|
| `\mathbf{R}` | navigation/body rotation matrix | dimensionless | `(3,3)` |
| `\mathbf{v}` | navigation-frame velocity | m/s | `(3,)` |
| `\mathbf{p}` | navigation-frame position | m | `(3,)` |
| `\mathbf{b}_\omega` | gyro bias | rad/s | `(3,)` |
| `\mathbf{b}_a` | accel bias | m/s² | `(3,)` |
| `\mathbf{R}_{ci}` | car-to-IMU rotation | dimensionless | `(3,3)` |
| `\mathbf{t}_{ci}` | car-to-IMU translation | m | `(3,)` |
| `\mathbf{g}` | gravity vector | m/s² | `(3,)` |
| `\mathbf{u}_\omega` | measured angular rate | rad/s | `(3,)` |
| `\mathbf{u}_a` | measured acceleration | m/s² | `(3,)` |
| `\Delta t` | sample interval | s | scalar |
| `\mathbf{P}` | covariance | state² | `(21,21)` |
| `\mathbf{Q}` | process-noise covariance | noise² | `(18,18)` |
| `\mathbf{F}` | continuous/localized transition generator | 1/s | `(21,21)` |
| `\mathbf{G}` | process-noise input matrix | state/noise | `(21,18)` |
| `\mathbf{\Phi}` | discrete transition matrix | dimensionless | `(21,21)` |
| `\mathbf{H}` | measurement Jacobian | residual/state | `(2,21)` |
| `\mathbf{r}` | innovation/residual | velocity units | `(2,)` |
| `\mathbf{R}_{meas}` | measurement covariance | residual² | `(2,2)` |
| `\mathbf{S}` | innovation covariance | residual² | `(2,2)` |
| `\mathbf{K}` | Kalman gain | state/residual | `(21,2)` |
| `\delta\mathbf{x}` | injected error/state increment | mixed | `(21,)` |
| `[\cdot]_\times` | skew-symmetric cross-product matrix | input-dependent | `(3,3)` |

Our project state is specified as 15 elements:
`[\delta\theta,\delta v,\delta p,\delta b_\omega,\delta b_a]`. Therefore:

| Project object | Required project shape |
|---|---:|
| `\delta\mathbf{x}` | `(15,)` |
| `\mathbf{P}` | `(15,15)` |
| `\mathbf{\Phi}` | `(15,15)` |
| `\mathbf{Q}` | ⚠️ CHECK: process-noise dimension is not fixed by the spec |
| `\mathbf{G}` | `(15,L)`, with `L` ⚠️ CHECK |
| GNSS `\mathbf{H}` | `(3,15)` for a 3D ENU position residual |
| GNSS `\mathbf{R}` | `(3,3)` |
| ZUPT `\mathbf{H}` | `(3,15)` for 3D zero velocity, unless reference NHC is retained |
| ZUPT `\mathbf{R}` | `(3,3)` for 3D zero velocity, unless reference NHC is retained |
| Generic `\mathbf{S}` | `(m,m)` for measurement dimension `m` |
| Generic `\mathbf{K}` | `(15,m)` |

The meaning of `δθ` and the order/side of its perturbation are ambiguous
between the project state table and the reference's `SE(3)`/`sen3exp`
injection. **⚠️ CHECK** before mapping any reference block into the 15-state
vector.

## 3. Dimensional consistency

### Reference code

The reference dimensions are internally compatible at the array level:

- `F (21,21)`, `G (21,18)`, `Q (18,18)`, `P (21,21)`.
- `Phi (21,21)` and `P_new (21,21)`.
- `H (2,21)`, `R (2,2)`, `S (2,2)`, `K (21,2)`, `r (2,)`.
- `K r` yields `dx (21,)`.

The code's physical unit consistency has unresolved questions because the
process-noise block interpretation and state ordering are not documented in
the spec. **⚠️ CHECK** the units of each `Q` diagonal and the use of
`G * dt` before porting.

### Project 15-state target

For a measurement dimension `m`, the generic update is dimensionally sound
only with:

```latex
\mathbf{P}_{15\times15},\quad
\mathbf{H}_{m\times15},\quad
\mathbf{R}_{m\times m},\quad
\mathbf{S}_{m\times m},\quad
\mathbf{K}_{15\times m},\quad
\mathbf{r}_{m\times1},\quad
\delta\mathbf{x}_{15\times1}.
```

The project specifications do not yet contain equations for `H`, `Q`, `G`,
GNSS, or ZUPT, so their dimensional and physical consistency cannot be
confirmed beyond these required shapes.

## 4. Project compatibility decision

The implementation shall preserve these project decisions:

| Item | Project decision |
|---|---|
| Error state | 15 elements: `δθ, δv, δp, δb_g, δb_a` |
| Navigation frame | ENU |
| Gravity | `+[0, 0, 9.80665]` m/s², pointing up |
| Rotation | Hamilton `(w,x,y,z)`, body-to-world |
| NHC | One scalar road-perpendicular velocity residual |
| Map matching on Android | Not run; consume accepted match records |

## 5. Paper-versus-code discrepancies

No paper equations were pasted into either spec, and the PDF could not be
text-extracted in the current environment. Consequently, **no
paper-versus-code discrepancy can be verified yet**. Do not resolve any of
the following observations as if they were paper discrepancies:

1. Reference code uses a 21-dimensional covariance state; the project spec
   requires a 15-element error state.
2. Reference code uses `g = [0,0,-9.80665]` in `utils_numpy_filter.py`, while
   the project rules require ENU gravity `+[0,0,9.80665]`.
3. Reference code uses a 2-row lateral/vertical vehicle-velocity update,
   while the project EKF spec calls for GNSS position and a ZUPT placeholder.
4. Reference code models car-to-IMU rotation and translation states; the
   project 15-state table does not include those six states.
5. NumPy and Torch implementations differ in some slice assignments and
   signs: NumPy uses `G[9:15, 6:12] = I6` and
   `H_t_c_i = -skew(t_c_i)`, while Torch uses four separate `G` blocks and
   `H_t_c_i = skew(t_c_i)` (`utils_numpy_filter.py:201-203,225-232`;
   `utils_torch_filter.py:197-200,220-228`). These are implementation
   discrepancies requiring a decision, not silently corrected here.

TODO: Extract the paper equations from `reference/brossard_ai_imu_dr.pdf`,
paste them into the HUMAN blocks, then rerun this table line by line against
the paper and both reference implementations.

## 6. Porting file map

- **Filter core:** `reference/ai-imu-dr/src/utils_numpy_filter.py` —
  port the propagation/update primitives to a 15-state pure-NumPy ENU core,
  remove KITTI/car-extrinsic states, and preserve only equations approved
  after the paper audit.
- **ZUPT/NHC logic:** `reference/ai-imu-dr/src/utils_numpy_filter.py` —
  port the existing body-frame lateral/vertical velocity residual only as
  reference material; replace/adapt it to the project ZUPT/NHC contract and
  IO-VNBD semantics after human equation review.
- **CNN:** `reference/ai-imu-dr/src/utils_torch_filter.py` —
  port the reference covariance-network behavior after the ML architecture
  block is filled; add the explicitly marked vehicle-speed extension head.
- **Training loop:** `reference/ai-imu-dr/src/train_torch_filter.py` —
  port the relative-displacement target preparation, optimizer setup, and
  training flow; replace KITTI sequence handling with IO-VNBD loaders,
  leave-one-driver-out splits, and the project outage metrics.
- **Dataset/IO support:** `reference/ai-imu-dr/src/dataset.py`,
  `reference/ai-imu-dr/src/main_kitti.py`, and `reference/ai-imu-dr/src/utils.py`
  — use only as migration references; replace KITTI/NED/navpy dependencies
  with IO-VNBD loaders and ENU conversion at the IO boundary.
