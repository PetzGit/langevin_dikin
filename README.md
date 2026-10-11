# cldiff

Barrier-Hessian ("Dikin metric") Langevin sampling and optimization on
polyhedra, plus a Heston stochastic-volatility calibration pipeline built on
top of it as an application. C++17, Eigen-based, CMake build.

Implements the method of Chok & Petzinna (2026), *Constrained Dikin–Langevin
diffusion for polyhedra* -- see [Reference](#reference) below.

## Layout

Organised by layer, not by model. Each folder's abstract interface sits at
its root; concrete implementations sit beside it, so anything with the same
interface plugs in anywhere it is used.

```
include/cldiff/                  public headers (src/ mirrors this)
  core/                          rng, estimator (running mean/variance)
  optim/                         generic optimisation/sampling -- no finance in here
    objective.hpp                Objective: value()/gradient()/dim()          <- interface
    problem.hpp                  Problem { Objective&, Polyhedron& }
    optimiser.hpp                Optimiser::minimise(problem, x0, rng)        <- interface
    sampler.hpp                  Sampler::sample(problem, x0, rng, n)         <- interface
    annealing_schedule.hpp       AnnealingSchedule::beta(step, n, chain, m)   <- interface
    polyhedron, tempered_objective
    geometry/                    kinetic_dikin (QR geometry + integrator for AKDL)
    optimisers/                  annealed_langevin, annealed_akdl
    samplers/                    dikin_langevin, akdl
    schedules/                   three_phase_log, parallel_floor
  market/                        Quote, MarketData
  models/                        one folder per model, holding its plug-ins for each method
    black_scholes/               closed-form price/vega/implied vol
    heston/                      characteristic_function, cumulants, feller,
                                 heston_cos_pricer, full_truncation_scheme (MC)
  products/                      Observable (payoff/path functional)          <- interface
                                 european_call
  pricing/
    pricer.hpp                   Pricer::price(params, market)                <- interface
    fourier/                     cos_pricer (generic COS engine)
    monte_carlo/                 SDEScheme                                    <- interface
    pde/                         grid (in progress)
  calibration/                   where pricing meets optim
    calibration_objective.hpp    CalibrationObjective(Pricer, MarketData, QuoteLoss, ParameterPenalty*)
    quote_loss.hpp               QuoteLoss                                    <- interface
    parameter_penalty.hpp        ParameterPenalty                             <- interface
    implied_vol_squared_loss, feller_penalty,
    heston_objective, corrected_log_space_heston_objective
experiments/                     buildable studies, one executable each (see below)
data_loader/                     yfinance -> CSV pipeline for real SPX option-chain data
tests/                           Catch2 scaffolding (not currently wired into the build)
```

Dependencies point one way: `core <- optim <- calibration -> pricing ->
models, products, market -> core`. `optim/` never includes a finance header
and pricers never see an optimiser; `calibration/` is the only bridge.

## optim: Polyhedron / samplers / optimisers

Optimisers and samplers take their hyperparameters at construction and the
problem per call, so one configured algorithm runs against many problems and
one problem against many algorithms:

```cpp
ThreePhaseLogSchedule schedule(h);
AnnealedLangevinOptimiser optimiser(schedule, r_rate, run_num, h, eps);
OptimResult result = optimiser.minimise({objective, polyhedron}, x0, rng);  // x0: d x chain_num
```

- **Polyhedron**: a convex region `{x : Ax <= b}`. Provides `contains()` and
  the log-barrier Hessian (`barrierHessian`) that gives the Dikin ellipsoid at
  a point `x` -- the local metric the sampler/optimiser use to stay inside
  the polytope without ever evaluating a hard constraint mid-step.
- **DikinLangevinSampler**: single-chain Dikin-Langevin sampler. Proposes moves scaled by
  the local barrier geometry, so step size naturally shrinks near the
  boundary and grows in the interior.
- **AKDLSampler**: Metropolis-adjusted kinetic Dikin-Langevin (persistent
  momentum), with optional Robbins-Monro step-size warmup (`AKDLWarmup`).
- **AnnealedLangevinOptimiser**: one Dikin-Langevin chain per column of `x0`, each at the
  temperature its `AnnealingSchedule` assigns, with periodic resampling (every
  `r_rate` steps) toward the best-performing chains. Returns the best point
  of the final population (`OptimResult`).
- **AnnealedAKDLOptimiser**: the same annealing/resampling scheme over AKDL chains,
  annealing the objective itself (`U/beta`).
- **Schedules**: `ThreePhaseLogSchedule(rate)` is the original three-phase
  cooling (`rate` = `h` or `epsilon_base`). `ParallelFloorSchedule(base,
  floor_max)` wraps any schedule with per-chain temperature floors so some
  chains stay warm (explorers) while others fully cool (refiners). Any
  schedule works with any annealed optimiser.

Demos: `experiments/run_box_gaussian.cpp` (sampler on a Gaussian target),
`experiments/run_optimiser_nonconvex.cpp` (optimiser on a nonconvex benchmark --
the sanity check that the core algorithm works before it's pointed at
anything financial).

## Models, pricing and calibration

Calibrates the Heston stochastic-volatility model to a set of
(maturity, strike, price) option quotes.

- **models/heston/characteristic_function.hpp**: `CharacteristicHeston`, the Heston
  characteristic function.
- **models/heston/cumulants.hpp, pricing/fourier/cos_pricer.hpp**: cumulant-based truncation range and the
  COS-method Fourier pricer (Fang & Oosterlee 2008). `make_range` clamps the
  truncation half-width to avoid catastrophic cancellation at extreme
  parameter draws.
- **models/heston/heston_cos_pricer.hpp**: `HestonCosPricer`, the Heston x COS `Pricer`.
- **models/black_scholes/black_scholes.hpp**: Black-Scholes price/vega/implied-vol (Newton solve),
  with a dividend yield `q` so forwards aren't biased when pricing an index.
- **market/market_data.hpp**: `MarketData` holds quotes by maturity and converts
  each to an implied vol on load; `loadFromCsv` reads whitespace-separated
  `T K price [weight]` rows.
- **models/heston/feller.hpp**: checks the Feller condition `2*kappa*theta >= xi^2`.
- **calibration/calibration_objective.hpp**: `CalibrationObjective` -- any
  `Pricer`, summed `QuoteLoss` over the quotes, plus an optional
  `ParameterPenalty`, exposed as an `Objective` (central finite-difference
  gradient).
- **calibration/heston_objective.hpp**: `HestonObjective` -- the standard
  assembly: `HestonCosPricer` + `ImpliedVolSquaredLoss` (sum of
  squared implied-vol errors across all quotes) + `FellerPenalty`, a smooth softplus
  relaxation of the Feller penalty. The penalty is smoothed rather than a
  hard `max(0, x)` because a finite-difference gradient stencil straddling a
  hard kink can be sign-flipped -- real calibrated fits often sit right at
  the Feller boundary, so this isn't a rare edge case.
- **calibration/corrected_log_space_heston_objective.hpp**: `CorrectedLogSpaceHestonObjective`
  -- wraps `HestonObjective` in a log-space reparameterization (kappa, theta,
  xi, v0 exp-transformed to stay positive; rho passed through) with the
  chain-rule scaling divided back out. Without this correction the gradient
  the optimizer sees shrinks in proportion to a parameter's own value, so
  parameters near zero stop generating signal even when the true physical
  sensitivity is still large -- this is the fix that makes calibration
  actually converge instead of collapsing to the box boundary. **Use this
  class, not the raw `HestonObjective`, when calibrating.**

### Experiments

- `experiments/test_pricer.cpp`, `test_black_scholes.cpp`, `test_heston_objective.cpp`:
  unit-style sanity checks for the pricer, Black-Scholes, and objective in
  isolation.
- `experiments/test_heston_calibration.cpp`: calibration on synthetic data.
- `experiments/heston/harness.cpp` (target `heston_reference_recovery`):
  reference validation run -- fits synthetic data (known true params) from
  two starting points, one of them adversarial, using three arms (GD alone,
  annealed Optimiser, Optimiser + GD polish) and scores out-of-sample on
  held-out strikes. Demonstrates the Optimiser reaching basins plain GD
  cannot, and that the corrected objective recovers the true parameters
  rather than collapsing to the box boundary.
- `experiments/heston/spx_calibration.cpp` (target `heston_spx_calibration`):
  the same corrected-objective + Optimiser + polish pipeline run against a
  real SPX snapshot from `data_loader/`. Run from `data_loader/` so
  `dat.csv`/`meta.txt` resolve from the working directory.

## data_loader

`data_loader.py` pulls a liquid SPX option chain via `yfinance`, estimates
the risk-free rate (`^IRX`) and dividend yield (put-call parity on near-money
strikes), filters for liquidity (open interest + bid-ask spread), and writes
`dat.csv` (`T K price` rows) and `meta.txt` (`S0 r q`) for
`heston_spx_calibration` to consume.

## Build

```
mkdir -p build && cd build
cmake ..
make -j
```

Fetches Eigen 3.4.0 and Catch2 at configure time. All targets link against
the `cldiff` static library, built with `-Wall -Wextra -Werror`.

## Reference

Chok, J. & Petzinna, D. (2026). Constrained Dikin–Langevin diffusion for
polyhedra. *IMA Journal of Applied Mathematics*, 91(2), 210-228.
https://doi.org/10.1093/imamat/hxag012
