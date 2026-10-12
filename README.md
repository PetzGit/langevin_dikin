# cldiff

Barrier-Hessian ("Dikin metric") Langevin sampling and optimisation on
polyhedra, with a Heston stochastic-volatility calibration pipeline and a
Monte Carlo pricing engine built on top of it. C++17, Eigen, CMake.

Implements the method of Chok & Petzinna (2026), *Constrained Dikin–Langevin
diffusion for polyhedra* -- see [Reference](#reference). The kinetic
(AKDL) sampler and optimiser come from follow-up work in progress and are
included as experimental code.

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
    polyhedron, tempered_objective, log_space_objective
    geometry/                    kinetic_dikin (QR geometry + integrator for AKDL)
    optimisers/                  annealed_langevin, annealed_akdl, projected_gradient_descent
    samplers/                    dikin_langevin, akdl
    schedules/                   three_phase_log, parallel_floor
  market/                        Quote, MarketData, implied_vol (Black-Scholes IV inversion)
  models/                        one folder per model, holding its plug-ins for each method
    heston/                      characteristic_function, cumulants, feller,
                                 heston_cos_pricer, full_truncation_scheme (MC)
    gbm/                         gbm_exact (MC)
  products/                      Observable (payoff/path functional)          <- interface
                                 european_call
  pricing/
    pricer.hpp                   Pricer::price(params, market)                <- interface
    fourier/                     cos_pricer (generic COS engine)
    monte_carlo/                 SDEScheme                                    <- interface
                                 mc_harness (MonteCarloHarness)
    pde/                         grid (in progress)
  calibration/                   where pricing meets optim
    calibration_objective.hpp    CalibrationObjective(Pricer, MarketData, QuoteLoss, ParameterPenalty*)
    quote_loss.hpp               QuoteLoss                                    <- interface
    parameter_penalty.hpp        ParameterPenalty                             <- interface
    implied_vol_squared_loss, feller_penalty,
    heston_objective, corrected_log_space_heston_objective
experiments/                     buildable studies, one executable each (see below)
tools/data_loader/               yfinance -> CSV pipeline for real SPX option-chain data
tests/                           Catch2 unit tests (ctest)
```

Everything lives in `namespace cldiff` (the AKDL geometry in
`cldiff::kinetic_dikin`). `optim/` depends only on `core/` and never includes
a finance header; models implement the interfaces in `pricing/` and
`products/`; pricers never see an optimiser. `calibration/` is the only
place finance meets optim.

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
- **AKDLSampler** (experimental): Metropolis-adjusted kinetic Dikin-Langevin
  (persistent momentum), with optional Robbins-Monro step-size warmup
  (`AKDLWarmup`).
- **AnnealedLangevinOptimiser**: one Dikin-Langevin chain per column of `x0`, each at the
  temperature its `AnnealingSchedule` assigns, with periodic resampling (every
  `r_rate` steps) toward the best-performing chains. Returns the best point
  of the final population (`OptimResult`).
- **AnnealedAKDLOptimiser** (experimental): the same annealing/resampling
  scheme over AKDL chains, annealing the objective itself (`U/beta`).
- **ProjectedGradientDescent**: deterministic backtracking gradient descent
  that rejects steps leaving the polyhedron. Serves as the local baseline and
  as the "polish" run from another optimiser's result.
- **Schedules**: `ThreePhaseLogSchedule(rate)` is the original three-phase
  cooling (`rate` = `h` or `epsilon_base`). `ParallelFloorSchedule(base,
  floor_max)` wraps any schedule with per-chain temperature floors so some
  chains stay warm (explorers) while others fully cool (refiners). Any
  schedule works with any annealed optimiser.

Demos: `experiments/run_box_gaussian.cpp` (sampler on a Gaussian target),
`experiments/run_optimiser_nonconvex.cpp` (optimiser on the paper's nonconvex
benchmark).

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
- **market/implied_vol.hpp**: `BlackScholesImpliedVol`, Black-Scholes price/vega/implied-vol
  (Newton solve), with a dividend yield `q` so forwards aren't biased when
  pricing an index. Calibration fits in implied-vol space rather than price
  space, so this is the quoting convention rather than a pricing model.
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
  relaxation of the Feller penalty. It is smoothed because a
  finite-difference stencil straddling a hard `max(0, x)` kink can flip the
  gradient's sign, and calibrated fits often sit on the Feller boundary.
- **calibration/corrected_log_space_heston_objective.hpp**: `CorrectedLogSpaceHestonObjective`
  -- the generic `optim/LogSpaceObjective` with Heston's mask, wrapping `HestonObjective` in a log-space reparameterization (kappa, theta,
  xi, v0 exp-transformed to stay positive; rho passed through) with the
  chain-rule scaling divided back out. Without the correction the gradient
  shrinks in proportion to each parameter's value, so parameters near zero
  stop moving and calibration collapses to the box boundary. **Use this
  class, not the raw `HestonObjective`, when calibrating.**

### Monte Carlo

`MonteCarloHarness(x0, scheme, observable, T, steps, n_paths)` simulates
paths with any `SDEScheme` and feeds them to any `Observable`, returning a
running mean and standard error per output (`Estimator`). Paths with
non-finite results are counted and skipped.

- **Schemes**: `GbmScheme` (exact log-normal step) and `HestonScheme`
  (full-truncation Euler on `(ln S, v)`).
- **Products**: `EuropeanCall` (discounted payoff at maturity). Observables
  can also opt in to per-step callbacks (`needsSteps()` / `onStep`) for
  path-dependent payoffs.

### Experiments

- `experiments/heston/multistart_calibration.cpp` (target `heston_multistart_calibration`):
  GD alone vs annealed optimiser vs optimiser + GD polish, on synthetic data
  from several starting points.
- `experiments/heston/harness.cpp` (target `heston_reference_recovery`):
  reference validation run -- fits synthetic data (known true params) from
  two starting points, one of them adversarial, using three arms (GD alone,
  annealed Optimiser, Optimiser + GD polish) and scores out-of-sample on
  held-out strikes. Demonstrates the Optimiser reaching basins plain GD
  cannot, and that the corrected objective recovers the true parameters
  rather than collapsing to the box boundary.
- `experiments/heston/spx_calibration.cpp` (target `heston_spx_calibration`):
  the same corrected-objective + Optimiser + polish pipeline run against a
  real SPX snapshot from `tools/data_loader/`. Run from `tools/data_loader/` so
  `dat.csv`/`meta.txt` resolve from the working directory.

## tools/data_loader

`data_loader.py` pulls a liquid SPX option chain via `yfinance`, estimates
the risk-free rate (`^IRX`) and dividend yield (put-call parity on near-money
strikes), filters for liquidity (open interest + bid-ask spread), and writes
`dat.csv` (`T K price` rows) and `meta.txt` (`S0 r q`) for
`heston_spx_calibration` to consume.

## Tests

`tests/` holds Catch2 unit tests, run with `ctest` from the build directory:

- Black-Scholes implied-vol round trip
- COS pricer against the Fang & Oosterlee (2008) benchmark
- `HestonObjective` around known true parameters
- `Estimator` moments and numerical stability
- GBM Monte Carlo against the Black-Scholes closed form
- AKDL smoke tests: QR geometry, sampler feasibility and reproducibility,
  warmup, and the annealed optimiser reaching an interior minimum

## Build

```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
ctest
```

Fetches Eigen 3.4.0 and Catch2 at configure time. All targets link against
the `cldiff` static library, which is built with `-Wall -Wextra -Werror`.
Without `CMAKE_BUILD_TYPE` CMake builds unoptimised, and the experiments run
several times slower.

## Roadmap

**Done**
- Dikin-Langevin sampler and annealed optimiser on polyhedra
- Heston calibration via the COS method, on synthetic data and a real SPX snapshot
- Monte Carlo harness with exact GBM and full-truncation Heston schemes

**In progress**
- Heston Monte Carlo validated against COS prices
- Parallel annealed optimiser (thread pool, results identical for any thread count)

**Planned**
- American options by Monte Carlo (Longstaff-Schwartz)
- PDE pricer (Crank-Nicolson on a sinh-stretched grid)
- Black-Scholes `Pricer`
- Risk: exposure profiles and PFE, VaR/CVaR, Kupiec and Christoffersen back-tests
- Parallel Monte Carlo paths and scenario revaluation, on the same thread pool
- Make the COS engine model-agnostic (it currently includes Heston headers)

## Reference

Chok, J. & Petzinna, D. (2026). Constrained Dikin–Langevin diffusion for
polyhedra. *IMA Journal of Applied Mathematics*, 91(2), 210-228.
https://doi.org/10.1093/imamat/hxag012
