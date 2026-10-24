// Heston Monte Carlo (full-truncation Euler) against the COS price.
//
// COS is the reference: it is deterministic and converges geometrically, so any
// gap between it and the Monte Carlo price is either sampling noise (reported
// by the standard error), Euler discretisation bias (shrinks with the step
// size), or a bug. The tests are built to tell those apart:
//
//   1. Benign parameters, moderate step count: bias is far below the SE, so the
//      MC price must sit within 3 SE of COS. A failure here is a bug, not bias.
//   2. Feller-violating parameters: Euler is visibly biased at coarse steps.
//      The test asserts the bias is real at 8 steps, shrinks at roughly first
//      order as dt falls, and is inside 3 SE at the finest step count.
//   3. Input validation on the scheme and harness.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "cldiff/core/rng.hpp"
#include "cldiff/market/market_inputs.hpp"
#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/models/heston/full_truncation_scheme.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"
#include "cldiff/pricing/monte_carlo/mc_harness.hpp"
#include "cldiff/products/european_call.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <stdexcept>

using namespace cldiff;

namespace {

// (kappa, theta, xi, rho, v0): the order HestonCosPricer and the calibration use.
Eigen::VectorXd fullParams(double kappa, double theta, double xi, double rho, double v0) {
  Eigen::VectorXd p(5);
  p << kappa, theta, xi, rho, v0;
  return p;
}

// The scheme takes (kappa, theta, xi, rho); v0 lives in the initial state.
Eigen::VectorXd schemeParams(const Eigen::VectorXd& p5) { return p5.head(4); }

double cosCall(const Eigen::VectorXd& p5, double S0, double K, double T, MarketInputs m,
               int terms) {
  const CharacteristicHeston phi(p5);
  Cumulants cum = cumulants(phi, T);
  const Range range = make_range(cum, 10.0);
  return cos_pricer(phi, range, K, T, m.r, terms, S0, m.q);
}

struct McPrice {
  double mean;
  double se;
  std::size_t n_nonfinite;
};

McPrice mcCall(const Eigen::VectorXd& p5, double S0, double K, double T, MarketInputs m,
               int steps, std::size_t n_paths, unsigned long seed) {
  const HestonScheme scheme(schemeParams(p5), m);
  EuropeanCall call(K, m, T);
  Eigen::VectorXd x0(2);
  x0 << std::log(S0), p5(4);  // state = (ln S, v)
  MonteCarloHarness harness(x0, scheme, call, T, steps, n_paths);
  Rng rng(seed);
  const MonteCarloResult res = harness.run(rng);
  return {res.outputs.at(0).getMean(), res.outputs.at(0).getStdError(), res.n_nonfinite};
}

}  // namespace

TEST_CASE("Heston MC matches COS within 3 SE (benign parameters)", "[mc][heston]") {
  const double S0 = 100.0, T = 1.0;
  const MarketInputs market{0.03, 0.01};
  const Eigen::VectorXd p5 = fullParams(2.0, 0.04, 0.30, -0.70, 0.04);

  const double K = GENERATE(90.0, 100.0, 110.0);

  // The reference itself must be converged before it is used to judge MC.
  const double ref = cosCall(p5, S0, K, T, market, 256);
  REQUIRE(std::abs(ref - cosCall(p5, S0, K, T, market, 128)) < 1e-5);

  const McPrice mc = mcCall(p5, S0, K, T, market, 64, 200000, 20261);
  const double z = (mc.mean - ref) / mc.se;
  CAPTURE(K, ref, mc.mean, mc.se, z);

  REQUIRE(mc.n_nonfinite == 0);
  REQUIRE(std::isfinite(mc.se));
  REQUIRE(std::abs(z) <= 3.0);
}

TEST_CASE("Euler bias shrinks at first order on a Feller-violating set", "[mc][heston]") {
  // 2*kappa*theta = 0.04 < xi^2 = 0.64: the variance hits zero often, which is
  // where full-truncation Euler is least accurate.
  const double S0 = 100.0, K = 100.0, T = 1.0;
  const MarketInputs market{0.03, 0.01};
  const Eigen::VectorXd p5 = fullParams(0.5, 0.04, 0.80, -0.70, 0.04);
  REQUIRE(2.0 * p5(0) * p5(1) < p5(2) * p5(2));

  const double ref = cosCall(p5, S0, K, T, market, 256);
  REQUIRE(std::abs(ref - cosCall(p5, S0, K, T, market, 128)) < 1e-5);

  const std::size_t n_paths = 300000;
  const McPrice c8   = mcCall(p5, S0, K, T, market, 8,   n_paths, 101);
  const McPrice c32  = mcCall(p5, S0, K, T, market, 32,  n_paths, 102);
  const McPrice c128 = mcCall(p5, S0, K, T, market, 128, n_paths, 103);
  const McPrice c512 = mcCall(p5, S0, K, T, market, 512, n_paths, 104);

  const double b8 = c8.mean - ref, b32 = c32.mean - ref;
  const double b128 = c128.mean - ref, b512 = c512.mean - ref;
  CAPTURE(ref, b8, b32, b128, b512, c8.se, c128.se, c512.se);

  REQUIRE(c8.n_nonfinite + c32.n_nonfinite + c128.n_nonfinite + c512.n_nonfinite == 0);

  // The bias at coarse steps is real, not noise: many standard errors out.
  REQUIRE(b8 / c8.se > 10.0);
  REQUIRE(b32 / c32.se > 5.0);

  // It shrinks monotonically as the step count rises.
  REQUIRE(b8 > b32);
  REQUIRE(b32 > b128);
  REQUIRE(b128 > 0.0);

  // Log-log slope of bias against dt (steps grow by 4x each time) is about 1.
  // Euler is first order; the band allows for sampling noise in b128.
  const double slope_8_32 = std::log(b8 / b32) / std::log(4.0);
  const double slope_32_128 = std::log(b32 / b128) / std::log(4.0);
  CAPTURE(slope_8_32, slope_32_128);
  REQUIRE(slope_8_32 > 0.7);
  REQUIRE(slope_8_32 < 1.5);
  REQUIRE(slope_32_128 > 0.7);
  REQUIRE(slope_32_128 < 1.6);

  // At the finest step count the discretisation bias is below the noise floor.
  REQUIRE(std::abs(b512 / c512.se) <= 3.0);
}

TEST_CASE("HestonScheme and harness reject invalid inputs", "[mc][heston]") {
  const MarketInputs market{0.03, 0.01};
  Eigen::VectorXd p(4);

  p << 2.0, 0.04, 0.30, 1.0;     // |rho| must be < 1
  REQUIRE_THROWS_AS(HestonScheme(p, market), std::invalid_argument);
  p << 2.0, 0.04, 0.30, -1.0;
  REQUIRE_THROWS_AS(HestonScheme(p, market), std::invalid_argument);
  p << -2.0, 0.04, 0.30, -0.7;   // kappa > 0
  REQUIRE_THROWS_AS(HestonScheme(p, market), std::invalid_argument);
  p << 2.0, 0.0, 0.30, -0.7;     // theta > 0
  REQUIRE_THROWS_AS(HestonScheme(p, market), std::invalid_argument);
  p << 2.0, 0.04, 0.0, -0.7;     // xi > 0
  REQUIRE_THROWS_AS(HestonScheme(p, market), std::invalid_argument);

  Eigen::VectorXd p3(3);
  p3 << 2.0, 0.04, 0.30;
  REQUIRE_THROWS_AS(HestonScheme(p3, market), std::invalid_argument);

  // Harness: initial state must match the scheme dimension (2 for Heston).
  Eigen::VectorXd good(4);
  good << 2.0, 0.04, 0.30, -0.7;
  const HestonScheme scheme(good, market);
  EuropeanCall call(100.0, market, 1.0);
  Eigen::VectorXd x0_wrong(1);
  x0_wrong << std::log(100.0);
  REQUIRE_THROWS_AS(MonteCarloHarness(x0_wrong, scheme, call, 1.0, 10, 100), std::invalid_argument);

  Eigen::VectorXd x0(2);
  x0 << std::log(100.0), 0.04;
  REQUIRE_THROWS_AS(MonteCarloHarness(x0, scheme, call, 0.0, 10, 100), std::invalid_argument);
  REQUIRE_THROWS_AS(MonteCarloHarness(x0, scheme, call, 1.0, 0, 100), std::invalid_argument);
}
