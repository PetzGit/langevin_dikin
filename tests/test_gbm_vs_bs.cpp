// Monte Carlo harness + exact GBM scheme + European call against the
// closed-form Black-Scholes price (with dividend yield). The GBM log step is
// exact, so there is no discretisation bias: any gap beyond 3 standard errors
// is a bug in the harness, estimator, scheme or payoff, not the step size.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "cldiff/core/rng.hpp"
#include "cldiff/market/implied_vol.hpp"
#include "cldiff/market/market_inputs.hpp"
#include "cldiff/models/gbm/gbm_exact.hpp"
#include "cldiff/pricing/monte_carlo/mc_harness.hpp"
#include "cldiff/products/european_call.hpp"
#include <cmath>
#include <cstddef>

using namespace cldiff;

namespace {

struct McPrice {
  double mean;
  double se;
  std::size_t n_nonfinite;
};

McPrice priceGbmCall(double S0, double K, double T, MarketInputs market,
                     double sigma, int steps, std::size_t n_paths,
                     unsigned long seed) {
  Eigen::VectorXd p(1);
  p << sigma;
  GbmScheme scheme(p, market);
  EuropeanCall call(K, market, T);

  Eigen::VectorXd x0(1);
  x0 << std::log(S0);

  MonteCarloHarness harness(x0, scheme, call, T, steps, n_paths);
  Rng rng(seed);
  const MonteCarloResult res = harness.run(rng);
  return {res.outputs.at(0).getMean(), res.outputs.at(0).getStdError(),
          res.n_nonfinite};
}

}  // namespace

TEST_CASE("GBM Monte Carlo matches Black-Scholes within 3 SE", "[mc][gbm]") {
  const double S0 = 100.0, T = 1.0, sigma = 0.20;
  const MarketInputs market{0.03, 0.01};  // r, q
  const std::size_t n_paths = 200000;

  const double K = GENERATE(80.0, 100.0, 120.0);  // ITM, ATM, OTM
  const double bs = BlackScholesImpliedVol(S0, K, T, market.r, market.q).price(sigma);

  // One step: the scheme is exact, so this is the full transition.
  const McPrice mc = priceGbmCall(S0, K, T, market, sigma, 1, n_paths, 42);
  const double z = (mc.mean - bs) / mc.se;
  CAPTURE(K, bs, mc.mean, mc.se, z);

  REQUIRE(mc.n_nonfinite == 0);
  REQUIRE(std::isfinite(mc.se));
  REQUIRE(std::abs(z) <= 3.0);
}

TEST_CASE("GBM Monte Carlo is step-count invariant", "[mc][gbm]") {
  // Exact scheme: 50 steps must also agree with Black-Scholes. A failure here
  // with the one-step test passing points at the step loop or dt, not the payoff.
  const double S0 = 100.0, K = 100.0, T = 1.0, sigma = 0.20;
  const MarketInputs market{0.03, 0.01};
  const double bs = BlackScholesImpliedVol(S0, K, T, market.r, market.q).price(sigma);

  const McPrice mc = priceGbmCall(S0, K, T, market, sigma, 50, 200000, 7);
  const double z = (mc.mean - bs) / mc.se;
  CAPTURE(bs, mc.mean, mc.se, z);

  REQUIRE(mc.n_nonfinite == 0);
  REQUIRE(std::abs(z) <= 3.0);
}

TEST_CASE("GBM scheme rejects invalid sigma", "[mc][gbm]") {
  const MarketInputs market{0.03, 0.01};
  Eigen::VectorXd p(1);

  p << 0.0;
  REQUIRE_THROWS_AS(GbmScheme(p, market), std::invalid_argument);
  p << -0.2;
  REQUIRE_THROWS_AS(GbmScheme(p, market), std::invalid_argument);
  p << std::nan("");
  REQUIRE_THROWS_AS(GbmScheme(p, market), std::invalid_argument);

  Eigen::VectorXd p2(2);
  p2 << 0.2, 0.3;
  REQUIRE_THROWS_AS(GbmScheme(p2, market), std::invalid_argument);
}
