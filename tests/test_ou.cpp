// OuScheme: exact one-factor Ornstein-Uhlenbeck step on the log price,
//   dx = kappa (mu - x) dt + sigma dW.
//
// Exact solution:
//   x_t ~ Normal( mu + (x0 - mu) e^{-kappa t},  sigma^2 (1 - e^{-2 kappa t}) / (2 kappa) )
//
// Because the step is exact, these moments hold for ANY step size. A mismatch
// is a bug in the scheme (or the harness), never discretisation error.
//
// What is checked:
//   1. Mean and variance of the stored state at every time match theory.
//   2. One big step and many small steps give the same terminal distribution.
//   3. Long-time limit: the state settles to Normal(mu, sigma^2 / (2 kappa)).
//   4. With negligible noise the step is the deterministic decay towards mu.
//   5. kappa -> 0 recovers Brownian motion without blowing up.
//   6. Parameter validation, and a failed setParameters leaves the scheme intact.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cldiff/core/rng.hpp"
#include "cldiff/models/ou/ou_exact.hpp"
#include "cldiff/pricing/monte_carlo/mc_harness.hpp"
#include "cldiff/products/observable.hpp"
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

using namespace cldiff;
using Catch::Approx;

namespace {

class TerminalState final : public Observable {
 public:
  int numOutputs() const override { return 1; }
  Eigen::VectorXd result(const Eigen::VectorXd& s) const override {
    Eigen::VectorXd o(1);
    o << s(0);
    return o;
  }
};

Eigen::VectorXd ouParams(double kappa, double mu, double sigma) {
  Eigen::VectorXd p(3);
  p << kappa, mu, sigma;
  return p;
}

double expMean(double x0, double kappa, double mu, double t) {
  return mu + (x0 - mu) * std::exp(-kappa * t);
}

double expVar(double kappa, double sigma, double t) {
  return sigma * sigma * (-std::expm1(-2.0 * kappa * t)) / (2.0 * kappa);
}

double sampleVar(const Eigen::VectorXd& col) {
  const double m = col.mean();
  return (col.array() - m).square().sum() / (static_cast<double>(col.size()) - 1.0);
}

// Terminal ln-price samples from the harness, one step count.
Eigen::VectorXd terminalSamples(const Eigen::VectorXd& p, double x0, double T,
                                int steps, std::size_t n, unsigned long seed) {
  const OuScheme scheme(p);
  TerminalState obs;
  Eigen::VectorXd x(1);
  x << x0;
  MonteCarloHarness harness(x, scheme, obs, T, steps, n);
  PathStore store;
  Rng rng(seed);
  harness.run(rng, &store);
  return store.states.back().col(0);
}

}  // namespace

TEST_CASE("OU stored states have the exact mean and variance at every time",
          "[mc][ou]") {
  const double kappa = 1.5, mu = std::log(60.0), sigma = 0.5;
  const double x0 = std::log(80.0), T = 1.0;
  const int steps = 8;
  const std::size_t n = 100000;
  const double N = static_cast<double>(n);

  const OuScheme scheme(ouParams(kappa, mu, sigma));
  TerminalState obs;
  Eigen::VectorXd x(1);
  x << x0;
  MonteCarloHarness harness(x, scheme, obs, T, steps, n);
  PathStore store;
  Rng rng(2026);
  harness.run(rng, &store);

  for (int j = 1; j <= steps; ++j) {
    const double t = j * T / steps;
    const Eigen::VectorXd col = store.states[static_cast<std::size_t>(j)].col(0);
    const double em = expMean(x0, kappa, mu, t);
    const double ev = expVar(kappa, sigma, t);
    const double se = std::sqrt(ev / N);
    const double rel = 5.0 * std::sqrt(2.0 / (N - 1.0));
    CAPTURE(j, t, col.mean(), em, se, sampleVar(col), ev);
    REQUIRE(std::abs(col.mean() - em) <= 4.0 * se);
    REQUIRE(std::abs(sampleVar(col) / ev - 1.0) <= rel);
  }
}

TEST_CASE("OU: one big step and many small steps agree", "[mc][ou]") {
  const double kappa = 2.0, mu = 4.0, sigma = 0.7, x0 = 3.0, T = 1.5;
  const std::size_t n = 100000;
  const double N = static_cast<double>(n);
  const double em = expMean(x0, kappa, mu, T);
  const double ev = expVar(kappa, sigma, T);
  const double se = std::sqrt(ev / N);
  const double rel = 5.0 * std::sqrt(2.0 / (N - 1.0));

  for (int steps : {1, 64}) {
    const Eigen::VectorXd s = terminalSamples(ouParams(kappa, mu, sigma), x0, T,
                                              steps, n, 11 + steps);
    CAPTURE(steps, s.mean(), em, sampleVar(s), ev);
    REQUIRE(std::abs(s.mean() - em) <= 4.0 * se);
    REQUIRE(std::abs(sampleVar(s) / ev - 1.0) <= rel);
  }
}

TEST_CASE("OU: long-time limit is Normal(mu, sigma^2 / (2 kappa))", "[mc][ou]") {
  const double kappa = 1.5, mu = 3.2, sigma = 0.4, x0 = 6.0, T = 10.0;
  const std::size_t n = 100000;
  const double N = static_cast<double>(n);
  const double ev = sigma * sigma / (2.0 * kappa);

  const Eigen::VectorXd s =
      terminalSamples(ouParams(kappa, mu, sigma), x0, T, 1, n, 77);
  CAPTURE(s.mean(), sampleVar(s), ev);
  REQUIRE(std::abs(s.mean() - mu) <= 4.0 * std::sqrt(ev / N));
  REQUIRE(std::abs(sampleVar(s) / ev - 1.0) <= 5.0 * std::sqrt(2.0 / (N - 1.0)));
}

TEST_CASE("OU: with negligible noise the step is deterministic decay", "[ou]") {
  const double kappa = 0.8, mu = 2.0, x0 = 5.0, dt = 0.25;
  const OuScheme scheme(ouParams(kappa, mu, 1e-12));
  Eigen::VectorXd x(1);
  x << x0;
  Rng rng(1);
  scheme.step(x, dt, rng);
  REQUIRE(x(0) == Approx(mu + (x0 - mu) * std::exp(-kappa * dt)).margin(1e-9));
}

TEST_CASE("OU: tiny kappa recovers Brownian motion", "[mc][ou]") {
  const double kappa = 1e-9, mu = 0.0, sigma = 0.3, x0 = 1.0, T = 1.0;
  const std::size_t n = 100000;
  const double N = static_cast<double>(n);

  const Eigen::VectorXd s =
      terminalSamples(ouParams(kappa, mu, sigma), x0, T, 4, n, 5);
  REQUIRE(s.allFinite());
  CAPTURE(s.mean(), sampleVar(s));
  REQUIRE(std::abs(s.mean() - x0) <= 4.0 * sigma * std::sqrt(T / N));
  REQUIRE(std::abs(sampleVar(s) / (sigma * sigma * T) - 1.0) <=
          5.0 * std::sqrt(2.0 / (N - 1.0)));
}

TEST_CASE("OU parameter validation", "[ou]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  REQUIRE_NOTHROW(OuScheme(ouParams(1.0, 0.0, 0.3)));
  REQUIRE_NOTHROW(OuScheme(ouParams(1.0, -2.5, 0.3)));  // mu may be negative

  REQUIRE_THROWS_AS(OuScheme(ouParams(0.0, 0.0, 0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(-1.0, 0.0, 0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(nan, 0.0, 0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(1.0, nan, 0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(1.0, inf, 0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(1.0, 0.0, 0.0)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(1.0, 0.0, -0.3)), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(ouParams(1.0, 0.0, nan)), std::invalid_argument);

  Eigen::VectorXd two(2), four(4);
  two << 1.0, 0.0;
  four << 1.0, 0.0, 0.3, 0.1;
  REQUIRE_THROWS_AS(OuScheme(two), std::invalid_argument);
  REQUIRE_THROWS_AS(OuScheme(four), std::invalid_argument);

  // A rejected setParameters must leave the scheme unchanged.
  OuScheme scheme(ouParams(1.0, 0.5, 0.3));
  REQUIRE_THROWS_AS(scheme.setParameters(ouParams(-1.0, 0.5, 0.3)),
                    std::invalid_argument);
  const Eigen::VectorXd p = scheme.parameters();
  REQUIRE(p.size() == 3);
  REQUIRE(p(0) == 1.0);
  REQUIRE(p(1) == 0.5);
  REQUIRE(p(2) == 0.3);

  REQUIRE(scheme.dimension() == 1);
  REQUIRE(scheme.numParameters() == 3);
}
