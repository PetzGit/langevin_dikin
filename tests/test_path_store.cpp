// PathStore: the optional per-path state storage in MonteCarloHarness::run.
//
// GBM is the test model because its log-state moments are known exactly:
//   ln S_t ~ Normal(x0 + (r - q - sigma^2/2) t,  sigma^2 t)
// and the exact step means there is no discretisation error to confuse things.
//
// What is checked:
//   1. Storing paths does not change the estimates (same RNG stream, same maths).
//   2. Shapes, and row 0 of the first matrix equals x0 for every path.
//   3. Sample mean and variance of the stored state at every time match theory.
//   4. The last stored matrix agrees with what the observable accumulated.
//   5. Non-finite paths are flagged in `valid` and counted, one-for-one.
//   6. Reusing a PathStore resets it (stale flags from an earlier run are gone).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cldiff/core/rng.hpp"
#include "cldiff/market/market_inputs.hpp"
#include "cldiff/models/gbm/gbm_exact.hpp"
#include "cldiff/pricing/monte_carlo/mc_harness.hpp"
#include "cldiff/products/observable.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

using namespace cldiff;
using Catch::Approx;

namespace {

// Returns the final ln S, so the estimator tracks E[ln S_T].
class TerminalLogS final : public Observable {
 public:
  int numOutputs() const override { return 1; }
  Eigen::VectorXd result(const Eigen::VectorXd& s) const override {
    Eigen::VectorXd o(1);
    o << s(0);
    return o;
  }
};

// Returns NaN for every third path (0, 3, 6, ...), the final ln S otherwise.
class FlakyObservable final : public Observable {
 public:
  int numOutputs() const override { return 1; }
  void onStart(const Eigen::VectorXd&) override { current_ = count_++; }
  Eigen::VectorXd result(const Eigen::VectorXd& s) const override {
    Eigen::VectorXd o(1);
    o << ((current_ % 3 == 0) ? std::numeric_limits<double>::quiet_NaN() : s(0));
    return o;
  }

 private:
  int count_ = 0;
  int current_ = -1;
};

struct Setup {
  double S0 = 100.0, T = 1.0, sigma = 0.20;
  MarketInputs market{0.03, 0.01};
  int steps = 8;
  Eigen::VectorXd x0;
  Eigen::VectorXd p;
  Setup() : x0(1), p(1) {
    x0 << std::log(S0);
    p << sigma;
  }
};

}  // namespace

TEST_CASE("Storing paths does not change the estimates", "[mc][paths]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 20000;
  TerminalLogS obs;
  MonteCarloHarness harness(s.x0, scheme, obs, s.T, s.steps, n);

  Rng rng_a(7);
  const MonteCarloResult a = harness.run(rng_a);

  PathStore store;
  Rng rng_b(7);
  const MonteCarloResult b = harness.run(rng_b, &store);

  REQUIRE(a.n_nonfinite == b.n_nonfinite);
  REQUIRE(a.outputs.at(0).getMean() == b.outputs.at(0).getMean());
  REQUIRE(a.outputs.at(0).getStdError() == b.outputs.at(0).getStdError());
}

TEST_CASE("PathStore has the right shape and starts from x0", "[mc][paths]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 500;
  TerminalLogS obs;
  MonteCarloHarness harness(s.x0, scheme, obs, s.T, s.steps, n);

  PathStore store;
  Rng rng(1);
  harness.run(rng, &store);

  REQUIRE(store.states.size() == static_cast<std::size_t>(s.steps + 1));
  for (const auto& m : store.states) {
    REQUIRE(static_cast<std::size_t>(m.rows()) == n);
    REQUIRE(m.cols() == 1);
  }
  REQUIRE(store.valid.size() == n);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(store.valid[i] != 0);
    REQUIRE(store.states[0](static_cast<Eigen::Index>(i), 0) == s.x0(0));
  }
}

TEST_CASE("Stored GBM states have the exact mean and variance at every time",
          "[mc][paths][gbm]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 100000;
  TerminalLogS obs;
  MonteCarloHarness harness(s.x0, scheme, obs, s.T, s.steps, n);

  PathStore store;
  Rng rng(2026);
  harness.run(rng, &store);

  const double drift = s.market.r - s.market.q - 0.5 * s.sigma * s.sigma;
  const double dt = s.T / s.steps;
  const double N = static_cast<double>(n);

  for (int j = 1; j <= s.steps; ++j) {
    const double t = j * dt;
    const Eigen::VectorXd col = store.states[static_cast<std::size_t>(j)].col(0);
    const double mean = col.mean();
    const double var = (col.array() - mean).square().sum() / (N - 1.0);

    const double exp_mean = s.x0(0) + drift * t;
    const double exp_var = s.sigma * s.sigma * t;
    const double se_mean = std::sqrt(exp_var / N);
    const double rel_tol_var = 5.0 * std::sqrt(2.0 / (N - 1.0));

    CAPTURE(j, t, mean, exp_mean, se_mean, var, exp_var);
    REQUIRE(std::abs(mean - exp_mean) <= 4.0 * se_mean);
    REQUIRE(std::abs(var / exp_var - 1.0) <= rel_tol_var);
  }
}

TEST_CASE("Last stored matrix agrees with the observable's estimator",
          "[mc][paths]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 5000;
  TerminalLogS obs;
  MonteCarloHarness harness(s.x0, scheme, obs, s.T, s.steps, n);

  PathStore store;
  Rng rng(5);
  const MonteCarloResult res = harness.run(rng, &store);

  const double stored_mean = store.states.back().col(0).mean();
  REQUIRE(stored_mean == Approx(res.outputs.at(0).getMean()).epsilon(1e-10));
}

TEST_CASE("Non-finite paths are flagged one-for-one", "[mc][paths]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 1000;
  FlakyObservable obs;
  MonteCarloHarness harness(s.x0, scheme, obs, s.T, s.steps, n);

  PathStore store;
  Rng rng(9);
  const MonteCarloResult res = harness.run(rng, &store);

  // Paths 0, 3, ..., 999 are NaN: 334 of them.
  REQUIRE(res.n_nonfinite == 334);
  std::size_t n_valid = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const bool expect_valid = (i % 3 != 0);
    CAPTURE(i);
    REQUIRE((store.valid[i] != 0) == expect_valid);
    if (store.valid[i] != 0) ++n_valid;
  }
  REQUIRE(n_valid == n - res.n_nonfinite);
}

TEST_CASE("Reusing a PathStore resets it", "[mc][paths]") {
  const Setup s;
  const GbmScheme scheme(s.p, s.market);
  const std::size_t n = 300;

  PathStore store;
  {
    FlakyObservable flaky;
    MonteCarloHarness h(s.x0, scheme, flaky, s.T, s.steps, n);
    Rng rng(3);
    h.run(rng, &store);  // leaves some valid[i] == 0
  }
  {
    TerminalLogS clean;
    MonteCarloHarness h(s.x0, scheme, clean, s.T, 4, n);  // different step count
    Rng rng(4);
    h.run(rng, &store);
  }

  REQUIRE(store.states.size() == 5);  // 4 steps + 1, not 9
  REQUIRE(store.valid.size() == n);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(store.valid[i] != 0);
  }
}
