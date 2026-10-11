// COS pricer against the Fang & Oosterlee (2008) Heston benchmark.

#include <catch2/catch_test_macros.hpp>
#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace {
// Fang & Oosterlee (2008) benchmark Heston parameters.
Eigen::VectorXd benchmarkParams() {
  Eigen::VectorXd params(5);
  params << 1.5768,   // kappa
            0.0398,   // theta
            0.5751,   // xi
           -0.5711,   // rho
            0.0175;   // v0
  return params;
}
}  // namespace

TEST_CASE("characteristic function is 1 at u = 0", "[cos]") {
  const CharacteristicHeston phi(benchmarkParams());
  REQUIRE(std::abs(phi.eval(0.0, 1.0) - 1.0) < 1e-14);
}

TEST_CASE("COS price converges to the Fang-Oosterlee reference", "[cos]") {
  const CharacteristicHeston phi(benchmarkParams());
  const double S0 = 100.0, K = 100.0, T = 1.0, r = 0.0, L = 10.0;
  const double reference = 5.785155450;  // Fang & Oosterlee (2008), T = 1

  Cumulants cum = cumulants(phi, T);
  const Range range = make_range(cum, L);
  auto err = [&](int N) { return std::abs(cos_pricer(phi, range, K, T, r, N, S0) - reference); };

  CHECK(err(32) < err(16));
  CHECK(err(64) < err(32));
  CHECK(err(128) < err(64));
  // Beyond N=128 the error plateaus at ~2e-7 (truncation-range limited).
  REQUIRE(err(128) < 1e-6);
}
