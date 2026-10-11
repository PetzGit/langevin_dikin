// Black-Scholes implied-vol round trip: price a call at a known sigma, then
// recover sigma from that price via impliedVol.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "cldiff/market/implied_vol.hpp"
#include <cmath>

using namespace cldiff;

TEST_CASE("impliedVol inverts price", "[implied_vol]") {
  const double sigma_true = GENERATE(0.10, 0.20, 0.40, 0.80);
  BlackScholesImpliedVol bs(100.0, 100.0, 1.0, 0.02);

  const double C = bs.price(sigma_true);
  CAPTURE(sigma_true, C);
  REQUIRE(std::abs(bs.impliedVol(C) - sigma_true) < 1e-8);
}
