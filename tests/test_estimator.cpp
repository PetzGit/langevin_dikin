// Welford running estimator: known moments of 1..5, and NaN for estimates
// that are undefined with too few samples.

#include <catch2/catch_test_macros.hpp>
#include "cldiff/core/estimator.hpp"
#include <cmath>

using namespace cldiff;

TEST_CASE("Estimator moments of 1..5", "[estimator]") {
  Estimator e;
  for (int i = 1; i <= 5; ++i) e.add(static_cast<double>(i));

  CAPTURE(e.getMean(), e.getVariance(), e.getSampleVariance(), e.getStdError());
  REQUIRE(std::abs(e.getMean() - 3.0) < 1e-12);
  REQUIRE(std::abs(e.getVariance() - 2.0) < 1e-12);          // population
  REQUIRE(std::abs(e.getSampleVariance() - 2.5) < 1e-12);    // n - 1
  REQUIRE(std::abs(e.getStdError() - std::sqrt(0.5)) < 1e-12);  // sqrt(2.5 / 5)
}

TEST_CASE("Estimator is NaN when estimates are undefined", "[estimator]") {
  SECTION("empty: mean is NaN") {
    Estimator e;
    REQUIRE(std::isnan(e.getMean()));
    REQUIRE(std::isnan(e.getSampleVariance()));
    REQUIRE(std::isnan(e.getStdError()));
  }
  SECTION("one sample: mean defined, sample variance and SE are NaN") {
    Estimator e;
    e.add(7.0);
    REQUIRE(e.getMean() == 7.0);
    REQUIRE(std::isnan(e.getSampleVariance()));
    REQUIRE(std::isnan(e.getStdError()));
  }
}

TEST_CASE("Estimator is stable with a large offset", "[estimator]") {
  // Naive sum-of-squares loses all precision here; Welford should not.
  Estimator e;
  const double offset = 1e9;
  for (int i = 1; i <= 5; ++i) e.add(offset + i);
  REQUIRE(std::abs(e.getMean() - (offset + 3.0)) < 1e-6);
  REQUIRE(std::abs(e.getSampleVariance() - 2.5) < 1e-6);
}
