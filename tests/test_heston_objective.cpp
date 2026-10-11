// HestonObjective on quotes priced under known true parameters: the pricing
// loss vanishes at the truth, rises away from it, and the gradient points
// downhill.

#include <catch2/catch_test_macros.hpp>
#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"
#include "cldiff/market/market_data.hpp"
#include "cldiff/calibration/heston_objective.hpp"
#include "cldiff/calibration/feller_penalty.hpp"
#include <Eigen/Dense>

namespace {
constexpr double S0 = 100.0, r = 0.02, L = 10.0;
constexpr int term_num = 128;

// Chosen to satisfy Feller: 2*kappa*theta = 0.16 >= xi^2 = 0.09
Eigen::VectorXd trueParams() {
  Eigen::VectorXd p(5);
  p << 2.0, 0.04, 0.30, -0.70, 0.04;  // kappa, theta, xi, rho, v0
  return p;
}

MarketData syntheticMarket() {
  const CharacteristicHeston phi(trueParams());
  MarketData market(S0, r);
  for (double T : {0.5, 1.0}) {
    Cumulants cum = cumulants(phi, T);
    const Range range = make_range(cum, L);
    for (double K : {90.0, 100.0, 110.0}) {
      market.addQuote(T, K, cos_pricer(phi, range, K, T, r, term_num, S0));
    }
  }
  return market;
}
}  // namespace

TEST_CASE("pricing loss is zero at the true parameters", "[heston_objective]") {
  const MarketData market = syntheticMarket();
  const Eigen::VectorXd truth = trueParams();

  HestonObjective unpenalised(market, term_num, L, 0.0);
  REQUIRE(unpenalised.value(truth) == 0.0);

  // With the penalty on, all that remains at the truth is the smoothed
  // Feller term, which is small but non-zero even inside the Feller region.
  HestonObjective penalised(market, term_num, L, 1.0);
  REQUIRE(penalised.value(truth) == FellerPenalty(1.0)(truth));
}

TEST_CASE("loss rises away from the truth and the gradient points downhill", "[heston_objective]") {
  const MarketData market = syntheticMarket();
  HestonObjective objective(market, term_num, L, 1.0);
  const Eigen::VectorXd truth = trueParams();
  Eigen::VectorXd perturbed = truth;
  perturbed(0) *= 1.5;  // bump kappa by 50%

  const double loss_perturbed = objective.value(perturbed);
  REQUIRE(loss_perturbed > objective.value(truth));

  const Eigen::VectorXd grad = objective.gradient(perturbed);
  const Eigen::VectorXd stepped = perturbed - 1e-5 * grad;
  REQUIRE(objective.value(stepped) < loss_perturbed);
}

TEST_CASE("gradient is near zero at the true parameters", "[heston_objective]") {
  const MarketData market = syntheticMarket();
  HestonObjective objective(market, term_num, L, 0.0);
  const Eigen::VectorXd truth = trueParams();
  Eigen::VectorXd perturbed = truth;
  perturbed(0) *= 1.5;

  // Not exactly zero: the finite-difference step (2e-3) is ~5% of theta and
  // v0, so third-order terms leave a ~3e-4 bias. Still ~90x below the
  // gradient at the perturbed point.
  REQUIRE(objective.gradient(truth).norm() < objective.gradient(perturbed).norm() / 20.0);
}
