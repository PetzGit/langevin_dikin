#include "cldiff/models/heston/full_truncation_scheme.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cldiff {

HestonScheme::HestonScheme(const Eigen::VectorXd& p, MarketInputs market)
    : market_(market) {
  setParameters(p);
}

void HestonScheme::step(Eigen::VectorXd& state, double dt, Rng& rng) const {
  const double z1 = rng.normal();
  const double z2 = rng.normal();
  const double eps1 = z1;
  const double eps2 = params_.rho * z1
                    + rho_bar_* z2;

  const double vp = std::max(0.0, state(1));
  const double sq = std::sqrt(vp * dt);

  state(0) += (market_.r - market_.q - 0.5 * vp) * dt + sq * eps1;
  state(1) += params_.kappa * (params_.theta - vp) * dt
            + params_.xi * sq * eps2;
}

int HestonScheme::dimension() const { return 2; }
int HestonScheme::numParameters() const { return 4; }

void HestonScheme::setParameters(const Eigen::VectorXd& p) {
  if (p.size() != numParameters()) {
    throw std::invalid_argument(
        "HestonScheme: expected 4 parameters (kappa, theta, xi, rho)");
  }
  if (!(p(0) > 0) || !(p(1) > 0) || !(p(2) > 0) || !(std::abs(p(3)) < 1)) {
    throw std::invalid_argument(
        "HestonScheme: need kappa, theta, xi > 0 and |rho| < 1");
  }
  params_.kappa = p(0);
  params_.theta = p(1);
  params_.xi    = p(2);
  params_.rho   = p(3);
  rho_bar_      = std::sqrt(1.0 - params_.rho * params_.rho);
}

Eigen::VectorXd HestonScheme::parameters() const {
  Eigen::VectorXd p(4);
  p << params_.kappa, params_.theta, params_.xi, params_.rho;
  return p;
}

}  // namespace cldiff
