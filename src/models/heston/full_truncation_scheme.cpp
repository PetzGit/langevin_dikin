#include "cldiff/models/heston/full_truncation_scheme.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cldiff {

HestonScheme::HestonScheme(const Eigen::VectorXd& p, double r, double q) {
  params_.r = r;
  params_.q = q;
  setParameters(p);
}

void HestonScheme::step(Eigen::VectorXd& state, double dt, Rng& rng) const {
  const double z1 = rng.normal();
  const double z2 = rng.normal();
  const double eps1 = z1;
  const double eps2 = params_.rho * z1
                    + std::sqrt(1.0 - params_.rho * params_.rho) * z2;

  const double vp = std::max(0.0, state(1));
  const double sq = std::sqrt(vp * dt);

  state(0) += (params_.r - params_.q - 0.5 * vp) * dt + sq * eps1;
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
  params_.kappa = p(0);
  params_.theta = p(1);
  params_.xi    = p(2);
  params_.rho   = p(3);
}

Eigen::VectorXd HestonScheme::parameters() const {
  Eigen::VectorXd p(4);
  p << params_.kappa, params_.theta, params_.xi, params_.rho;
  return p;
}

}  // namespace cldiff
