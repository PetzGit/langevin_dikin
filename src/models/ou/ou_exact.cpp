#include "cldiff/models/ou/ou_exact.hpp"
#include <stdexcept>
#include <cmath>

namespace cldiff {

void OuScheme::setParameters(const Eigen::VectorXd& p) {
  if ((p.size() != numParameters()) || (!(p(0) > 0)) || (!(p(2) > 0)) || (!std::isfinite(p(1)))) {
    throw std::invalid_argument("check the parameters");
  }
  params_ = OuParams{p(0),p(1),p(2)}; //p(0) : kappa, p(1) : mu, p(2) : sigma
}

OuScheme::OuScheme(const Eigen::VectorXd& p) {
  setParameters(p);
}

int OuScheme::dimension() const {
  return 1;
}

int OuScheme::numParameters() const {
  return 3;
}

Eigen::VectorXd OuScheme::parameters() const {
  Eigen::VectorXd p(3);
  p(0) = params_.kappa;
  p(1) = params_.mu;
  p(2) = params_.sigma;
  return p;
}

void OuScheme::step(Eigen::VectorXd& state, double dt, Rng& rng) const {
  double decay = std::exp(-params_.kappa * dt);
  double sd = params_.sigma * std::sqrt(-std::expm1(-2.0*params_.kappa*dt) / (2.0*params_.kappa));
  state(0) = params_.mu + (state(0) - params_.mu) * decay + sd * rng.normal();
}
}
