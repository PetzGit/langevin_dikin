#include "cldiff/models/gbm/gbm_exact.hpp"
#include <stdexcept>
#include <cmath>

namespace cldiff {

void GbmScheme::setParameters(const Eigen::VectorXd& p) {
  if (p.size() != numParameters() || !(p(0) > 0)) {
    throw std::invalid_argument("check the parameters");
  }
  params_ = GbmParams{p(0)};
}

GbmScheme::GbmScheme(const Eigen::VectorXd& p, MarketInputs market) 
: market_(market) {
  setParameters(p);
}

int GbmScheme::dimension() const {
  return 1;
}

int GbmScheme::numParameters() const {
  return 1;
}

Eigen::VectorXd GbmScheme::parameters() const {
  Eigen::VectorXd p(1);
  p(0) = params_.sigma;
  return p;
}

void GbmScheme::step(Eigen::VectorXd& state, double dt, Rng& rng) const {
  double z = rng.normal();
  state(0) += (market_.r - market_.q - 0.5 * params_.sigma * params_.sigma) * dt + params_.sigma * std::sqrt(dt) * z;
}
}

