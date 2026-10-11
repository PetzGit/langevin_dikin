#include "cldiff/optim/log_space_objective.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cldiff {

LogSpaceObjective::LogSpaceObjective(const Objective& inner, std::vector<bool> log_dims, double fd_step)
: inner_(inner), log_dims_(std::move(log_dims)), fd_step_(fd_step) {
  if (static_cast<int>(log_dims_.size()) != inner_.dim() || fd_step_ <= 0) {
    throw std::invalid_argument("Invalid arguments: log_dims must have one flag per dimension, fd_step > 0");
  }
}

double LogSpaceObjective::value(const Eigen::VectorXd& y) const {
  return inner_.value(toPhysical(y));
}

Eigen::VectorXd LogSpaceObjective::gradient(const Eigen::VectorXd& y) const {
  const int d = dim();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(d);
  const double h = fd_step_;
  for (int i = 0; i < d; i++) {
    Eigen::VectorXd yp = y, ym = y;
    yp(i) += h;
    ym(i) -= h;
    grad(i) = (value(yp) - value(ym)) / (2.0 * h);
  }
  const Eigen::VectorXd p = toPhysical(y);
  for (int i = 0; i < d; i++) {
    if (log_dims_[i]) {
      grad(i) /= std::max(p(i), 1e-8);
    }
  }
  return grad;
}

int LogSpaceObjective::dim() const {
  return inner_.dim();
}

Eigen::VectorXd LogSpaceObjective::toPhysical(const Eigen::VectorXd& y) const {
  return toPhysical(y, log_dims_);
}

Eigen::VectorXd LogSpaceObjective::toLogSpace(const Eigen::VectorXd& p) const {
  return toLogSpace(p, log_dims_);
}

Eigen::VectorXd LogSpaceObjective::toPhysical(const Eigen::VectorXd& y, const std::vector<bool>& log_dims) {
  Eigen::VectorXd p(y.size());
  for (int i = 0; i < y.size(); i++) {
    p(i) = log_dims[i] ? std::exp(y(i)) : y(i);
  }
  return p;
}

Eigen::VectorXd LogSpaceObjective::toLogSpace(const Eigen::VectorXd& p, const std::vector<bool>& log_dims) {
  Eigen::VectorXd y(p.size());
  for (int i = 0; i < p.size(); i++) {
    y(i) = log_dims[i] ? std::log(p(i)) : p(i);
  }
  return y;
}

}  // namespace cldiff
