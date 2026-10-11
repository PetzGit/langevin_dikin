#include "cldiff/calibration/corrected_log_space_heston_objective.hpp"

namespace cldiff {

namespace {
// (kappa, theta, xi, rho, v0): everything but rho is strictly positive.
const std::vector<bool>& hestonLogDims() {
  static const std::vector<bool> log_dims = {true, true, true, false, true};
  return log_dims;
}
}  // namespace

CorrectedLogSpaceHestonObjective::CorrectedLogSpaceHestonObjective(const Objective& inner)
: LogSpaceObjective(inner, hestonLogDims()) {}

Eigen::VectorXd CorrectedLogSpaceHestonObjective::toPhysical(const Eigen::VectorXd& y) {
  return LogSpaceObjective::toPhysical(y, hestonLogDims());
}

Eigen::VectorXd CorrectedLogSpaceHestonObjective::toLogSpace(const Eigen::VectorXd& p) {
  return LogSpaceObjective::toLogSpace(p, hestonLogDims());
}

}  // namespace cldiff
