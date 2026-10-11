#pragma once
#include <Eigen/Dense>

namespace cldiff {

bool satisfiesFeller(const Eigen::VectorXd& params);
// xi^2 - 2*kappa*theta: positive exactly when the Feller condition is violated.
double fellerViolation(const Eigen::VectorXd& params);

}  // namespace cldiff
