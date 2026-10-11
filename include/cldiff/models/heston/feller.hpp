#pragma once
#include <Eigen/Dense>
bool satisfiesFeller(const Eigen::VectorXd& params);
// xi^2 - 2*kappa*theta: positive exactly when the Feller condition is violated.
double fellerViolation(const Eigen::VectorXd& params);
