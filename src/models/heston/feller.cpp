#include "cldiff/models/heston/feller.hpp"

namespace cldiff {

bool satisfiesFeller(const Eigen::VectorXd& params) {
  return 2*params(0)*params(1) >= params(2)*params(2);
}

double fellerViolation(const Eigen::VectorXd& params) {
  return params(2)*params(2) - 2*params(1)*params(0);
}

}  // namespace cldiff
