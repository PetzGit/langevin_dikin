#include "cldiff/calibration/feller_penalty.hpp"
#include "cldiff/models/heston/feller.hpp"
#include <algorithm>
#include <cmath>

namespace {
// Numerically-stable softplus(x/scale)*scale, the standard smooth
// replacement for max(0,x): converges to max(0,x) as scale->0, matches it
// closely for |x| >> scale, and is differentiable everywhere (no kink at
// x=0). Stable form avoids overflow in exp() for large |x|/scale.
double smoothRelu(double x, double scale) {
  const double z = x / scale;
  return scale * (std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z))));
}
}  // namespace

FellerPenalty::FellerPenalty(double lambda, double smooth_scale)
: lambda_(lambda), smooth_scale_(smooth_scale) {}

double FellerPenalty::operator()(const Eigen::VectorXd& params) const {
  return lambda_*smoothRelu(fellerViolation(params), smooth_scale_);
}
