#pragma once
#include "cldiff/calibration/parameter_penalty.hpp"

// lambda * softplus-smoothed max(0, xi^2 - 2*kappa*theta) for Heston params.
// smooth_scale: width over which the hard max(0,x) is smoothed into a
// softplus. The hard max(0,x) is continuous but has a non-differentiable
// kink at x=0; whenever a finite-difference stencil straddles that kink
// (common in practice -- real calibrated Heston fits routinely sit right at
// the Feller boundary), the estimated gradient can be badly wrong, even
// sign-flipped, even though the true (piecewise) function is well
// approximated by a smooth curve away from the immediate kink. Default
// 0.01 is comfortably wider than the finite-difference step sizes used
// elsewhere in calibration gradients (~1e-3 in log-space).
class FellerPenalty final : public ParameterPenalty {
public:
  explicit FellerPenalty(double lambda, double smooth_scale = 0.01);
  double operator()(const Eigen::VectorXd& params) const override;
private:
  double lambda_;
  double smooth_scale_;
};
