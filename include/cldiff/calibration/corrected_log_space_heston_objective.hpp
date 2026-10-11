#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/log_space_objective.hpp"

namespace cldiff {

// Log-space parameterization: kappa, theta, xi and v0 are exp-transformed so
// they stay positive, rho passes through unchanged. The chain-rule scaling
// (dL/dy = dL/dp * p) is divided back out for the exp-transformed dims.
// Without that division the log-space gradient shrinks in proportion to the
// parameter's own value as it approaches zero, even when the true physical
// sensitivity is still large -- confirmed directly: at kappa=0.01, physical
// dL/dkappa was -0.134 but log-space dL/d(log kappa) was only -0.00134, a 100x
// suppression. That produced a real, repeatable failure mode: on a reduced
// (subset) training grid the uncorrected version of this class collapsed to
// the same degenerate corner (kappa pinned at its box floor, rho pinned at its
// box wall) regardless of starting point, and generalized badly out of sample.
// Dividing by p(i) restores a step whose *relative* size tracks the true
// physical sensitivity instead of being conflated with how small p already is,
// and fixes the collapse (kappa/xi/rho all land near truth instead,
// out-of-sample error drops ~5-9x in testing). rho (index 3) isn't
// exp-transformed, so it needs no correction.
//
// This is LogSpaceObjective with Heston's mask (kappa, theta, xi, v0 logged;
// rho not). The statics apply the same mask without needing an instance.
class CorrectedLogSpaceHestonObjective : public LogSpaceObjective {
  public:
    explicit CorrectedLogSpaceHestonObjective(const Objective& inner);
    static Eigen::VectorXd toPhysical(const Eigen::VectorXd& y);
    static Eigen::VectorXd toLogSpace(const Eigen::VectorXd& p);
};

}  // namespace cldiff
