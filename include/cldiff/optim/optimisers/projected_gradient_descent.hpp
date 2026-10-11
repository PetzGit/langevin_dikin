#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/optim/optimiser.hpp"

namespace cldiff {

// Backtracking gradient descent that rejects any trial step leaving the
// polyhedron, the same way AnnealedLangevinOptimiser rejects an out-of-box
// proposal. Without this the descent walks straight past hard parameter
// bounds -- rho outside [-1,1] being the one that actually bit, since the fits
// it produced were not valid Heston models at all. Each iteration halves the
// trial step up to max_backtracks times until it lands inside the domain and
// improves the loss, then grows the next starting step by 1.5x.
//
// Deterministic: uses x0.col(0) and never draws from rng. Used both as a
// baseline and as a local "polish" run from another optimiser's result.
class ProjectedGradientDescent final : public Optimiser {
public:
  ProjectedGradientDescent(int max_iters, int max_backtracks, double initial_lr = 0.1);
  OptimResult minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) override;
  std::string name() const override { return "projected_gradient_descent"; }
  // Whether the most recent minimise() stopped because the line search found
  // no admissible improving step, as opposed to hitting max_iters.
  bool stuck() const { return stuck_; }
private:
  int max_iters_;
  int max_backtracks_;
  double initial_lr_;
  bool stuck_ = false;
};

}  // namespace cldiff
