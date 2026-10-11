#include "cldiff/optim/optimisers/projected_gradient_descent.hpp"
#include <stdexcept>

namespace cldiff {

ProjectedGradientDescent::ProjectedGradientDescent(int max_iters, int max_backtracks, double initial_lr)
    : max_iters_(max_iters), max_backtracks_(max_backtracks), initial_lr_(initial_lr)
{
  if (max_iters_ < 0 || max_backtracks_ < 1 || initial_lr_ <= 0) {
    throw std::invalid_argument("Invalid arguments: check max_iters, max_backtracks, initial_lr");
  }
}

OptimResult ProjectedGradientDescent::minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& /*rng*/) {
  const Objective& obj = problem.objective;
  const Polyhedron& polyhedron = problem.domain;
  if (x0.cols() < 1 || x0.rows() != obj.dim()) {
    throw std::invalid_argument("Invalid arguments: check dimensions of x0");
  }

  Eigen::VectorXd y = x0.col(0);
  double lr = initial_lr_;
  double current_loss = obj.value(y);
  stuck_ = false;

  for (int iter = 1; iter <= max_iters_; iter++) {
    Eigen::VectorXd grad = obj.gradient(y);
    double trial_lr = lr;
    Eigen::VectorXd y_new;
    double new_loss = current_loss;
    bool improved = false;

    for (int bt = 0; bt < max_backtracks_; bt++) {
      y_new = y - trial_lr * grad;
      if (polyhedron.contains(y_new)) {
        new_loss = obj.value(y_new);
        if (new_loss < current_loss) {
          improved = true;
          break;
        }
      }
      trial_lr *= 0.5;
    }

    if (!improved) {
      stuck_ = true;
      break;
    }

    y = y_new;
    current_loss = new_loss;
    lr = trial_lr * 1.5;
  }

  return OptimResult{y, current_loss};
}

}  // namespace cldiff
