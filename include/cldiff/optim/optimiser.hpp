#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/problem.hpp"

namespace cldiff {

struct OptimResult {
  Eigen::VectorXd x;
  double value;
};

// Interface for every minimiser in the library. Constructors take only the
// algorithm's hyperparameters; the problem, starting points and randomness
// arrive with each call.
class Optimiser {
public:
  virtual ~Optimiser() = default;
  // x0 is d x n: particle methods use every column as a chain, single-start
  // methods use column 0.
  virtual OptimResult minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) = 0;
  virtual std::string name() const = 0;
};

}  // namespace cldiff
