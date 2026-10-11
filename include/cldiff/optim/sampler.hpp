#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/problem.hpp"

// Interface for every sampler of exp(-U(x)) on a polytope. Same split as
// Optimiser: hyperparameters at construction, problem per call.
class Sampler {
public:
  virtual ~Sampler() = default;
  // Returns d x n, one draw per column.
  virtual Eigen::MatrixXd sample(const Problem& problem, const Eigen::VectorXd& x0, Rng& rng, int n) = 0;
  virtual std::string name() const = 0;
};
