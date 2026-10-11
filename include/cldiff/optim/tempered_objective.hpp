#pragma once
#include "cldiff/optim/objective.hpp"

// Wraps an objective as U(x)/beta, with a mutable beta -- the textbook
// simulated-annealing mechanism: as beta -> 0, exp(-U(x)/beta) collapses
// onto the global minimum of U. Distinct from scaling injected noise (the
// approach used elsewhere in this codebase): this tempers the target
// density itself, so a Metropolis-adjusted sampler run against it at a
// sequence of decreasing beta values is a proper simulated-annealing chain,
// not a heuristic.
class TemperedObjective : public Objective {
public:
  explicit TemperedObjective(const Objective& inner, double beta = 1.0);
  void setBeta(double beta);
  double beta() const;
  double value(const Eigen::VectorXd& x) const override;
  Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override;
  int dim() const override;

private:
  const Objective& inner_;
  double beta_;
};
