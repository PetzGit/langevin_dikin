#include "cldiff/optim/tempered_objective.hpp"

namespace cldiff {

TemperedObjective::TemperedObjective(const Objective& inner, double beta)
    : inner_(inner), beta_(beta) {}

void TemperedObjective::setBeta(double beta) { beta_ = beta; }
double TemperedObjective::beta() const { return beta_; }

double TemperedObjective::value(const Eigen::VectorXd& x) const {
  return inner_.value(x) / beta_;
}

Eigen::VectorXd TemperedObjective::gradient(const Eigen::VectorXd& x) const {
  return inner_.gradient(x) / beta_;
}

int TemperedObjective::dim() const { return inner_.dim(); }

}  // namespace cldiff
