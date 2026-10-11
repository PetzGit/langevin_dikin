#pragma once
#include <string>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "cldiff/optim/sampler.hpp"

namespace cldiff {

struct Proposal {
  Eigen::VectorXd Y;
  Eigen::VectorXd x_i;
};

// Metropolis-adjusted overdamped Dikin-Langevin sampler on a polytope.
class DikinLangevinSampler final : public Sampler {
public:
  DikinLangevinSampler(double eps, double h_max);
  Eigen::MatrixXd sample(const Problem& problem, const Eigen::VectorXd& x0, Rng& rng, int n) override;
  std::string name() const override { return "dikin_langevin"; }
private:
  struct Run;
  void step(Run& run) const;
  Eigen::VectorXd proposeState(const Run& run, const Eigen::VectorXd& x, const Eigen::LLT<Eigen::MatrixXd>& llt, const double beta) const;
  Proposal propose(Run& run, const Eigen::VectorXd& x, const Eigen::VectorXd& m_x, double h, const Eigen::MatrixXd& L, const double beta) const;
  double eps_;
  double h_max_;
};

}  // namespace cldiff
