#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/optim/sampler.hpp"

namespace cldiff {

// Robbins-Monro step-size adaptation run at the start of every sample() call
// (matches the reference implementation's update_log_step): for n_warmup
// steps, after each step log(epsilon) += eta0*(i+t0)^(-kappa) * (accept_prob - target),
// using the continuous acceptance probability (not the binary accept/reject
// outcome) as the feedback signal, then freezes epsilon at its final value.
// Warmup draws are not returned. n_warmup = 0 disables it.
struct AKDLWarmup {
  int n_warmup = 0;
  double target_accept = 0.9;
  double eta0 = 5.0;
  double t0 = 10.0;
  double kappa = 0.75;
  double min_epsilon = 1e-8;
  double max_epsilon = 10.0;
};

// Numerical adjusted kinetic Dikin-Langevin (AKDL), Algorithm 4 of
// Chok, Petzinna & Vasil (2026). Metropolis-adjusted momentum sampler on a
// polytope: unlike the overdamped DikinLangevinSampler, this retains an auxiliary
// velocity between iterations (persistent momentum) via a direction
// variable sigma in {-1,+1}, and uses an implicit-midpoint integrator whose
// volume change (Jacobian) enters the acceptance probability directly.
class AKDLSampler final : public Sampler {
public:
  // epsilon: refresh/step-size magnitude. gamma: OU friction (velocity
  // persistence rho = exp(-gamma*epsilon)). L: deterministic steps per
  // iteration. tau_mid/n_mid: modified-Newton tolerance/iteration cap for
  // the midpoint solve (paper defaults: 1e-10, 20).
  AKDLSampler(double epsilon, double gamma, int L, AKDLWarmup warmup = AKDLWarmup{},
              double tau_mid = 1e-10, int n_mid = 20);
  Eigen::MatrixXd sample(const Problem& problem, const Eigen::VectorXd& x0, Rng& rng, int n) override;
  std::string name() const override { return "akdl"; }
  // Both describe the most recent sample() call: the post-warmup acceptance
  // rate and the (possibly adapted) step size actually used.
  double acceptanceRate() const;
  double epsilon() const { return epsilon_; }
private:
  struct Run;
  // Runs one step at a caller-specified epsilon (used by warmup); returns
  // the continuous acceptance probability actually used for the decision.
  double stepWithEpsilon(Run& run, double eps);
  double epsilon_init_;
  double gamma_;
  int L_;
  AKDLWarmup warmup_;
  double tau_mid_;
  int n_mid_;
  double epsilon_;
  long n_steps_ = 0;
  long n_accept_ = 0;
};

}  // namespace cldiff
