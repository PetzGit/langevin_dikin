#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"

// Numerical adjusted kinetic Dikin-Langevin (AKDL), Algorithm 4 of
// Chok, Petzinna & Vasil (2026). Metropolis-adjusted momentum sampler on a
// polytope: unlike the overdamped Sampler, this retains an auxiliary
// velocity between iterations (persistent momentum) via a direction
// variable sigma in {-1,+1}, and uses an implicit-midpoint integrator whose
// volume change (Jacobian) enters the acceptance probability directly.
class SamplerMomentum {
public:
  // epsilon: refresh/step-size magnitude. gamma: OU friction (velocity
  // persistence rho = exp(-gamma*epsilon)). L: deterministic steps per
  // iteration. tau_mid/n_mid: modified-Newton tolerance/iteration cap for
  // the midpoint solve (paper defaults: 1e-10, 20).
  SamplerMomentum(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                   Eigen::VectorXd x, double epsilon, double gamma, int L,
                   double tau_mid = 1e-10, int n_mid = 20);
  void step();
  Eigen::MatrixXd run(int n);
  double acceptanceRate() const;
  double epsilon() const { return epsilon_; }

  // Robbins-Monro step-size adaptation (matches the reference
  // implementation's update_log_step): for n_warmup steps, after each step
  // log(epsilon) += eta0*(i+t0)^(-kappa) * (accept_prob - target), using the
  // continuous acceptance probability (not the binary accept/reject outcome)
  // as the feedback signal, then freezes epsilon at its final value.
  // Resets the accept-rate counters afterward so acceptanceRate() reflects
  // only post-warmup steps.
  void warmup(int n_warmup, double target_accept = 0.9, double eta0 = 5.0,
              double t0 = 10.0, double kappa = 0.75,
              double min_epsilon = 1e-8, double max_epsilon = 10.0);

private:
  // Runs one step at a caller-specified epsilon (used by warmup); returns
  // the continuous acceptance probability actually used for the decision.
  double stepWithEpsilon(double eps);

  const Objective& objective_;
  const Polyhedron& polyhedron_;
  Rng& rng_;
  Eigen::VectorXd x_;
  Eigen::VectorXd v_;
  int sigma_;
  double epsilon_;
  double gamma_;
  int L_;
  double tau_mid_;
  int n_mid_;
  long n_steps_ = 0;
  long n_accept_ = 0;
};
