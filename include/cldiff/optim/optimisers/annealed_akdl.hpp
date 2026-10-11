#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/tempered_objective.hpp"
#include "cldiff/optim/optimisers/annealed_langevin.hpp"  // reuses ProposalResult

// Annealed optimiser built on genuine AKDL (Metropolis-adjusted kinetic
// Dikin-Langevin, Algorithm 4) chains, rather than the unadjusted UKDL
// dynamics: each chain carries momentum coherently across L deterministic
// sub-steps and can reject a bad proposal (unlike UKDL, which takes any step
// that merely passes the feasibility/Newton checks, with no safety net --
// this was the direct cause of outright divergence at larger step sizes in
// an earlier unadjusted design).
//
// Annealing tempers the *objective itself*, U_beta(x) = U(x)/beta (the
// textbook simulated-annealing mechanism -- as beta -> 0, exp(-U(x)/beta)
// collapses onto the global minimum), rather than scaling injected noise.
// Per-step epsilon is scaled as epsilon_base * sqrt(beta), matching MALA
// optimal-scaling theory (step size ~ (target sharpness)^{-1/2}) and the
// empirically-observed relationship between likelihood scale and required
// step size found while tuning SamplerMomentum on the Heston posterior.
// beta is floored away from exactly 0 (min_beta) since AKDL's dynamics --
// unlike the overdamped SDE's separate noise/drift terms -- divide the
// force by beta directly and are only well-posed for beta > 0.
class OptimiserMomentum {
public:
  OptimiserMomentum(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                     Eigen::MatrixXd x_0, int r_rate, int chain_num, int run_num,
                     double epsilon_base, double gamma, int L, double min_beta = 1e-3,
                     double tau_mid = 1e-10, int n_mid = 20);
  virtual ~OptimiserMomentum() = default;
  ProposalResult run();

protected:
  void step();
  // Temperature assigned to chain i at the current step_num_ (before the
  // min_beta floor is applied). Default: the single shared schedule,
  // identical for every chain. Override for heterogeneous schedules.
  virtual double chainBeta(int chain_index) const;
  // Runs one AKDL macro-step on chain i in place, at the given beta.
  void updateChain(int chain_index, double beta);
  void resample();
  Eigen::VectorXd genTemp(int run_num) const;

  const Objective& objective_;
  const Polyhedron& polyhedron_;
  Rng& rng_;
  TemperedObjective tempered_;
  Eigen::MatrixXd x_;  // d x chain_num
  Eigen::MatrixXd v_;  // d x chain_num
  Eigen::VectorXi sigma_;  // chain_num, direction in {-1,+1}
  Eigen::VectorXd temp_;
  int r_rate_;
  int chain_num_;
  int run_num_;
  int step_num_;
  double epsilon_base_;
  double gamma_;
  int L_;
  double min_beta_;
  double tau_mid_;
  int n_mid_;
};
