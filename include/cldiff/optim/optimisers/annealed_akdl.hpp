#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/optim/optimiser.hpp"
#include "cldiff/optim/annealing_schedule.hpp"

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
// step size found while tuning AKDLSampler on the Heston posterior.
// beta is floored away from exactly 0 (min_beta) since AKDL's dynamics --
// unlike the overdamped SDE's separate noise/drift terms -- divide the
// force by beta directly and are only well-posed for beta > 0.
//
// ThreePhaseLogSchedule(epsilon_base) reproduces the original schedule.
class AnnealedAKDLOptimiser final : public Optimiser {
public:
  AnnealedAKDLOptimiser(const AnnealingSchedule& schedule, int r_rate, int run_num,
                        double epsilon_base, double gamma, int L, double min_beta = 1e-3,
                        double tau_mid = 1e-10, int n_mid = 20);
  OptimResult minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) override;
  std::string name() const override { return "annealed_akdl"; }
private:
  struct Run;
  // Runs one AKDL macro-step on chain i in place, at the given beta.
  void updateChain(Run& run, int chain_index, double beta) const;
  void step(Run& run) const;
  void resample(Run& run) const;
  const AnnealingSchedule& schedule_;
  int r_rate_;
  int run_num_;
  double epsilon_base_;
  double gamma_;
  int L_;
  double min_beta_;
  double tau_mid_;
  int n_mid_;
};
