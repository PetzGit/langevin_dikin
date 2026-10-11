#pragma once
#include "cldiff/optim/optimisers/annealed_akdl.hpp"

// ParallelAnnealingOptimiser's momentum counterpart: per-chain temperature
// floors so some chains stay persistently warmer (exploring) throughout the
// run instead of every chain reaching beta=0 together. See
// optimiser_parallel_annealing.hpp for the full rationale.
class ParallelAnnealingOptimiserMomentum : public OptimiserMomentum {
public:
  ParallelAnnealingOptimiserMomentum(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                                      Eigen::MatrixXd x_0, int r_rate, int chain_num, int run_num,
                                      double epsilon_base, double gamma, int L, double floor_max,
                                      double min_beta = 1e-3, double tau_mid = 1e-10, int n_mid = 20);

protected:
  double chainBeta(int chain_index) const override;

private:
  Eigen::VectorXd floors_;
};
