#include "cldiff/optim/optimisers/annealed_akdl_parallel.hpp"
#include <algorithm>

ParallelAnnealingOptimiserMomentum::ParallelAnnealingOptimiserMomentum(
    const Objective& objective, const Polyhedron& polyhedron, Rng& rng, Eigen::MatrixXd x_0,
    int r_rate, int chain_num, int run_num, double epsilon_base, double gamma, int L,
    double floor_max, double min_beta, double tau_mid, int n_mid)
    : OptimiserMomentum(objective, polyhedron, rng, std::move(x_0), r_rate, chain_num, run_num,
                         epsilon_base, gamma, L, min_beta, tau_mid, n_mid),
      floors_(chain_num)
{
  for (int i = 0; i < chain_num; i++) {
    floors_(i) = chain_num > 1 ? floor_max * static_cast<double>(i) / (chain_num - 1) : 0.0;
  }
}

double ParallelAnnealingOptimiserMomentum::chainBeta(int chain_index) const {
  return std::max(temp_(step_num_), floors_(chain_index));
}
