#include "cldiff/optim/optimisers/annealed_langevin_parallel.hpp"
#include <algorithm>

ParallelAnnealingOptimiser::ParallelAnnealingOptimiser(
    const Objective& objective, const Polyhedron& polyhedron, Rng& rng, Eigen::MatrixXd x_0,
    int r_rate, int chain_num, int run_num, double h, double eps, double floor_max)
    : Optimiser(objective, polyhedron, rng, std::move(x_0), r_rate, chain_num, run_num, h, eps),
      floors_(chain_num)
{
  for (int i = 0; i < chain_num; i++) {
    floors_(i) = chain_num > 1 ? floor_max * static_cast<double>(i) / (chain_num - 1) : 0.0;
  }
}

double ParallelAnnealingOptimiser::chainBeta(int chain_index) const {
  return std::max(temp_(step_num_), floors_(chain_index));
}
