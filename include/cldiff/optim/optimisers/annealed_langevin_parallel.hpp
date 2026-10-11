#pragma once
#include "cldiff/optim/optimisers/annealed_langevin.hpp"

// Optimiser variant addressing a specific failure mode of the shared
// schedule: every chain reaches beta=0 (fully deterministic) together at
// the end of the run, so once the population has collectively committed to
// a region via resampling, there is no mechanism left to escape it no
// matter how much budget remains -- observed directly in this project as
// larger run_num sometimes making results *worse*, not better.
//
// Fix: give each chain a persistent temperature floor, spread across the
// population (chain 0 floors at 0 -- free to fully cool, becomes a
// "refiner"; the last chain never drops below floor_max, staying a
// permanent "explorer"). Actual per-chain temperature is
// max(shared_schedule(step), floor_i). The existing resample() step already
// propagates a hot chain's lucky find into the cooling population by
// fitness, so floors keep that channel fed with genuine exploration
// throughout the run instead of it drying up once the shared schedule
// hits zero.
class ParallelAnnealingOptimiser : public Optimiser {
public:
  ParallelAnnealingOptimiser(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                              Eigen::MatrixXd x_0, int r_rate, int chain_num, int run_num,
                              double h, double eps, double floor_max);

protected:
  double chainBeta(int chain_index) const override;

private:
  Eigen::VectorXd floors_;
};
