#pragma once
#include "cldiff/optim/annealing_schedule.hpp"

// The original shared schedule: beta_k = 1/log(3 + 2*rate*k) for the first
// 87.5% of steps, its fourth power until 97.5%, then 0. Identical for every
// chain. rate was h in Optimiser and epsilon_base in OptimiserMomentum.
class ThreePhaseLogSchedule final : public AnnealingSchedule {
public:
  explicit ThreePhaseLogSchedule(double rate);
  double beta(int step, int num_steps, int chain, int num_chains) const override;
private:
  double rate_;
};
