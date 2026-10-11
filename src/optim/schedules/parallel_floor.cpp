#include "cldiff/optim/schedules/parallel_floor.hpp"
#include <algorithm>

ParallelFloorSchedule::ParallelFloorSchedule(const AnnealingSchedule& base, double floor_max)
    : base_(base), floor_max_(floor_max) {}

double ParallelFloorSchedule::beta(int step, int num_steps, int chain, int num_chains) const {
  const double floor = num_chains > 1 ? floor_max_ * static_cast<double>(chain) / (num_chains - 1) : 0.0;
  return std::max(base_.beta(step, num_steps, chain, num_chains), floor);
}
