#pragma once
#include "cldiff/optim/annealing_schedule.hpp"

namespace cldiff {

// Wraps another schedule with per-chain temperature floors spread linearly
// over [0, floor_max]: chain 0 cools fully (refiner), the last chain never
// drops below floor_max (explorer). See the rationale formerly in
// optimiser_parallel_annealing.hpp.
class ParallelFloorSchedule final : public AnnealingSchedule {
public:
  ParallelFloorSchedule(const AnnealingSchedule& base, double floor_max);
  double beta(int step, int num_steps, int chain, int num_chains) const override;
private:
  const AnnealingSchedule& base_;
  double floor_max_;
};

}  // namespace cldiff
