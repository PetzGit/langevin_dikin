#pragma once

// Temperature assigned to a chain at a step. Replaces the chainBeta/genTemp
// hooks that were duplicated across Optimiser and OptimiserMomentum, so a
// schedule is written once and works with every annealed optimiser.
// Stateless: everything it may depend on is passed in.
class AnnealingSchedule {
public:
  virtual ~AnnealingSchedule() = default;
  virtual double beta(int step, int num_steps, int chain, int num_chains) const = 0;
};
