#include "cldiff/optim/schedules/three_phase_log.hpp"
#include <cmath>

ThreePhaseLogSchedule::ThreePhaseLogSchedule(double rate) : rate_(rate) {}

double ThreePhaseLogSchedule::beta(int step, int num_steps, int /*chain*/, int /*num_chains*/) const {
  const int phase1_end = static_cast<int>(0.875 * num_steps);
  const int phase2_end = static_cast<int>(0.975 * num_steps);
  const double base = 1.0 / std::log(3.0 + 2.0 * rate_ * step);
  if (step < phase1_end) {
    return base;
  } else if (step < phase2_end) {
    return std::pow(base, 4.0);
  }
  return 0.0;
}
