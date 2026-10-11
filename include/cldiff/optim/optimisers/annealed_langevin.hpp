#pragma once
#include <string>
#include <Eigen/Dense>
#include "cldiff/optim/optimiser.hpp"
#include "cldiff/optim/annealing_schedule.hpp"

namespace cldiff {

// Annealed Langevin-on-polyhedron optimiser: every column of x0 is a chain
// run through overdamped Dikin-Langevin dynamics at the temperature the
// schedule assigns it, with fitness resampling (best chain kept) every
// r_rate steps. The temperature strategy is pluggable via AnnealingSchedule
// -- ThreePhaseLogSchedule(h) reproduces the original shared schedule,
// wrapping it in ParallelFloorSchedule reproduces ParallelAnnealingOptimiser.
class AnnealedLangevinOptimiser final : public Optimiser {
public:
  AnnealedLangevinOptimiser(const AnnealingSchedule& schedule, int r_rate, int run_num, double h, double eps);
  OptimResult minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) override;
  std::string name() const override { return "annealed_langevin"; }
private:
  struct Run;
  // todo: abstract this into a "chain" object which we feed into Sampler and Optimiser.
  // Currently just reimplementing the proposeState and proposal logic.
  Eigen::VectorXd updateChain(Run& run, const Eigen::VectorXd& x, double beta) const;
  void step(Run& run) const;
  void resample(Run& run) const;
  const AnnealingSchedule& schedule_;
  int r_rate_;
  int run_num_;
  double h_;
  double eps_;
};

}  // namespace cldiff
