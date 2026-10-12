#pragma once
#include <cstddef>
#include "cldiff/core/estimator.hpp"
#include "cldiff/pricing/monte_carlo/sde_scheme.hpp"
#include "cldiff/products/observable.hpp"
#include "cldiff/core/rng.hpp"
#include <vector>

namespace cldiff {

struct MonteCarloResult {
  std::vector<Estimator> outputs;
  std::size_t n_nonfinite = 0;
};

class MonteCarloHarness {
  public:
    explicit MonteCarloHarness(const Eigen::VectorXd& x0, const SDEScheme& scheme, Observable& obs, double T, int steps, std::size_t n_paths);
    MonteCarloResult run(Rng& rng);
  private:
    Eigen::VectorXd x0_;
    const SDEScheme& scheme_;
    Observable& obs_;
    int n_steps_;
    std::size_t n_paths_;
    double dt_;
};
}
