#include "cldiff/pricing/monte_carlo/mc_harness.hpp"
#include <stdexcept>
#include <cmath>
namespace cldiff {
MonteCarloHarness::MonteCarloHarness(const Eigen::VectorXd& x0, const SDEScheme& scheme, Observable& obs, double T, int steps, std::size_t n_paths)
: x0_(x0), scheme_(scheme), obs_(obs), n_paths_(n_paths) {
  if (T <= 0 || steps <= 0 || x0_.size() != scheme_.dimension()) {  
    throw std::invalid_argument("Check input parameters");
  }
  n_steps_ = steps;
  dt_ = T / steps;
}

MonteCarloResult MonteCarloHarness::run(Rng& rng, PathStore* store) {
  const bool step_flag = obs_.needsSteps();
  MonteCarloResult res;
  res.outputs.resize(obs_.numOutputs());
  Eigen::VectorXd state(x0_.size());
  if (store) {
    store->valid.assign(n_paths_, 1);
    store->states.assign(n_steps_ + 1, Eigen::MatrixXd(n_paths_, x0_.size()));
    for (std::size_t i{0}; i < n_paths_; i++) {
      store->states[0].row(i) = x0_.transpose();
    }
  }
  for (std::size_t i{0}; i < n_paths_; i++) {
    state = x0_;
    obs_.onStart(state);
    for (int j{0}; j < n_steps_; j++) {
      scheme_.step(state, dt_, rng);
      if (step_flag) {
        obs_.onStep(j, (j+1) * dt_, state);
      }
      if (store) {
        store->states[j+1].row(i) = state.transpose(); 
      }
    }
    const Eigen::VectorXd out = obs_.result(state);
    if (out.size() != obs_.numOutputs()) {
      throw std::logic_error("MonteCarloHarness: result() size != numOutputs()");
    }
    if (out.allFinite()) {
      for (Eigen::Index k = 0; k < out.size(); ++k) {
        res.outputs[static_cast<std::size_t>(k)].add(out(k));
      }
    }
    else {
      ++res.n_nonfinite;
    }
  if (store) store->valid[i] = out.allFinite();
  }
  return res;
}
}
