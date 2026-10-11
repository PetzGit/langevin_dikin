#include "cldiff/optim/samplers/akdl.hpp"
#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace cldiff {

struct AKDLSampler::Run {
  const Objective& objective;
  const Polyhedron& polyhedron;
  Rng& rng;
  Eigen::VectorXd x;
  Eigen::VectorXd v;
  int sigma;
};

AKDLSampler::AKDLSampler(double epsilon, double gamma, int L, AKDLWarmup warmup,
                         double tau_mid, int n_mid)
    : epsilon_init_(epsilon), gamma_(gamma), L_(L), warmup_(warmup), tau_mid_(tau_mid), n_mid_(n_mid),
      epsilon_(epsilon)
{
  if (epsilon_init_ <= 0.0 || gamma_ <= 0.0 || L_ < 1 || warmup_.n_warmup < 0) {
    throw std::invalid_argument("Invalid arguments: check epsilon, gamma, L, n_warmup");
  }
}

double AKDLSampler::stepWithEpsilon(Run& run, double eps) {
  n_steps_ += 1;
  const int d = run.x.size();

  // O_epsilon refresh (eq. 4.1).
  const double rho = std::exp(-gamma_ * eps);
  Eigen::VectorXd xi(d);
  for (int i = 0; i < d; i++) xi(i) = run.rng.normal();
  const Eigen::VectorXd v0 = rho * run.v + std::sqrt(1.0 - rho * rho) * xi;
  const Eigen::VectorXd x0 = run.x;
  const double h = run.sigma * eps;

  Eigen::VectorXd cur_x = x0, cur_v = v0;
  double Lambda_hat = 0.0;
  bool all_ok = true;
  for (int l = 0; l < L_; l++) {
    kinetic_dikin::StepResult res =
        kinetic_dikin::dstep(run.objective, run.polyhedron, cur_x, cur_v, h, tau_mid_, n_mid_);
    if (!res.success) { all_ok = false; break; }
    cur_x = res.x_next;
    cur_v = res.v_next;
    Lambda_hat += res.log_jacobian;
  }

  if (!all_ok) {
    run.x = x0;
    run.v = v0;
    run.sigma = -run.sigma;
    return 0.0;  // matches the reference: log_alpha = -inf on failure -> accept_prob = 0
  }

  const double H0 = run.objective.value(x0) + 0.5 * v0.squaredNorm();
  const double HL = run.objective.value(cur_x) + 0.5 * cur_v.squaredNorm();
  const double log_alpha = H0 - HL + Lambda_hat;
  const double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_alpha)));
  const double u = run.rng.uniform();
  if (u < accept_prob) {
    run.x = cur_x;
    run.v = cur_v;
    n_accept_ += 1;
    // sigma unchanged: accepted trajectories continue in the same direction.
  } else {
    run.x = x0;
    run.v = v0;
    run.sigma = -run.sigma;
  }
  return accept_prob;
}

Eigen::MatrixXd AKDLSampler::sample(const Problem& problem, const Eigen::VectorXd& x0, Rng& rng, int n) {
  if (problem.domain.dimension() != x0.size() || problem.domain.dimension() != problem.objective.dim() ||
      !problem.domain.contains(x0)) {
    throw std::invalid_argument("Invalid arguments: check dimensions and interiority of x0");
  }
  Run run{problem.objective, problem.domain, rng, x0, Eigen::VectorXd::Zero(x0.size()), 1};
  for (int i = 0; i < x0.size(); i++) run.v(i) = rng.normal();
  n_steps_ = 0;
  n_accept_ = 0;

  double log_eps = std::log(epsilon_init_);
  const double log_min = std::log(warmup_.min_epsilon);
  const double log_max = std::log(warmup_.max_epsilon);
  for (int i = 1; i <= warmup_.n_warmup; i++) {
    const double accept_prob = stepWithEpsilon(run, std::exp(log_eps));
    const double eta = warmup_.eta0 * std::pow(static_cast<double>(i) + warmup_.t0, -warmup_.kappa);
    log_eps += eta * (accept_prob - warmup_.target_accept);
    log_eps = std::min(std::max(log_eps, log_min), log_max);
  }
  epsilon_ = warmup_.n_warmup > 0 ? std::exp(log_eps) : epsilon_init_;
  n_steps_ = 0;
  n_accept_ = 0;

  Eigen::MatrixXd trajectory(x0.size(), n);
  for (int i = 0; i < n; i++) {
    trajectory.col(i) = run.x;
    stepWithEpsilon(run, epsilon_);
  }
  return trajectory;
}

double AKDLSampler::acceptanceRate() const {
  return n_steps_ == 0 ? 0.0 : static_cast<double>(n_accept_) / static_cast<double>(n_steps_);
}

}  // namespace cldiff
