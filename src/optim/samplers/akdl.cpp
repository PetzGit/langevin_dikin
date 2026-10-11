#include "cldiff/optim/samplers/akdl.hpp"
#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include <stdexcept>
#include <cmath>

SamplerMomentum::SamplerMomentum(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                                  Eigen::VectorXd x, double epsilon, double gamma, int L,
                                  double tau_mid, int n_mid)
    : objective_(objective), polyhedron_(polyhedron), rng_(rng), x_(std::move(x)),
      sigma_(1), epsilon_(epsilon), gamma_(gamma), L_(L), tau_mid_(tau_mid), n_mid_(n_mid)
{
  if (polyhedron_.size() != x_.size() || polyhedron_.size() != objective_.dim() ||
      !polyhedron_.contains(x_) || epsilon_ <= 0.0 || gamma_ <= 0.0 || L_ < 1) {
    throw std::invalid_argument("Invalid arguments: check dimensions, interiority, epsilon, gamma, L");
  }
  v_ = Eigen::VectorXd::Zero(x_.size());
  for (int i = 0; i < x_.size(); i++) v_(i) = rng_.normal();
}

double SamplerMomentum::stepWithEpsilon(double eps) {
  n_steps_ += 1;
  const int d = x_.size();

  // O_epsilon refresh (eq. 4.1).
  const double rho = std::exp(-gamma_ * eps);
  Eigen::VectorXd xi(d);
  for (int i = 0; i < d; i++) xi(i) = rng_.normal();
  const Eigen::VectorXd v0 = rho * v_ + std::sqrt(1.0 - rho * rho) * xi;
  const Eigen::VectorXd x0 = x_;
  const double h = sigma_ * eps;

  Eigen::VectorXd cur_x = x0, cur_v = v0;
  double Lambda_hat = 0.0;
  bool all_ok = true;
  for (int l = 0; l < L_; l++) {
    kinetic_dikin::StepResult res =
        kinetic_dikin::dstep(objective_, polyhedron_, cur_x, cur_v, h, tau_mid_, n_mid_);
    if (!res.success) { all_ok = false; break; }
    cur_x = res.x_next;
    cur_v = res.v_next;
    Lambda_hat += res.log_jacobian;
  }

  if (!all_ok) {
    x_ = x0;
    v_ = v0;
    sigma_ = -sigma_;
    return 0.0;  // matches the reference: log_alpha = -inf on failure -> accept_prob = 0
  }

  const double H0 = objective_.value(x0) + 0.5 * v0.squaredNorm();
  const double HL = objective_.value(cur_x) + 0.5 * cur_v.squaredNorm();
  const double log_alpha = H0 - HL + Lambda_hat;
  const double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_alpha)));
  const double u = rng_.uniform();
  if (u < accept_prob) {
    x_ = cur_x;
    v_ = cur_v;
    n_accept_ += 1;
    // sigma_ unchanged: accepted trajectories continue in the same direction.
  } else {
    x_ = x0;
    v_ = v0;
    sigma_ = -sigma_;
  }
  return accept_prob;
}

void SamplerMomentum::step() {
  stepWithEpsilon(epsilon_);
}

Eigen::MatrixXd SamplerMomentum::run(int n) {
  Eigen::MatrixXd trajectory(x_.size(), n);
  for (int i = 0; i < n; i++) {
    trajectory.col(i) = x_;
    step();
  }
  return trajectory;
}

double SamplerMomentum::acceptanceRate() const {
  return n_steps_ == 0 ? 0.0 : static_cast<double>(n_accept_) / static_cast<double>(n_steps_);
}

void SamplerMomentum::warmup(int n_warmup, double target_accept, double eta0, double t0,
                              double kappa, double min_epsilon, double max_epsilon) {
  double log_eps = std::log(epsilon_);
  const double log_min = std::log(min_epsilon);
  const double log_max = std::log(max_epsilon);
  for (int i = 1; i <= n_warmup; i++) {
    const double accept_prob = stepWithEpsilon(std::exp(log_eps));
    const double eta = eta0 * std::pow(static_cast<double>(i) + t0, -kappa);
    log_eps += eta * (accept_prob - target_accept);
    log_eps = std::min(std::max(log_eps, log_min), log_max);
  }
  epsilon_ = std::exp(log_eps);
  n_steps_ = 0;
  n_accept_ = 0;
}
