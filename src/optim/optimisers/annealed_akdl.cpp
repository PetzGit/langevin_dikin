#include "cldiff/optim/optimisers/annealed_akdl.hpp"
#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>

OptimiserMomentum::OptimiserMomentum(const Objective& objective, const Polyhedron& polyhedron, Rng& rng,
                                      Eigen::MatrixXd x_0, int r_rate, int chain_num, int run_num,
                                      double epsilon_base, double gamma, int L, double min_beta,
                                      double tau_mid, int n_mid)
    : objective_(objective), polyhedron_(polyhedron), rng_(rng), tempered_(objective_, 1.0),
      x_(std::move(x_0)), r_rate_(r_rate), chain_num_(chain_num), run_num_(run_num), step_num_(0),
      epsilon_base_(epsilon_base), gamma_(gamma), L_(L), min_beta_(min_beta),
      tau_mid_(tau_mid), n_mid_(n_mid)
{
  if (polyhedron_.size() != x_.rows() || polyhedron_.size() != objective_.dim() ||
      epsilon_base_ <= 0 || gamma_ <= 0 || L_ < 1 || min_beta_ <= 0) {
    throw std::invalid_argument("Invalid arguments: check dimensions, epsilon_base, gamma, L, min_beta");
  }
  for (int i = 0; i < x_.cols(); ++i) {
    if (!polyhedron_.contains(x_.col(i))) {
      throw std::invalid_argument("Every initial particle must be interior to the polyhedron");
    }
  }
  v_ = Eigen::MatrixXd::Zero(x_.rows(), chain_num_);
  for (int i = 0; i < x_.rows(); i++)
    for (int c = 0; c < chain_num_; c++)
      v_(i, c) = rng_.normal();
  sigma_ = Eigen::VectorXi::Ones(chain_num_);
  temp_ = genTemp(run_num_);
}

// Same annealing shape as Optimiser::genTemp, for a comparable schedule.
Eigen::VectorXd OptimiserMomentum::genTemp(int run_num) const {
  Eigen::VectorXd temp(run_num);
  const int phase1_end = static_cast<int>(0.875 * run_num);
  const int phase2_end = static_cast<int>(0.975 * run_num);
  for (int k = 0; k < run_num; k++) {
    const double base = 1.0 / std::log(3.0 + 2.0 * epsilon_base_ * k);
    if (k < phase1_end) {
      temp(k) = base;
    } else if (k < phase2_end) {
      temp(k) = std::pow(base, 4.0);
    } else {
      temp(k) = 0.0;
    }
  }
  return temp;
}

double OptimiserMomentum::chainBeta(int /*chain_index*/) const {
  return temp_(step_num_);
}

void OptimiserMomentum::updateChain(int chain_index, double beta) {
  const double eff_beta = std::max(beta, min_beta_);
  tempered_.setBeta(eff_beta);
  const double eps = epsilon_base_ * std::sqrt(eff_beta);

  Eigen::VectorXd x = x_.col(chain_index);
  Eigen::VectorXd v = v_.col(chain_index);
  int sigma = sigma_(chain_index);
  const int d = x.size();

  // O_epsilon refresh (eq. 4.1).
  const double rho = std::exp(-gamma_ * eps);
  Eigen::VectorXd xi(d);
  for (int i = 0; i < d; i++) xi(i) = rng_.normal();
  const Eigen::VectorXd v0 = rho * v + std::sqrt(1.0 - rho * rho) * xi;
  const Eigen::VectorXd x0 = x;
  const double h = sigma * eps;

  Eigen::VectorXd cur_x = x0, cur_v = v0;
  double Lambda_hat = 0.0;
  bool all_ok = true;
  for (int l = 0; l < L_; l++) {
    kinetic_dikin::StepResult res =
        kinetic_dikin::dstep(tempered_, polyhedron_, cur_x, cur_v, h, tau_mid_, n_mid_);
    if (!res.success) { all_ok = false; break; }
    cur_x = res.x_next;
    cur_v = res.v_next;
    Lambda_hat += res.log_jacobian;
  }

  if (!all_ok) {
    x_.col(chain_index) = x0;
    v_.col(chain_index) = v0;
    sigma_(chain_index) = -sigma;
    return;
  }

  const double H0 = tempered_.value(x0) + 0.5 * v0.squaredNorm();
  const double HL = tempered_.value(cur_x) + 0.5 * cur_v.squaredNorm();
  const double log_alpha = H0 - HL + Lambda_hat;
  const double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_alpha)));
  const double u = rng_.uniform();
  if (u < accept_prob) {
    x_.col(chain_index) = cur_x;
    v_.col(chain_index) = cur_v;
    // sigma unchanged: accepted trajectories continue in the same direction.
  } else {
    x_.col(chain_index) = x0;
    v_.col(chain_index) = v0;
    sigma_(chain_index) = -sigma;
  }
}

void OptimiserMomentum::step() {
  for (int i = 0; i < chain_num_; i++) {
    updateChain(i, chainBeta(i));
  }
  if ((step_num_ + 1) % r_rate_ == 0) {
    resample();
  }
  step_num_ += 1;
}

void OptimiserMomentum::resample() {
  const int N = x_.cols();
  Eigen::VectorXd f_values(N);
  for (int i = 0; i < N; i++) f_values(i) = objective_.value(x_.col(i));
  const double f_min = f_values.minCoeff();

  Eigen::VectorXd evals(N);
  int best_ind = 0;
  for (int i = 0; i < N; i++) {
    evals(i) = std::exp(-(f_values(i) - f_min));
    if (f_values(i) < f_values(best_ind)) best_ind = i;
  }
  std::cout << "Best so far (momentum): " << step_num_ << " : " << f_min << "\n";
  evals /= evals.sum();

  Eigen::VectorXd cum_sum(N);
  cum_sum(0) = evals(0);
  for (int k = 1; k < N; k++) cum_sum(k) = cum_sum(k - 1) + evals(k);

  Eigen::MatrixXd new_x(x_.rows(), N), new_v(v_.rows(), N);
  Eigen::VectorXi new_sigma(N);
  new_x.col(0) = x_.col(best_ind);
  new_v.col(0) = v_.col(best_ind);
  new_sigma(0) = sigma_(best_ind);
  for (int j = 1; j < N; j++) {
    const double target = rng_.uniform();
    int l = 0;
    while (l < N - 1 && cum_sum(l) < target) ++l;
    new_x.col(j) = x_.col(l);
    new_v.col(j) = v_.col(l);
    new_sigma(j) = sigma_(l);
  }
  x_ = std::move(new_x);
  v_ = std::move(new_v);
  sigma_ = std::move(new_sigma);
}

ProposalResult OptimiserMomentum::run() {
  while (step_num_ < run_num_) {
    step();
  }
  double best_eval = objective_.value(x_.col(0));
  int best_ind = 0;
  for (int i = 1; i < x_.cols(); i++) {
    const double val = objective_.value(x_.col(i));
    if (val < best_eval) {
      best_eval = val;
      best_ind = i;
    }
  }
  return ProposalResult{x_.col(best_ind), best_eval};
}
