#include "cldiff/optim/optimisers/annealed_akdl.hpp"
#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include "cldiff/optim/tempered_objective.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace cldiff {

struct AnnealedAKDLOptimiser::Run {
  const Objective& objective;
  const Polyhedron& polyhedron;
  Rng& rng;
  TemperedObjective tempered;
  Eigen::MatrixXd x;  // d x chain_num
  Eigen::MatrixXd v;  // d x chain_num
  Eigen::VectorXi sigma;  // chain_num, direction in {-1,+1}
  int step_num;
};

AnnealedAKDLOptimiser::AnnealedAKDLOptimiser(const AnnealingSchedule& schedule, int r_rate, int run_num,
                                             double epsilon_base, double gamma, int L, double min_beta,
                                             double tau_mid, int n_mid)
    : schedule_(schedule), r_rate_(r_rate), run_num_(run_num),
      epsilon_base_(epsilon_base), gamma_(gamma), L_(L), min_beta_(min_beta),
      tau_mid_(tau_mid), n_mid_(n_mid)
{
  if (epsilon_base_ <= 0 || gamma_ <= 0 || L_ < 1 || min_beta_ <= 0 || r_rate_ < 1 || run_num_ < 1) {
    throw std::invalid_argument("Invalid arguments: check epsilon_base, gamma, L, min_beta, r_rate, run_num");
  }
}

void AnnealedAKDLOptimiser::updateChain(Run& run, int chain_index, double beta) const {
  const double eff_beta = std::max(beta, min_beta_);
  run.tempered.setBeta(eff_beta);
  const double eps = epsilon_base_ * std::sqrt(eff_beta);

  Eigen::VectorXd x = run.x.col(chain_index);
  Eigen::VectorXd v = run.v.col(chain_index);
  int sigma = run.sigma(chain_index);
  const int d = x.size();

  // O_epsilon refresh (eq. 4.1).
  const double rho = std::exp(-gamma_ * eps);
  Eigen::VectorXd xi(d);
  for (int i = 0; i < d; i++) xi(i) = run.rng.normal();
  const Eigen::VectorXd v0 = rho * v + std::sqrt(1.0 - rho * rho) * xi;
  const Eigen::VectorXd x0 = x;
  const double h = sigma * eps;

  Eigen::VectorXd cur_x = x0, cur_v = v0;
  double Lambda_hat = 0.0;
  bool all_ok = true;
  for (int l = 0; l < L_; l++) {
    kinetic_dikin::StepResult res =
        kinetic_dikin::dstep(run.tempered, run.polyhedron, cur_x, cur_v, h, tau_mid_, n_mid_);
    if (!res.success) { all_ok = false; break; }
    cur_x = res.x_next;
    cur_v = res.v_next;
    Lambda_hat += res.log_jacobian;
  }

  if (!all_ok) {
    run.x.col(chain_index) = x0;
    run.v.col(chain_index) = v0;
    run.sigma(chain_index) = -sigma;
    return;
  }

  const double H0 = run.tempered.value(x0) + 0.5 * v0.squaredNorm();
  const double HL = run.tempered.value(cur_x) + 0.5 * cur_v.squaredNorm();
  const double log_alpha = H0 - HL + Lambda_hat;
  const double accept_prob = std::min(1.0, std::exp(std::min(0.0, log_alpha)));
  const double u = run.rng.uniform();
  if (u < accept_prob) {
    run.x.col(chain_index) = cur_x;
    run.v.col(chain_index) = cur_v;
    // sigma unchanged: accepted trajectories continue in the same direction.
  } else {
    run.x.col(chain_index) = x0;
    run.v.col(chain_index) = v0;
    run.sigma(chain_index) = -sigma;
  }
}

void AnnealedAKDLOptimiser::step(Run& run) const {
  const int chain_num = run.x.cols();
  for (int i = 0; i < chain_num; i++) {
    updateChain(run, i, schedule_.beta(run.step_num, run_num_, i, chain_num));
  }
  if ((run.step_num + 1) % r_rate_ == 0) {
    resample(run);
  }
  run.step_num += 1;
}

void AnnealedAKDLOptimiser::resample(Run& run) const {
  const int N = run.x.cols();
  Eigen::VectorXd f_values(N);
  for (int i = 0; i < N; i++) f_values(i) = run.objective.value(run.x.col(i));
  const double f_min = f_values.minCoeff();

  Eigen::VectorXd evals(N);
  int best_ind = 0;
  for (int i = 0; i < N; i++) {
    evals(i) = std::exp(-(f_values(i) - f_min));
    if (f_values(i) < f_values(best_ind)) best_ind = i;
  }
  std::cout << "Best so far (momentum): " << run.step_num << " : " << f_min << "\n";
  evals /= evals.sum();

  Eigen::VectorXd cum_sum(N);
  cum_sum(0) = evals(0);
  for (int k = 1; k < N; k++) cum_sum(k) = cum_sum(k - 1) + evals(k);

  Eigen::MatrixXd new_x(run.x.rows(), N), new_v(run.v.rows(), N);
  Eigen::VectorXi new_sigma(N);
  new_x.col(0) = run.x.col(best_ind);
  new_v.col(0) = run.v.col(best_ind);
  new_sigma(0) = run.sigma(best_ind);
  for (int j = 1; j < N; j++) {
    const double target = run.rng.uniform();
    int l = 0;
    while (l < N - 1 && cum_sum(l) < target) ++l;
    new_x.col(j) = run.x.col(l);
    new_v.col(j) = run.v.col(l);
    new_sigma(j) = run.sigma(l);
  }
  run.x = std::move(new_x);
  run.v = std::move(new_v);
  run.sigma = std::move(new_sigma);
}

OptimResult AnnealedAKDLOptimiser::minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) {
  const Polyhedron& polyhedron = problem.domain;
  if (polyhedron.dimension() != x0.rows() || polyhedron.dimension() != problem.objective.dim() || x0.cols() < 1) {
    throw std::invalid_argument("Invalid arguments: check dimensions of x0");
  }
  for (int i = 0; i < x0.cols(); ++i) {
    if (!polyhedron.contains(x0.col(i))) {
      throw std::invalid_argument("Every initial particle must be interior to the polyhedron");
    }
  }

  const int chain_num = x0.cols();
  Run run{problem.objective, polyhedron, rng, TemperedObjective(problem.objective, 1.0), x0,
          Eigen::MatrixXd::Zero(x0.rows(), chain_num), Eigen::VectorXi::Ones(chain_num), 0};
  for (int i = 0; i < x0.rows(); i++)
    for (int c = 0; c < chain_num; c++)
      run.v(i, c) = rng.normal();

  while (run.step_num < run_num_) {
    step(run);
  }
  double best_eval = run.objective.value(run.x.col(0));
  int best_ind = 0;
  for (int i = 1; i < run.x.cols(); i++) {
    const double val = run.objective.value(run.x.col(i));
    if (val < best_eval) {
      best_eval = val;
      best_ind = i;
    }
  }
  return OptimResult{run.x.col(best_ind), best_eval};
}

}  // namespace cldiff
