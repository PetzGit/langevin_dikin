#include "cldiff/optim/optimisers/annealed_langevin.hpp"
#include <Eigen/Cholesky>
#include <stdexcept>
#include <cmath>
#include <iostream>

struct AnnealedLangevinOptimiser::Run {
  const Objective& objective;
  const Polyhedron& polyhedron;
  Rng& rng;
  Eigen::MatrixXd x;
  int step_num;
};

AnnealedLangevinOptimiser::AnnealedLangevinOptimiser(const AnnealingSchedule& schedule, int r_rate, int run_num,
                                                     double h, double eps)
    : schedule_(schedule), r_rate_(r_rate), run_num_(run_num), h_(h), eps_(eps)
{
  if (h_ <= 0 || r_rate_ < 1 || run_num_ < 1) {
    throw std::invalid_argument("Invalid arguments: check h, r_rate, run_num");
  }
}

Eigen::VectorXd AnnealedLangevinOptimiser::updateChain(Run& run, const Eigen::VectorXd& x, double beta) const {
  const Eigen::MatrixXd H_eps = run.polyhedron.barrierHessian(x, eps_);
  const Eigen::LLT<Eigen::MatrixXd> llt(H_eps);
  const Eigen::VectorXd div_c_eps = run.polyhedron.barrierHessianDivergence(llt, x);
  const Eigen::VectorXd grad_term = -1.0 * llt.solve(run.objective.gradient(x));

  const int d = x.size();
  Eigen::VectorXd n_i(d);
  for (int i = 0; i < d; i++) {
    n_i(i) = run.rng.normal();
  }
  const Eigen::VectorXd eta = llt.matrixL().transpose().solve(n_i);
  return x + h_ * (grad_term + div_c_eps * beta) + std::sqrt(2.0 * h_ * beta) * eta;
}

void AnnealedLangevinOptimiser::step(Run& run) const {
  const int chain_num = run.x.cols();
  for (int i = 0; i < chain_num; i++) {
    const double beta = schedule_.beta(run.step_num, run_num_, i, chain_num);
    Eigen::VectorXd proposal = updateChain(run, run.x.col(i), beta);
    if (!run.polyhedron.contains(proposal)) {
      continue;
    }
    run.x.col(i) = proposal;
  }
  if ((run.step_num + 1) % r_rate_ == 0) {
    resample(run);
  }
  run.step_num += 1;
}

void AnnealedLangevinOptimiser::resample(Run& run) const {
  const int N = run.x.cols();

  Eigen::VectorXd f_values(N);
  for (int i = 0; i < N; i++) {
    f_values(i) = run.objective.value(run.x.col(i));
  }
  const double f_min = f_values.minCoeff();

  Eigen::VectorXd evals(N);
  int best_ind = 0;
  for (int i = 0; i < N; i++) {
    evals(i) = std::exp(-(f_values(i) - f_min));
    if (f_values(i) < f_values(best_ind)) {
      best_ind = i;
    }
  }
  std::cout <<"Best so far: " << run.step_num << " : " << f_min << "\n";
  evals /= evals.sum();

  Eigen::VectorXd cum_sum(N);
  cum_sum(0) = evals(0);
  for (int k = 1; k < N; k++) {
    cum_sum(k) = cum_sum(k - 1) + evals(k);
  }

  Eigen::MatrixXd new_x(run.x.rows(), N);
  new_x.col(0) = run.x.col(best_ind);
  for (int j = 1; j < N; j++) {
    const double target = run.rng.uniform();
    int l = 0;
    while (l < N - 1 && cum_sum(l) < target) {
      ++l;
    }
    new_x.col(j) = run.x.col(l);
  }
  run.x = std::move(new_x);

  // todo: replace this linear scan with a binary search (std::lower_bound
  // against cum_sum) for O(log N) per draw instead of O(N).
}

OptimResult AnnealedLangevinOptimiser::minimise(const Problem& problem, const Eigen::MatrixXd& x0, Rng& rng) {
  const Polyhedron& polyhedron = problem.domain;
  if (polyhedron.size() != x0.rows() || polyhedron.size() != problem.objective.dim() || x0.cols() < 1) {
    throw std::invalid_argument("Invalid arguments: check dimensions of x0");
  }
  for (int i = 0; i < x0.cols(); ++i) {
    if (!polyhedron.contains(x0.col(i))) {
      throw std::invalid_argument("Every initial particle must be interior to the polyhedron");
    }
  }

  Run run{problem.objective, polyhedron, rng, x0, 0};
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
