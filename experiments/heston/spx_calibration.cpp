// Calibrates Heston to a real SPX option-chain snapshot produced by
// data_loader/data_loader.py (dat.csv + meta.txt), using the corrected
// log-space objective, chain_num=50/run_num=1000/r_rate=20 Optimiser, then a
// box-constrained GD polish.
//
// Run from data_loader/ (so dat.csv/meta.txt resolve from cwd):
//   cd data_loader && ../build/experiments/heston_spx_calibration
#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"
#include "cldiff/models/black_scholes/black_scholes.hpp"
#include "cldiff/market/market_data.hpp"
#include "cldiff/calibration/heston_objective.hpp"
#include "cldiff/calibration/corrected_log_space_heston_objective.hpp"
#include "cldiff/models/heston/feller.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/optimisers/annealed_langevin.hpp"
#include "cldiff/optim/schedules/three_phase_log.hpp"
#include "cldiff/optim/optimisers/projected_gradient_descent.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <fstream>


int main() {
  std::ifstream meta("meta.txt");
  std::string key;
  double S0 = 0, r = 0, q = 0;
  while (meta >> key) {
    if (key == "S0") meta >> S0;
    else if (key == "r") meta >> r;
    else if (key == "q") meta >> q;
  }
  MarketData market(S0, r, q);
  market.loadFromCsv("dat.csv");
  double n_quotes = 0;
  for (auto& [T, quotes] : market.quotesByMaturity()) n_quotes += quotes.size();

  const int term_num = 128;
  const double L = 10.0, lambda_feller = 1.0;
  HestonObjective objective(market, term_num, L, lambda_feller);
  CorrectedLogSpaceHestonObjective log_objective(objective);

  Eigen::MatrixXd A(10, 5);
  Eigen::VectorXd b(10);
  A.setZero();
  auto setBounds = [&](int row, int col, double lo, double hi) {
    A(row, col) = -1.0;      b(row) = -lo;
    A(row + 1, col) = 1.0;   b(row + 1) = hi;
  };
  setBounds(0, 0, -5.0, 3.0);
  setBounds(2, 1, -5.0, 0.4);
  setBounds(4, 2, -5.0, 1.5);
  setBounds(6, 3, -0.999, 0.999);
  setBounds(8, 4, -5.0, 0.4);
  Polyhedron polyhedron(A, b);

  const int chain_num = 50, run_num = 1000, r_rate = 20;
  const double h_opt = 1e-2, eps_opt = 1e-6;
  const int polish_iters = 500, polish_backtracks = 30;

  Eigen::VectorXd start(5);
  start << 2.0, 0.04, 0.5, -0.7, 0.04;
  Rng rng(4002);
  Eigen::VectorXd y0 = CorrectedLogSpaceHestonObjective::toLogSpace(start);
  Eigen::MatrixXd x_0(5, chain_num);
  for (int c = 0; c < chain_num; c++) {
    Eigen::VectorXd col = y0;
    for (int j = 0; j < 5; j++) col(j) += 0.3 * rng.normal();
    if (!polyhedron.contains(col)) col = y0;
    x_0.col(c) = col;
  }
  ThreePhaseLogSchedule schedule(h_opt);
  AnnealedLangevinOptimiser optimiser(schedule, r_rate, run_num, h_opt, eps_opt);
  OptimResult result = optimiser.minimise({log_objective, polyhedron}, x_0, rng);

  ProjectedGradientDescent polish(polish_iters, polish_backtracks);
  OptimResult polished = polish.minimise({log_objective, polyhedron}, result.x, rng);
  const double polish_loss = polished.value;
  const bool stuck = polish.stuck();
  Eigen::VectorXd phys = CorrectedLogSpaceHestonObjective::toPhysical(polished.x);

  std::printf("S0=%.2f r=%.4f q=%.4f  quotes=%d\n", S0, r, q, (int)n_quotes);
  std::printf("opt_loss=%.4e polish_loss=%.6e RMSE=%.4f vol pts  polish_stuck=%s Feller=%s\n",
              result.value, polish_loss, std::sqrt(polish_loss / n_quotes) * 100.0,
              stuck ? "yes" : "no", satisfiesFeller(phys) ? "yes" : "no");
  const char* names[5] = {"kappa", "theta", "xi", "rho", "v0"};
  for (int i = 0; i < 5; i++) std::printf("  %-8s = %.4f\n", names[i], phys(i));
  return 0;
}
