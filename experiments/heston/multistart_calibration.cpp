// experiments/heston/multistart_calibration.cpp
//
// Compares plain gradient descent against the annealed Optimiser across
// several diverse starting points, on the same synthetic market. The point:
// show whether the annealed method is actually more robust to bad
// initialization than a local method -- which is the whole justification
// for building it.
//
// Added: a GD "polish" pass run from the Optimiser's own result. This is
// the actual test of "did the Optimiser find a better basin than GD could
// reach on its own" -- if polishing drives the loss down near the true
// minimum, the Optimiser did real global work; if the polish loss stays
// near result.value, the Optimiser landed in the same local minimum GD
// would have found anyway, and the two methods aren't meaningfully
// different on that starting point.
//
// Also added: a deliberately far starting point (index 4), further from
// true_params than any of the originals, to stress-test basin escape
// harder than the existing set.

#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"
#include "cldiff/models/black_scholes/black_scholes.hpp"
#include "cldiff/market/market_data.hpp"
#include "cldiff/calibration/heston_objective.hpp"
#include "cldiff/calibration/corrected_log_space_heston_objective.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/optimisers/annealed_langevin.hpp"
#include "cldiff/optim/schedules/three_phase_log.hpp"
#include "cldiff/optim/optimisers/projected_gradient_descent.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <vector>


int main() {
  const double S0 = 100.0;
  const double r  = 0.02;
  const int    term_num = 128;
  const double L = 10.0;
  const double lambda_feller = 1.0;

  Eigen::VectorXd true_params(5);
  true_params << 2.0, 0.04, 0.30, -0.70, 0.04;

  CharacteristicHeston phi_true(true_params);

  struct Point { double T; double K; };
  std::vector<Point> points;
  std::vector<double> maturities = {0.25, 0.5, 1.0, 1.5, 2.0};
  std::vector<double> strikes    = {80.0, 90.0, 100.0, 110.0, 120.0};
  for (double T : maturities) {
    for (double K : strikes) {
      points.push_back({T, K});
    }
  }

  MarketData market(S0, r);
  for (const auto& pt : points) {
    Cumulants cum = cumulants(phi_true, pt.T);
    Range range = make_range(cum, L);
    double price = cos_pricer(phi_true, range, pt.K, pt.T, r, term_num, S0);
    market.addQuote(pt.T, pt.K, price);
  }

  HestonObjective objective(market, term_num, L, lambda_feller);
  CorrectedLogSpaceHestonObjective log_objective(objective);

  // Box, in log-space, shared across all Optimiser runs.
  Eigen::MatrixXd A(10, 5);
  Eigen::VectorXd b(10);
  A.setZero();
  auto setBounds = [&](int row, int col, double lo, double hi) {
    A(row, col) = -1.0;      b(row) = -lo;
    A(row + 1, col) = 1.0;   b(row + 1) = hi;
  };
  setBounds(0, 0, -5.0, 3.0);      // log(kappa): ~0.0067 to ~20.1
  setBounds(2, 1, -5.0, 0.4);      // log(theta): ~0.0067 to ~1.49
  setBounds(4, 2, -5.0, 1.5);      // log(xi):    ~0.0067 to ~4.48
  setBounds(6, 3, -0.999, 0.999);  // rho
  setBounds(8, 4, -5.0, 0.4);      // log(v0):    ~0.0067 to ~1.49
  Polyhedron polyhedron(A, b);

  // Diverse starting points, in physical units.
  std::vector<Eigen::VectorXd> starts;
  auto addStart = [&](double k, double t, double x, double rh, double v) {
    Eigen::VectorXd p(5);
    p << k, t, x, rh, v;
    starts.push_back(p);
  };
  addStart(4.0, 0.10, 0.10, 0.10, 0.10);   // the one that worked for plain GD
  addStart(0.5, 0.01, 1.50, -0.90, 0.01);
  addStart(8.0, 0.50, 0.05, 0.50, 0.50);
  addStart(1.0, 0.04, 0.30, -0.70, 0.04);  // near truth, sanity check
  addStart(6.0, 0.02, 1.20, 0.80, 0.30);   // deliberately far: high kappa, low
                                            // theta, high xi, WRONG-SIGN rho
                                            // (true rho is -0.70), high v0 --
                                            // harder than the existing points

  const int chain_num = 5;
  const int run_num = 1000;   // trimmed down per current test run
  const int r_rate = 20;
  const double h_opt = 1e-2;
  const double eps_opt = 1e-6;

  const int polish_iters = 1000;
  const int polish_backtracks = 30;

  std::printf("%-3s %-42s %-13s %-13s %-13s %-9s %-9s\n",
              "#", "start (kappa,theta,xi,rho,v0)",
              "GD loss", "Opt loss", "Opt+Polish", "GD stuck", "Pol stuck");

  for (size_t i = 0; i < starts.size(); i++) {
    // --- Gradient descent, direct from the raw starting point ---
    // GD is deterministic and never draws from rng, so creating it first
    // leaves the annealed optimiser's random stream unchanged.
    Rng rng(1000 + static_cast<std::uint64_t>(i));
    ProjectedGradientDescent gd(500, 30);
    const double gd_loss =
        gd.minimise({log_objective, polyhedron}, CorrectedLogSpaceHestonObjective::toLogSpace(starts[i]), rng).value;
    const bool stuck = gd.stuck();

    // --- Annealed Optimiser ---
    Eigen::VectorXd y0 = CorrectedLogSpaceHestonObjective::toLogSpace(starts[i]);
    Eigen::MatrixXd x_0(5, chain_num);
    for (int c = 0; c < chain_num; c++) {
      Eigen::VectorXd col = y0;
      for (int j = 0; j < 5; j++) {
        col(j) += 0.1 * rng.normal();
      }
      if (!polyhedron.contains(col)) {
        col = y0;
      }
      x_0.col(c) = col;
    }
    ThreePhaseLogSchedule schedule(h_opt);
    AnnealedLangevinOptimiser optimiser(schedule, r_rate, run_num, h_opt, eps_opt);
    OptimResult result = optimiser.minimise({log_objective, polyhedron}, x_0, rng);

    // --- GD polish, starting from the Optimiser's own result ---
    // This is the actual test: did the Optimiser find a genuinely better
    // basin than GD reaches on its own? If polish loss << Opt loss and
    // approaches the true minimum, yes. If polish loss stays close to
    // Opt loss, the Optimiser converged to the same (possibly local)
    // minimum a local method would have found from there anyway.
    ProjectedGradientDescent polish(polish_iters, polish_backtracks);
    const double polish_loss = polish.minimise({log_objective, polyhedron}, result.x, rng).value;
    const bool polish_stuck = polish.stuck();

    std::printf("%-3zu [%.2f, %.3f, %.2f, %.2f, %.3f]  %-13.4e %-13.4e %-13.4e %-9s %-9s\n",
                i, starts[i](0), starts[i](1), starts[i](2), starts[i](3), starts[i](4),
                gd_loss, result.value, polish_loss,
                stuck ? "yes" : "no", polish_stuck ? "yes" : "no");
  }

  return 0;
}
