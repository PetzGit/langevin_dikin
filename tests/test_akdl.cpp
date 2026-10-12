// Smoke tests for the kinetic Dikin-Langevin code (AKDL sampler, annealed
// AKDL optimiser and their geometry): QR factor identities, tempering, draws
// staying inside the polytope, seed reproducibility, and the optimiser
// finding an interior minimum.

#include <catch2/catch_test_macros.hpp>
#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include "cldiff/optim/tempered_objective.hpp"
#include "cldiff/optim/samplers/akdl.hpp"
#include "cldiff/optim/optimisers/annealed_akdl.hpp"
#include "cldiff/optim/schedules/three_phase_log.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace cldiff;

namespace {

// U(x) = 0.5 * ||x - c||^2
class Quadratic : public Objective {
public:
  explicit Quadratic(Eigen::VectorXd c) : c_(std::move(c)) {}
  double value(const Eigen::VectorXd& x) const override { return 0.5 * (x - c_).squaredNorm(); }
  Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override { return x - c_; }
  int dim() const override { return static_cast<int>(c_.size()); }
private:
  Eigen::VectorXd c_;
};

// The box [-1, 1]^d as Ax <= b.
Polyhedron box(int d) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * d, d);
  for (int i = 0; i < d; ++i) {
    A(2 * i, i) = 1.0;
    A(2 * i + 1, i) = -1.0;
  }
  return Polyhedron(A, Eigen::VectorXd::Ones(2 * d));
}

}  // namespace

TEST_CASE("TemperedObjective divides value and gradient by beta", "[akdl]") {
  const Quadratic inner(Eigen::Vector2d(0.3, -0.2));
  TemperedObjective tempered(inner, 0.5);
  const Eigen::VectorXd x = Eigen::Vector2d(0.1, 0.4);

  REQUIRE(tempered.dim() == 2);
  REQUIRE(std::abs(tempered.value(x) - inner.value(x) / 0.5) < 1e-14);
  REQUIRE((tempered.gradient(x) - inner.gradient(x) / 0.5).norm() < 1e-14);

  tempered.setBeta(2.0);
  REQUIRE(tempered.beta() == 2.0);
  REQUIRE(std::abs(tempered.value(x) - inner.value(x) / 2.0) < 1e-14);
}

TEST_CASE("Kinetic Dikin factors satisfy S^{-1}A = QR", "[akdl]") {
  const Polyhedron poly = box(3);
  const Eigen::Vector3d x(0.2, -0.5, 0.7);
  const kinetic_dikin::DikinFactors f = kinetic_dikin::computeFactors(poly.A(), poly.b(), x);

  const Eigen::VectorXd s = poly.b() - poly.A() * x;
  const Eigen::MatrixXd SinvA = s.cwiseInverse().asDiagonal() * poly.A();

  CAPTURE((f.Q * f.R - SinvA).norm());
  REQUIRE((f.Q * f.R - SinvA).norm() < 1e-12);
  REQUIRE((f.Q.transpose() * f.Q - Eigen::MatrixXd::Identity(3, 3)).norm() < 1e-12);
  REQUIRE((f.R.diagonal().array() > 0.0).all());

  // C = R^{-1}: applying R after C (and C^T after R^T) gives the input back.
  const Eigen::Vector3d w(0.4, -1.1, 0.3);
  REQUIRE((f.R * kinetic_dikin::applyC(f, w) - w).norm() < 1e-12);
  REQUIRE((kinetic_dikin::applyCT(f, f.R.transpose() * w) - w).norm() < 1e-12);
}

TEST_CASE("AKDLSampler draws stay in the polytope and are seed-reproducible", "[akdl]") {
  const Polyhedron poly = box(2);
  const Quadratic gaussian(Eigen::Vector2d::Zero());
  const Problem problem{gaussian, poly};
  const Eigen::Vector2d x0(0.5, -0.5);
  const int n = 2000;

  AKDLSampler sampler(0.3, 1.0, 3);
  Rng rng(7);
  const Eigen::MatrixXd draws = sampler.sample(problem, x0, rng, n);

  REQUIRE(draws.rows() == 2);
  REQUIRE(draws.cols() == n);
  for (int i = 0; i < n; ++i) REQUIRE(poly.contains(draws.col(i)));

  // The chain must actually move: some proposals accepted, and the mean of a
  // standard Gaussian truncated to the symmetric box is near zero.
  CAPTURE(sampler.acceptanceRate(), draws.rowwise().mean().transpose());
  REQUIRE(sampler.acceptanceRate() > 0.1);
  REQUIRE(draws.rowwise().mean().norm() < 0.2);

  AKDLSampler again(0.3, 1.0, 3);
  Rng rng_again(7);
  REQUIRE(again.sample(problem, x0, rng_again, n) == draws);
}

TEST_CASE("AKDLSampler warmup adapts the step size", "[akdl]") {
  const Polyhedron poly = box(2);
  const Quadratic gaussian(Eigen::Vector2d::Zero());
  AKDLWarmup warmup;
  warmup.n_warmup = 500;

  // A start far too large for the box: warmup has to shrink it.
  AKDLSampler sampler(5.0, 1.0, 3, warmup);
  Rng rng(11);
  sampler.sample({gaussian, poly}, Eigen::Vector2d::Zero(), rng, 500);

  CAPTURE(sampler.epsilon(), sampler.acceptanceRate());
  REQUIRE(sampler.epsilon() < 5.0);
  REQUIRE(sampler.acceptanceRate() > 0.5);
}

TEST_CASE("AnnealedAKDLOptimiser finds an interior minimum", "[akdl]") {
  const Polyhedron poly = box(2);
  const Eigen::Vector2d c(0.3, -0.4);
  const Quadratic objective(c);
  const ThreePhaseLogSchedule schedule(0.3);

  // r_rate > run_num: no resampling (keeps the test quiet and isolates the chains).
  AnnealedAKDLOptimiser optimiser(schedule, 1000, 400, 0.3, 1.0, 3);
  Eigen::MatrixXd x0(2, 8);
  for (int i = 0; i < 8; ++i) x0.col(i) = Eigen::Vector2d(-0.8 + 0.2 * i, 0.6);

  Rng rng(3);
  const OptimResult res = optimiser.minimise({objective, poly}, x0, rng);

  // Every start is at least 1.0 from c. The annealed stage reaches the right
  // neighbourhood but does not converge tightly (across seeds it lands
  // within ~0.1); tight convergence is the job of a finisher such as GD/LM.
  CAPTURE(res.x.transpose(), res.value);
  REQUIRE(poly.contains(res.x));
  REQUIRE((res.x - c).norm() < 0.15);
}
