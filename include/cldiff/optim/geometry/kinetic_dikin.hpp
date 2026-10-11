#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"
#include "cldiff/optim/polyhedron.hpp"

namespace cldiff {

// Shared geometry and integrator for kinetic Dikin-Langevin dynamics, per
// Chok, Petzinna & Vasil, "Kinetic Dikin-Langevin Dynamics on Polytopes"
// (2026). Used by both AKDLSampler (AKDL, Algorithm 4) and
// AnnealedAKDLOptimiser (annealed AKDL chains).
//
// Geometry: for slacks s(x) = b - Ax, the thin QR factorization
// S(x)^{-1} A = Q(x) R(x) (Q orthonormal columns, R upper triangular with
// positive diagonal) gives the barrier Hessian G = R^T R and the triangular
// factor C = R^{-1} (Section 4.3.1). C(x)v is the position velocity; div C
// and the volume-correction matrix B(x;w) follow the explicit QR formulas of
// Appendix A, avoiding any d x d x d derivative tensor.
namespace kinetic_dikin {

struct DikinFactors {
  Eigen::MatrixXd Q;  // nc x d, orthonormal columns
  Eigen::MatrixXd R;  // d x d, upper triangular, positive diagonal
};

// Computes S(x)^{-1}A = QR with a positive-diagonal sign convention.
DikinFactors computeFactors(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                             const Eigen::VectorXd& x);

// C(x)*w and C(x)^T*u via triangular solves against R (C = R^{-1}).
Eigen::VectorXd applyC(const DikinFactors& f, const Eigen::VectorXd& w);
Eigen::VectorXd applyCT(const DikinFactors& f, const Eigen::VectorXd& u);

// (div C(x))_j = -sum_r q_rj [q_rj^2 + 2 sum_{k<j} q_rk^2]  (eq. A.3)
Eigen::VectorXd divC(const DikinFactors& f);

// F(x) = div C(x) - C(x)^T grad U(x)  (eq. 2.2)
Eigen::VectorXd force(const DikinFactors& f, const Eigen::VectorXd& gradU);

// B(x;w) = That(w)^T Q(x), That(w)_rj = q_rj(q_rj w_j + 2 sum_{k>j} q_rk w_k)
// (eq. 4.8). Used for both the midpoint-solve Jacobian and the volume
// (log-Jacobian) correction.
Eigen::MatrixXd computeB(const DikinFactors& f, const Eigen::VectorXd& w);

// J_g(x;w) = -C(x) B(x;w) C(x)^{-1} = -C(x) B(x;w) R(x)  (eq. A.9)
Eigen::MatrixXd computeJg(const DikinFactors& f, const Eigen::MatrixXd& B);

// Sufficient feasibility certificate, |h|/2 * sqrt(2d) * ||w|| < 1 (eq. 3.2).
bool certificateOk(int d, double h, const Eigen::VectorXd& w);

struct MidpointResult {
  Eigen::VectorXd m;
  DikinFactors factors;  // geometry at the converged midpoint m
  bool success = false;
};

// Modified-Newton solve of m = x + (h/2)C(m)w with the derivative frozen at
// the initial guess m0 (eq. 4.9-4.11): one LU factorization is reused for
// every correction. Requires the certificate to already hold for (h,w).
MidpointResult solveMidpoint(const Polyhedron& polyhedron, const Eigen::VectorXd& x,
                              const Eigen::VectorXd& w, double h, double tau_mid, int n_mid);

struct StepResult {
  Eigen::VectorXd x_next;
  Eigen::VectorXd v_next;
  double log_jacobian = 0.0;  // lambda_hat, eq. 4.6
  bool success = false;
};

// One deterministic step D_h(x,v) (eq. 3.1 / Algorithm 2): a velocity
// half-kick, an implicit-midpoint position step, and a second half-kick.
// Fails (success=false) if the certificate, the midpoint solve, or either
// feasibility/positive-determinant check fails.
StepResult dstep(const Objective& objective, const Polyhedron& polyhedron,
                  const Eigen::VectorXd& x, const Eigen::VectorXd& v, double h,
                  double tau_mid, int n_mid);

}  // namespace kinetic_dikin

}  // namespace cldiff
