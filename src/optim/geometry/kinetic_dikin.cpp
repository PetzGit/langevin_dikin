#include "cldiff/optim/geometry/kinetic_dikin.hpp"
#include <Eigen/QR>
#include <Eigen/LU>
#include <cmath>

namespace cldiff {

namespace kinetic_dikin {

DikinFactors computeFactors(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                             const Eigen::VectorXd& x) {
  const int nc = A.rows();
  const int d = A.cols();
  const Eigen::VectorXd s = b - A * x;
  const Eigen::MatrixXd M = s.asDiagonal().inverse() * A;  // S(x)^{-1} A

  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(nc, d);
  Eigen::MatrixXd R = qr.matrixQR().topRows(d).triangularView<Eigen::Upper>();

  // Sign convention: positive diagonal on R. Flipping a column of Q together
  // with the matching row of R leaves Q*R unchanged (the two sign flips
  // cancel in every product term that uses column/row j of Q/R).
  for (int j = 0; j < d; j++) {
    if (R(j, j) < 0.0) {
      R.row(j) *= -1.0;
      Q.col(j) *= -1.0;
    }
  }
  return DikinFactors{Q, R};
}

Eigen::VectorXd applyC(const DikinFactors& f, const Eigen::VectorXd& w) {
  return f.R.triangularView<Eigen::Upper>().solve(w);
}

Eigen::VectorXd applyCT(const DikinFactors& f, const Eigen::VectorXd& u) {
  return f.R.transpose().triangularView<Eigen::Lower>().solve(u);
}

Eigen::VectorXd divC(const DikinFactors& f) {
  const int nc = f.Q.rows();
  const int d = f.Q.cols();
  Eigen::VectorXd result = Eigen::VectorXd::Zero(d);
  for (int r = 0; r < nc; r++) {
    double cumsum_sq = 0.0;  // sum_{k<j} q_rk^2, built incrementally
    for (int j = 0; j < d; j++) {
      const double qrj = f.Q(r, j);
      result(j) -= qrj * (qrj * qrj + 2.0 * cumsum_sq);
      cumsum_sq += qrj * qrj;
    }
  }
  return result;
}

Eigen::VectorXd force(const DikinFactors& f, const Eigen::VectorXd& gradU) {
  return divC(f) - applyCT(f, gradU);
}

Eigen::MatrixXd computeB(const DikinFactors& f, const Eigen::VectorXd& w) {
  const int nc = f.Q.rows();
  const int d = f.Q.cols();
  Eigen::MatrixXd That(nc, d);
  for (int r = 0; r < nc; r++) {
    double suffix = 0.0;  // sum_{k>j} q_rk w_k, built incrementally (j descending)
    for (int j = d - 1; j >= 0; j--) {
      const double qrj = f.Q(r, j);
      That(r, j) = qrj * (qrj * w(j) + 2.0 * suffix);
      suffix += qrj * w(j);
    }
  }
  return That.transpose() * f.Q;
}

Eigen::MatrixXd computeJg(const DikinFactors& f, const Eigen::MatrixXd& B) {
  const int d = f.R.cols();
  const Eigen::MatrixXd C = f.R.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(d, d));
  return -C * B * f.R;  // -C * B * C^{-1}, C^{-1} = R
}

bool certificateOk(int d, double h, const Eigen::VectorXd& w) {
  const double theta = (std::abs(h) / 2.0) * std::sqrt(2.0 * d) * w.norm();
  return theta < 1.0 && std::isfinite(theta);
}

MidpointResult solveMidpoint(const Polyhedron& polyhedron, const Eigen::VectorXd& x,
                              const Eigen::VectorXd& w, double h, double tau_mid, int n_mid) {
  MidpointResult out;
  const Eigen::MatrixXd& A = polyhedron.A();
  const Eigen::VectorXd& b = polyhedron.b();
  const double beta = h / 2.0;

  DikinFactors f0 = computeFactors(A, b, x);
  Eigen::VectorXd m = x + beta * applyC(f0, w);
  if (!polyhedron.contains(m) || !m.allFinite()) return out;

  DikinFactors fm0 = computeFactors(A, b, m);
  Eigen::MatrixXd Bm0 = computeB(fm0, w);
  Eigen::MatrixXd Jg0 = computeJg(fm0, Bm0);
  Eigen::MatrixXd M0 = Eigen::MatrixXd::Identity(x.size(), x.size()) - beta * Jg0;
  Eigen::PartialPivLU<Eigen::MatrixXd> lu(M0);

  DikinFactors f_cur = fm0;
  double eta = 0.0;
  for (int k = 0; k < n_mid; k++) {
    f_cur = computeFactors(A, b, m);
    const Eigen::VectorXd residual = m - x - beta * applyC(f_cur, w);
    eta = residual.norm() / (1.0 + (m - x).norm());
    if (!std::isfinite(eta)) return out;
    if (eta <= tau_mid) {
      out.m = m;
      out.factors = f_cur;
      out.success = true;
      return out;
    }
    m = m - lu.solve(residual);
    if (!polyhedron.contains(m) || !m.allFinite()) return out;
  }
  // One final residual check after the last permitted correction.
  f_cur = computeFactors(A, b, m);
  const Eigen::VectorXd residual = m - x - beta * applyC(f_cur, w);
  eta = residual.norm() / (1.0 + (m - x).norm());
  if (std::isfinite(eta) && eta <= tau_mid) {
    out.m = m;
    out.factors = f_cur;
    out.success = true;
  }
  return out;
}

StepResult dstep(const Objective& objective, const Polyhedron& polyhedron,
                  const Eigen::VectorXd& x, const Eigen::VectorXd& v, double h,
                  double tau_mid, int n_mid) {
  StepResult out;
  const int d = x.size();
  const Eigen::MatrixXd& A = polyhedron.A();
  const Eigen::VectorXd& b = polyhedron.b();

  const DikinFactors fx = computeFactors(A, b, x);
  const Eigen::VectorXd w = v + (h / 2.0) * force(fx, objective.gradient(x));
  if (!certificateOk(d, h, w)) return out;

  MidpointResult mid = solveMidpoint(polyhedron, x, w, h, tau_mid, n_mid);
  if (!mid.success) return out;

  const Eigen::VectorXd x_next = 2.0 * mid.m - x;
  if (!polyhedron.contains(x_next) || !x_next.allFinite()) return out;

  const DikinFactors fxp = computeFactors(A, b, x_next);
  const Eigen::VectorXd v_next = w + (h / 2.0) * force(fxp, objective.gradient(x_next));

  const Eigen::MatrixXd Bm = computeB(mid.factors, w);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(d, d);
  const double det_minus = (I - (h / 2.0) * Bm).determinant();
  const double det_plus = (I + (h / 2.0) * Bm).determinant();
  if (!(det_minus > 0.0) || !(det_plus > 0.0) || !std::isfinite(det_minus) || !std::isfinite(det_plus)) {
    return out;
  }

  out.x_next = x_next;
  out.v_next = v_next;
  out.log_jacobian = std::log(det_minus) - std::log(det_plus);
  out.success = true;
  return out;
}

}  // namespace kinetic_dikin

}  // namespace cldiff
