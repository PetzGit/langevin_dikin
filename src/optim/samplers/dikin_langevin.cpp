#include "cldiff/optim/samplers/dikin_langevin.hpp"
#include <stdexcept>
#include <Eigen/Cholesky>
#include <cmath>

struct DikinLangevinSampler::Run {
    const Objective& objective;
    const Polyhedron& polyhedron;
    Rng& rng;
    Eigen::VectorXd x;
};

DikinLangevinSampler::DikinLangevinSampler(double eps, double h_max)
    : eps_(eps), h_max_(h_max)
{
    if (h_max_ < 0) {
        throw std::invalid_argument("Invalid arguments: h_max must be non-negative");
    }
}

Eigen::MatrixXd DikinLangevinSampler::sample(const Problem& problem, const Eigen::VectorXd& x0, Rng& rng, int n) {
    if ((x0.size() != problem.domain.size()) || (x0.size() != problem.objective.dim()) || !problem.domain.contains(x0)) {
        throw std::invalid_argument("Invalid arguments: please check the dimensions and interiority of x0");
    }
    Run run{problem.objective, problem.domain, rng, x0};
    Eigen::MatrixXd trajectory(x0.size(),n);
    for(int i{0}; i < n; i++) {
      trajectory.col(i) = run.x;
      step(run);
    }
    return trajectory;
}

void DikinLangevinSampler::step(Run& run) const {
    const Eigen::MatrixXd H_eps = run.polyhedron.barrierHessian(run.x, eps_);
    const Eigen::LLT<Eigen::MatrixXd> llt(H_eps);
    const Eigen::MatrixXd L = llt.matrixL();
    const Eigen::VectorXd m_x = proposeState(run, run.x, llt,1);
    const double h = run.rng.uniform() * h_max_;
    const Proposal proposal1 = propose(run, run.x, m_x, h, L,1);
    const Eigen::VectorXd y = proposal1.Y;
    if (!run.polyhedron.contains(y)) {
        return;
    }
    const double log_diff = run.objective.value(run.x) - run.objective.value(y);
    const double quad1 = 2.0 * h * proposal1.x_i.squaredNorm();
    const double log_det1 = -2.0 * L.diagonal().array().log().sum();
    const Eigen::MatrixXd H_y_eps = run.polyhedron.barrierHessian(y, eps_);
    const Eigen::LLT<Eigen::MatrixXd> llt_y(H_y_eps);
    const Eigen::MatrixXd L_y = llt_y.matrixL();
    const Eigen::VectorXd m_y = proposeState(run, y, llt_y, 1);
    const Eigen::VectorXd diff = run.x - (y + h * m_y);
    const double quad2 = diff.dot(H_y_eps * diff);
    const double log_det2 = -2.0 * L_y.diagonal().array().log().sum();
    const double acceptance = log_diff + (0.25/h) * quad1 + (0.5)*log_det1 - (0.25/h)* quad2 - (0.5)*log_det2;
    if (std::min(0.0, acceptance) > std::log(run.rng.uniform())) {
        run.x = y;
        return;
    }
}
Eigen::VectorXd DikinLangevinSampler::proposeState(const Run& run, const Eigen::VectorXd& x, const Eigen::LLT<Eigen::MatrixXd>& llt, const double beta) const {
    const Eigen::VectorXd div_c_eps = run.polyhedron.barrierHessianDivergence(llt, x);
    const Eigen::VectorXd grad_term = -1.0 * llt.solve(run.objective.gradient(x))/ beta;
    return grad_term + div_c_eps;
}
Proposal DikinLangevinSampler::propose(Run& run, const Eigen::VectorXd& x, const Eigen::VectorXd& m_x, const double h, const Eigen::MatrixXd& L, const double beta) const {
    const int d = m_x.size();
    Eigen::VectorXd x_i(d);
    for (int i{0}; i < d; i++) {
        x_i(i) = run.rng.normal();
    }
    const Eigen::VectorXd eta = L.triangularView<Eigen::Lower>().transpose().solve(x_i);
    return Proposal{x + h * m_x + std::sqrt(2.0 * h / beta) * eta, x_i};
}
