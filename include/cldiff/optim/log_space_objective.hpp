#pragma once
#include <vector>
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"

// Reparameterises an objective so the dims flagged in log_dims are optimised
// as y = log(p) (keeping p > 0); the remaining dims pass through unchanged.
//
// The gradient is the "corrected" one: central finite differences in y, with
// the chain-rule factor dL/dy = dL/dp * p divided back out for each log dim.
// Without that division the log-space gradient shrinks in proportion to the
// parameter's own value, so a parameter heading towards zero stops
// generating signal even when the physical sensitivity is still large. See
// CorrectedLogSpaceHestonObjective for the failure this fixed in practice.
class LogSpaceObjective : public Objective {
  public:
    LogSpaceObjective(const Objective& inner, std::vector<bool> log_dims, double fd_step = 3e-3);
    double value(const Eigen::VectorXd& y) const override;
    Eigen::VectorXd gradient(const Eigen::VectorXd& y) const override;
    int dim() const override;
    Eigen::VectorXd toPhysical(const Eigen::VectorXd& y) const;
    Eigen::VectorXd toLogSpace(const Eigen::VectorXd& p) const;

    static Eigen::VectorXd toPhysical(const Eigen::VectorXd& y, const std::vector<bool>& log_dims);
    static Eigen::VectorXd toLogSpace(const Eigen::VectorXd& p, const std::vector<bool>& log_dims);

  private:
    const Objective& inner_;
    std::vector<bool> log_dims_;
    double fd_step_;
};
