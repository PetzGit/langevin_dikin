#pragma once
#include <Eigen/Dense>

namespace cldiff {

// Soft constraint on the model parameters, added to a calibration loss.
class ParameterPenalty {
public:
  virtual ~ParameterPenalty() = default;
  virtual double operator()(const Eigen::VectorXd& params) const = 0;
};

}  // namespace cldiff
