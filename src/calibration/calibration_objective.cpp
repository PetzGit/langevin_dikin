#include "cldiff/calibration/calibration_objective.hpp"
#include <algorithm>
#include <cmath>

CalibrationObjective::CalibrationObjective(const Pricer& pricer, const MarketData& market_data,
                                           const QuoteLoss& loss, const ParameterPenalty* penalty)
: pricer_(pricer), market_data_(market_data), loss_(loss), penalty_(penalty) {}

double CalibrationObjective::value(const Eigen::VectorXd& params) const {
  const Eigen::VectorXd prices = pricer_.price(params, market_data_);
  double loss = 0.0;
  Eigen::Index i = 0;
  for (const auto& [T, quotes] : market_data_.quotesByMaturity()) {
    for (const Quote& quote : quotes) {
      loss += loss_(prices(i++), T, quote, market_data_);
    }
  }
  return penalty_ ? loss + (*penalty_)(params) : loss;
}

Eigen::VectorXd CalibrationObjective::gradient(const Eigen::VectorXd& params) const {
  const int d = dim();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(d);
  double h_base = 2e-3;
  for (int i{0}; i < d; i++) {
    double h_i = h_base * std::max(1.0, std::abs(params(i)));
    Eigen::VectorXd plus_params = params;
    plus_params(i) += h_i;
    Eigen::VectorXd minus_params = params;
    minus_params(i) -= h_i;
    grad(i) = (value(plus_params) - value(minus_params))/(2*h_i);
  }
  return grad;
}

int CalibrationObjective::dim() const {
  return pricer_.numParameters();
}
