#pragma once
#include <Eigen/Dense>
#include "cldiff/market/market_data.hpp"

namespace cldiff {

// Model + method, priced against a set of quotes: the seam between pricing
// and calibration. CalibrationObjective only ever sees this interface, so
// COS, Monte Carlo and PDE pricers of any model are interchangeable to it.
//
// Parameters are passed per call rather than held, so a pricer can be shared
// by const reference and price() stays free of hidden state. Monte Carlo
// implementations should fix their seed across calls (common random numbers),
// otherwise the calibration objective is noisy in the parameters.
class Pricer {
public:
  virtual ~Pricer() = default;
  virtual int numParameters() const = 0;
  // One model price per quote, in MarketData::quotesByMaturity() order.
  virtual Eigen::VectorXd price(const Eigen::VectorXd& params, const MarketData& market) const = 0;
};

}  // namespace cldiff
