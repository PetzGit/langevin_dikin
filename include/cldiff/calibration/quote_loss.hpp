#pragma once
#include "cldiff/market/market_data.hpp"

// Contribution of one quote to a calibration loss, given the model's price
// for it. T is the quote's maturity (the key it is stored under in MarketData).
class QuoteLoss {
public:
  virtual ~QuoteLoss() = default;
  virtual double operator()(double model_price, double T, const Quote& quote, const MarketData& market) const = 0;
};
