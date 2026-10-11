#pragma once
#include "cldiff/calibration/quote_loss.hpp"

// weight * (model iv - market iv)^2. A quote whose market or model implied
// vol is not finite contributes its full weight instead.
class ImpliedVolSquaredLoss final : public QuoteLoss {
public:
  double operator()(double model_price, double T, const Quote& quote, const MarketData& market) const override;
};
