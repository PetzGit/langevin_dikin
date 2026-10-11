#include "cldiff/calibration/implied_vol_squared_loss.hpp"
#include "cldiff/models/black_scholes/black_scholes.hpp"
#include <cmath>

namespace cldiff {

double ImpliedVolSquaredLoss::operator()(double model_price, double T, const Quote& quote,
                                         const MarketData& market) const {
  BlackSchole iv_calc(market.S0(), quote.K, T, market.r(), market.q());
  double calc_iv = iv_calc.impliedVol(model_price);
  if (!std::isfinite(quote.market_iv) || !std::isfinite(calc_iv)) {
    return quote.weight;
  }
  return (calc_iv - quote.market_iv)*(calc_iv - quote.market_iv)*quote.weight;
}

}  // namespace cldiff
