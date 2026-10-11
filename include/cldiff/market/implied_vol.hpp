#pragma once

namespace cldiff {

// Black-Scholes implied volatility for a call with fixed (S0, K, T, r, q):
// impliedVol inverts price(sigma) by Newton iteration using vega(sigma).
// Calibration compares models in implied-vol space rather than price space,
// so this is the quoting convention, not a pricing model.
class BlackScholesImpliedVol {
  public:
    BlackScholesImpliedVol(double S0, double K, double T, double r, double q = 0.0);
    double price(double sigma) const;
    double vega(double sigma) const;
    double impliedVol(double C_mkt) const;

  private:
    double S0_;
    double K_;
    double T_;
    double r_;
    double q_;
};

}  // namespace cldiff
