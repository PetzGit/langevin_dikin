#pragma once
#include "cldiff/pricing/pricer.hpp"

namespace cldiff {

// Heston European calls via the COS method. params = (kappa, theta, xi, rho, v0).
// Cumulants and the truncation range are computed once per maturity.
class HestonCosPricer final : public Pricer {
public:
  HestonCosPricer(int term_num, double L);
  int numParameters() const override { return 5; }
  Eigen::VectorXd price(const Eigen::VectorXd& params, const MarketData& market) const override;
private:
  int term_num_;
  double L_;
};

}  // namespace cldiff
