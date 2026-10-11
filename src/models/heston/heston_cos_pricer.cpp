#include "cldiff/models/heston/heston_cos_pricer.hpp"
#include "cldiff/models/heston/characteristic_function.hpp"
#include "cldiff/models/heston/cumulants.hpp"
#include "cldiff/pricing/fourier/cos_pricer.hpp"

HestonCosPricer::HestonCosPricer(int term_num, double L) : term_num_(term_num), L_(L) {}

Eigen::VectorXd HestonCosPricer::price(const Eigen::VectorXd& params, const MarketData& market) const {
  CharacteristicHeston phi(params);
  std::vector<double> prices;
  for (const auto& [T, quotes] : market.quotesByMaturity()) {
    Cumulants cum = cumulants(phi, T);
    Range range = make_range(cum, L_);
    for (const Quote& quote : quotes) {
      prices.push_back(cos_pricer(phi, range, quote.K, T, market.r(), term_num_, market.S0(), market.q()));
    }
  }
  return Eigen::Map<Eigen::VectorXd>(prices.data(), static_cast<Eigen::Index>(prices.size()));
}
