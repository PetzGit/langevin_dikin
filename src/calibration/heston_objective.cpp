#include "cldiff/calibration/heston_objective.hpp"

HestonObjective::HestonObjective(const MarketData& market_data, int term_num, double L, double lambda_feller,
                                  double feller_smooth_scale)
: pricer_(term_num, L), loss_(), feller_(lambda_feller, feller_smooth_scale),
  objective_(pricer_, market_data, loss_, &feller_) {}

double HestonObjective::value(const Eigen::VectorXd& params) const {
  return objective_.value(params);
}

Eigen::VectorXd HestonObjective::gradient(const Eigen::VectorXd& params) const {
  return objective_.gradient(params);
}

int HestonObjective::dim() const {
  return objective_.dim();
}
