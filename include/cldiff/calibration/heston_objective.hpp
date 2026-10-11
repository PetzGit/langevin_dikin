#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"
#include "cldiff/market/market_data.hpp"
#include "cldiff/models/heston/heston_cos_pricer.hpp"
#include "cldiff/calibration/calibration_objective.hpp"
#include "cldiff/calibration/implied_vol_squared_loss.hpp"
#include "cldiff/calibration/feller_penalty.hpp"

namespace cldiff {

// The standard Heston calibration setup, pre-assembled: COS prices, squared
// implied-vol error per quote, smoothed Feller penalty (see FellerPenalty for
// feller_smooth_scale). Equivalent to building a CalibrationObjective from
// those three parts by hand.
class HestonObjective : public Objective {
  public:
    HestonObjective(const MarketData& market_data, int term_num, double L, double lambda_feller,
                     double feller_smooth_scale = 0.01);
    double value(const Eigen::VectorXd& params) const override;
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const override;
    int dim() const override;
    // objective_ refers to the members below, so a copy would dangle.
    HestonObjective(const HestonObjective&) = delete;
    HestonObjective& operator=(const HestonObjective&) = delete;
  private:
    HestonCosPricer pricer_;
    ImpliedVolSquaredLoss loss_;
    FellerPenalty feller_;
    CalibrationObjective objective_;
};

}  // namespace cldiff
