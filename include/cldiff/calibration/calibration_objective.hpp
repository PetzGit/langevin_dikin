#pragma once
#include <Eigen/Dense>
#include "cldiff/optim/objective.hpp"
#include "cldiff/pricing/pricer.hpp"
#include "cldiff/market/market_data.hpp"
#include "cldiff/calibration/quote_loss.hpp"
#include "cldiff/calibration/parameter_penalty.hpp"

namespace cldiff {

// The bridge from pricing to optim: sum over quotes of loss(model price),
// plus an optional parameter penalty, as an Objective over the pricer's
// parameters. Any Pricer x QuoteLoss x ParameterPenalty combination plugs in.
// All arguments are held by reference and must outlive this object.
class CalibrationObjective : public Objective {
  public:
    CalibrationObjective(const Pricer& pricer, const MarketData& market_data, const QuoteLoss& loss,
                         const ParameterPenalty* penalty = nullptr);
    double value(const Eigen::VectorXd& params) const override;
    // Central finite differences, step 2e-3 * max(1, |p_i|).
    //todo implement autodiff (this requires changing type signatures everywhere current imp
    //is finite differences, which is numerically unstable somewhat and costly
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const override;
    int dim() const override;
  private:
    const Pricer& pricer_;
    const MarketData& market_data_;
    const QuoteLoss& loss_;
    const ParameterPenalty* penalty_;
};

}  // namespace cldiff
