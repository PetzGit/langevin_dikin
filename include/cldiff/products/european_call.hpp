#pragma once
#include <Eigen/Dense>
#include "cldiff/products/observable.hpp"
#include "cldiff/market/market_inputs.hpp"

namespace cldiff {

class EuropeanCall final : public Observable {
  public:
    explicit EuropeanCall(double K, MarketInputs market, double T);
    int numOutputs() const override { return 1; }
    Eigen::VectorXd result(const Eigen::VectorXd& final_state) const override;

  private:
    double K_;
    double discount_;
};

}  // namespace cldiff
