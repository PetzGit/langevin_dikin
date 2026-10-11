#pragma once
#include <Eigen/Dense>
#include "cldiff/products/observable.hpp"

class EuropeanCall final : public Observable {
  public:
    EuropeanCall(double K, double r, double T);
    int numOutputs() const override { return 1; }
    Eigen::VectorXd result(const Eigen::VectorXd& final_state) const override;

  private:
    double K_;
    double discount_;
};
