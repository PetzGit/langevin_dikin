#pragma once
#include <Eigen/Dense>

class Observable {
  public:
    virtual ~Observable() = default;
    virtual int numOutputs() const = 0;
    virtual bool needsSteps() const { return false; }
    virtual void onStart(const Eigen::VectorXd& /*x0*/) {}
    virtual void onStep(int /*i*/, double /*t*/, const Eigen::VectorXd& /*state*/) {}
    virtual Eigen::VectorXd result(const Eigen::VectorXd& final_state) const = 0;
};
