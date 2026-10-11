#pragma once
#include <Eigen/Dense>
#include "cldiff/core/rng.hpp"

class SDEScheme {
  public:
    virtual ~SDEScheme() = default;
    virtual void step(Eigen::VectorXd& state, double dt, Rng& rng) const = 0;
    virtual int dimension() const = 0;
    virtual int numParameters() const = 0;
    virtual void setParameters(const Eigen::VectorXd& paras) = 0;
    virtual Eigen::VectorXd parameters() const = 0;    
};

