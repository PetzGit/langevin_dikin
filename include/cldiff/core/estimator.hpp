#pragma once
#include <cstddef>

class Estimator {
  public:
    Estimator();
    void add(double x);
    double getMean() const;
    double getVariance() const;
    double getSampleVariance() const;
    double getStdError() const;
  private:
    double mean_;
    double M2_;
    std::size_t count_;
};
