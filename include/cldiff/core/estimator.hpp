#pragma once
#include <cstddef>

namespace cldiff {

class Estimator {
  public:
    Estimator();
    void add(double x);
    double getMean() const;
    double getVariance() const;
    double getSampleVariance() const;
    double getStdError() const;
    std::size_t getCount() const;
  private:
    double mean_;
    double M2_;
    std::size_t count_;
};

}  // namespace cldiff
