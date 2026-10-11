#include "cldiff/core/estimator.hpp"
#include <cmath>
#include <limits>
Estimator::Estimator() : mean_(0), M2_(0), count_(0) {}

void Estimator::add(double x) {
 count_ += 1;
 double old_mean = mean_;
 mean_ += (x - mean_) / (count_);
 M2_ += (x - old_mean) * (x - mean_);
}

double Estimator::getMean() const {
  return mean_;
}

double Estimator::getVariance() const {
  return M2_/count_;
}

double Estimator::getSampleVariance() const {
  if (count_ < 2) {return std::numeric_limits<double>::quiet_NaN();}
  return M2_/(count_ -1);
}

double Estimator::getStdError() const {
 return (std::sqrt(getSampleVariance()/(count_)));
}
