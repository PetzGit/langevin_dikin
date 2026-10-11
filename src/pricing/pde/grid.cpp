#include "cldiff/pricing/pde/grid.hpp"
#include <cmath>

Grid::Grid(int N,double K,double alpha,double S_max) : N_(N), K_(K), alpha_(alpha), S_max_(S_max) {
  const double lower = std::asinh(-1.0*K_/alpha_);
  const double upper = std::asinh((S_max_-K_)/alpha_);
  const int N_1 = std::lround(N* (-lower)/(upper-lower));
  const double inc_1 = -lower/N_1;
  const double inc_2 = upper/(N-N_1);
  double curr_1 = lower;
  int i{0};
  for (; i < N_1; i++){
    grid_.push_back(std::sinh(curr_1)*alpha_ + K_);
    curr_1 += inc_1;
  }
  double curr_2 = curr_1 + inc_2;
  for (i < N; i++) {
    grid_.push_back(std::sinh(curr_2) * alpha_ +K_);
    curr_2 += inc_2;
  }
}

const std::vector<double>& Grid::grid() const {
  return grid_;
}

