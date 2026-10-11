#pragma once
#include <vector>

namespace cldiff {

class Grid {
  public:
    std::vector<double> grid();

  private:
  int N_;
  double K_;
  double alpha_;
  std::vector<double> grid_;
  S_max_;
}

}  // namespace cldiff
