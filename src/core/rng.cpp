#include "cldiff/core/rng.hpp"

namespace cldiff {

double Rng::uniform() {
  return uniform_(gen_);
}
double Rng::normal() {
  return normal_(gen_);
}

}  // namespace cldiff
