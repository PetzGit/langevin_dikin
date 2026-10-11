#pragma once
#include "cldiff/optim/objective.hpp"
#include "cldiff/optim/polyhedron.hpp"

namespace cldiff {

// What is being minimised / sampled, kept separate from how. Optimisers and
// samplers receive this per call rather than at construction, so one
// configured algorithm can be run against many problems and one problem
// against many algorithms.
struct Problem {
  const Objective& objective;
  const Polyhedron& domain;
};

}  // namespace cldiff
