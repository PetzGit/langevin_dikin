#pragma once
#include "cldiff/pricing/monte_carlo/sde_scheme.hpp"
#include "Eigen/Dense"

namespace cldiff {

struct OuParams {
  double kappa;
  double mu;
  double sigma;
};

class OuScheme final: public SDEScheme {
  public:
    explicit OuScheme(const Eigen::VectorXd& p);
    void step(Eigen::VectorXd& state, double dt, Rng& rng) const override;
    int dimension() const override;
    int numParameters() const override;
    Eigen::VectorXd parameters() const override;
    void setParameters(const Eigen::VectorXd& p) override;

  private:
    OuParams params_;
};

}
