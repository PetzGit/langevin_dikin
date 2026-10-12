#pragma once
#include "cldiff/pricing/monte_carlo/sde_scheme.hpp"
#include "Eigen/Dense"
#include "cldiff/market/market_inputs.hpp"

namespace cldiff {

struct GbmParams {
  double sigma;  
};

class GbmScheme final: public SDEScheme {
  public:
    explicit GbmScheme(const Eigen::VectorXd& p, MarketInputs market);
    void step(Eigen::VectorXd& state, double dt, Rng& rng) const override;
    int dimension() const override;
    int numParameters() const override;
    Eigen::VectorXd parameters() const override;
    void setParameters(const Eigen::VectorXd& p) override;

  private:
    GbmParams params_;
    MarketInputs market_;
};

}
