#pragma once
#include "cldiff/pricing/monte_carlo/sde_scheme.hpp"
#include <Eigen/Dense>
#include "cldiff/market/market_inputs.hpp"

namespace cldiff {

struct HestonParams {
  double kappa{};
  double theta{};
  double xi{};
  double rho{};
};
class HestonScheme final : public SDEScheme {
  public:
    // p = (kappa, theta, xi, rho); r and q are market inputs.
    // State layout: state(0) = ln S, state(1) = v.
    explicit HestonScheme(const Eigen::VectorXd& p, MarketInputs market);
    void step(Eigen::VectorXd& state, double dt, Rng& rng) const override;
    int dimension() const override;
    int numParameters() const override;
    void setParameters(const Eigen::VectorXd& p) override;
    Eigen::VectorXd parameters() const override;

  private:
    HestonParams params_;
    MarketInputs market_;
    double rho_bar_;
};

}  // namespace cldiff
