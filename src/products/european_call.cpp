#include "cldiff/products/european_call.hpp"
#include <algorithm>
#include <cmath>

EuropeanCall::EuropeanCall(double K, double r, double T)
    : K_(K), discount_(std::exp(-r * T)) {}

Eigen::VectorXd EuropeanCall::result(const Eigen::VectorXd& final_state) const {
    Eigen::VectorXd out(1);
    // state(0) is ln S, so exponentiate before comparing with K
    out(0) = discount_ * std::max(std::exp(final_state(0)) - K_, 0.0);
    return out;
}
