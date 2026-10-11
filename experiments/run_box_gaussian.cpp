#include "cldiff/optim/samplers/dikin_langevin.hpp"
#include "cldiff/optim/polyhedron.hpp"
#include "cldiff/core/rng.hpp"
#include "cldiff/optim/objective.hpp"
#include <Eigen/Dense>
#include <iostream>
class GaussianObjective : public Objective {
public:
    explicit GaussianObjective(int d) : d_(d) {}

    double value(const Eigen::VectorXd& x) const override {
        return 0.5 * x.squaredNorm();
    }
    Eigen::VectorXd gradient(const Eigen::VectorXd& x) const override {
        return x;
    }
    int dim() const override { return d_; }

private:
    int d_;
};



int main() {
 const Eigen::VectorXd b = Eigen::VectorXd::Ones(6);
 Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6,3);
 for(int i{0}; i < 3; i++) {
   A(2*i,i) = 1;
   A(2*i+1,i) = -1;
 }
 Polyhedron poly(A,b);
 Rng rng(2);
 GaussianObjective gaussian(3);
 double eps = 1e-5;
 double h_max = 0.1;
 Eigen::VectorXd x_0 = Eigen::VectorXd::Zero(3);
 DikinLangevinSampler sampler(eps, h_max);
 Eigen::MatrixXd output = sampler.sample({gaussian, poly}, x_0, rng, 100000);
 //Eigen::VectorXd mean_position = output.rowwise().mean();
 std::cout << output << std::endl;
 return 0;
}
