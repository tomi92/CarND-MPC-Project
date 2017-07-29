#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC
{
private:
  vector<double> x_values_;
  vector<double> y_values_;

public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  const vector<double> &getXValues() { return x_values_; }
  const vector<double> &getYValues() { return y_values_; }
};

#endif /* MPC_H */
