#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

int N = 25;
double dt = 0.10;

const double Lf = 2.67;

const double ref_v = 50 * 0.44704;

int x_start = 0;
int y_start = x_start + N;
int psi_start = y_start + N;
int v_start = psi_start + N;
int cte_start = v_start + N;
int epsi_start = cte_start + N;
int delta_start = epsi_start + N;
int a_start = delta_start + N - 1;

class FG_eval
{
public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector &fg, const ADvector &vars)
  {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    for (int t = 0; t < N; t++)
    {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += 1000 * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    for (int t = 0; t < N - 1; t++)
    {
      fg[0] += 5000 * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }

    for (int t = 0; t < N - 2; t++)
    {
      fg[0] += 1000 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    //
    // Setup Constraints
    //
    // NOTE: In this section you'll setup the model constraints.

    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 0; t < N - 1; t++)
    {
      AD<double> x0 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t];
      AD<double> v0 = vars[v_start + t];
      AD<double> cte0 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t];

      AD<double> delta0 = vars[delta_start + t];
      AD<double> a0 = vars[a_start + t];

      AD<double> fx0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3] * x0 * x0 * x0;
      AD<double> psiDesired0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * x0 * x0);

      AD<double> x1 = vars[x_start + t + 1];
      AD<double> y1 = vars[y_start + t + 1];
      AD<double> psi1 = vars[psi_start + t + 1];
      AD<double> v1 = vars[v_start + t + 1];
      AD<double> cte1 = vars[cte_start + t + 1];
      AD<double> epsi1 = vars[epsi_start + t + 1];

      fg[1 + x_start + t + 1] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t + 1] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t + 1] = psi1 - (psi0 + v0 / Lf * delta0 * dt);
      fg[1 + v_start + t + 1] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t + 1] = cte1 - (fx0 - y0 + v0 * CppAD::sin(epsi0) * dt);
      fg[1 + epsi_start + t + 1] = epsi1 - (psi0 - psiDesired0 + v0 / Lf * delta0 * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs)
{
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  double delta = state[6];
  double a = state[7];

  double latency = 0.1;
  x += v * cos(psi) * latency;
  y += v * sin(psi) * latency;
  psi += v * delta / Lf * latency;
  v += a * latency;
  double fx = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x;
  cte = y - fx;
  double psi_target = atan(coeffs[1] + 2 * coeffs[2] * x + 3 * coeffs[3] * x * x);
  epsi = psi - psi_target;

  int n_vars = N * 6 + (N - 1) * 2;
  int n_constraints = N * 6;

  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++)
  {
    vars[i] = 0;
  }

  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  for (int i = 0; i < delta_start; i++)
  {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  for (int i = delta_start; i < a_start; i++)
  {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  for (int i = a_start; i < n_vars; i++)
  {
    vars_lowerbound[i] = -0.6; // Don't accelerate too much
    vars_upperbound[i] = 0.6;
  }

  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);

  for (int i = 0; i < n_constraints; i++)
  {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  x_values_ = {};
  y_values_ = {};

  if (!ok)
  {
    std::cout << "There was an error." << std::endl;
    return {0, 0};
  }

  for (int i = 0; i < N; i++)
  {
    x_values_.push_back(solution.x[x_start + i]);
    y_values_.push_back(solution.x[y_start + i]);
  }

  return {solution.x[delta_start], solution.x[a_start]};
}
