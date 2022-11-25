# GSA NLP solver

An implementation of the algorithm GSA to solve constrained nonlinear programming problems with Lipschitzian functions. GSA was introduced by prof. R.G. Strongin (see R. G. Strongin, D. L. Markin, Minimization of multiextremal functions under nonconvex constraints, Cybernetics 22(4), 486-493. Translated from Russian. Consultant Bureau. New York, 1986. [[link]][paper]). The method exploits Peano-type curve to reduce dimension of the source bounded multidimensional constrained NLP problem and then solves a univariate one.

GSA is proven to converge to a global optima if all objectives and constraints satisfy Lipschitz condition in a given hyperrectangle, the reliability parameter `r` is large enough and accuracy parameter `eps` is zero.

This implementation of GSA is included into [NLOpt](https://github.com/stevengj/nlopt) library.

## Clone & build, run samples
- on Linux:
```bash
git clone --recursive https://github.com/UNN-ITMM-Software/gsa_nlp_solver.git
cd gsa_nlp_solver
mkdir build
cd build
cmake ..
make -j 4
./bin/solve_constrained
./bin/solve_set
```
- on Windows:
```batch
git clone --recursive https://github.com/UNN-ITMM-Software/gsa_nlp_solver.git
cd gsa_nlp_solver
mkdir build
cd build
cmake ..
cmake --build . --config RELEASE
./bin/Release/solve_constrained.exe
./bin/Release/solve_set.exe
```
[paper]: https://www.tandfonline.com/doi/abs/10.1080/17442508908833568?journalCode=gssr19

## Python bindings

GSA is also available from Python. To build the bindings add the following commands to cmake call:
```bash
 cmake .. -DBUILD_BINDINGS=ON -DPYBIND11_PYTHON_VERSION=<required python version>
```
If `PYBIND11_PYTHON_VERSION` is not specified, bindings would be built for the latest found Python version.
Running python example (after calling `make` or `cmake --build`) from **build folder**
- on Linux:
```bash
cd ..
cp -r 3rd-party/global-optimization-test-problems/benchmark_tools/ build/bin/
export PYTHONPATH=build/bin
python samples/python/solve_constrained.py
```
- on Windows (bash):
```bash
cd ..
cp -r 3rd-party/global-optimization-test-problems/benchmark_tools/ build/bin/Release/
export PYTHONPATH=build/bin/Release
python samples/python/solve_constrained.py
```

## Example of usage (C++)
```C++
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>

#include "solver.hpp"

using namespace ags;

int main(int argc, char** argv)
{
  auto parameters = SolverParameters();
  parameters.refineSolution = true; // refine solution with a local optimizer
  parameters.epsR = 0.1

  NLPSolver solver;
  solver.SetParameters(parameters);
  //First 3 functions -- nonlinear inequality constraints g_i(y)<=0
  //Last function -- objective
  //Last 2 arguments -- bounds of the search hyperrectangle
  solver.SetProblem({
    [](const double* x) {return 0.01*(pow(x[0] - 2.2, 2) + pow(x[1] - 1.2, 2) - 2.25);},
    [](const double* x) {return 100 * (1 - pow(x[0] - 2, 2) / 1.44 - pow(0.5*x[1], 2));},
    [](const double* x) {return 10 * (x[1] - 1.5 - 1.5*sin(2*M_PI*(x[0] - 1.75)));},
    [](const double* x) {return -1.5*pow(x[0], 2) * exp(1 - pow(x[0], 2)
        - 20.25*pow(x[0] - x[1], 2)) - pow(0.5 * (x[1] - 1)*(x[0]- 1), 4)
        * exp(2 - pow(0.5 * (x[0] - 1), 4) - pow(x[1] - 1, 4));}
  }, {0, -1}, {4, 3});

  auto optimalPoint = solver.Solve();
  auto calcCounters = solver.GetCalculationsStatistics();
  auto holderConstEstimations = solver.GetHolderConstantsEstimations();

  for (size_t i = 0; i < calcCounters.size() - 1; i++)
    std::cout << "Number of calculations of constraint # " << i << ": " << calcCounters[i] << "\n";
  std::cout << "Number of calculations of objective: " << calcCounters.back() << "\n";

  for (size_t i = 0; i < holderConstEstimations.size() - 1; i++)
    std::cout << "Estimation of Holder constant of function # " << i << ": " << holderConstEstimations[i] << "\n";
  std::cout << "Estimation of Holder constant of objective: " << holderConstEstimations.back() << "\n";


  //Optimal point has it's index -- number of the first broken constraint
  //If index equals to the number of constraints, then the point if feasible and
  //objective was evaluated at this point. If the solver returned unfeasible
  //optimal point, the set of feasible points is most likely to be empty.
  if (optimalPoint.idx < 3)
    std::cout << "Feasible point not found" << "\n";
  else
  {
    std::cout << "Optimal value: " << optimalPoint.g[optimalPoint.idx] << "\n";
    std::cout << "x = " << optimalPoint.y[0] << " y = " << optimalPoint.y[1] << "\n";
  }
  return 0;
}
```

Visualization of the obtained solution:
![contours](samples/pics/contours.png)
