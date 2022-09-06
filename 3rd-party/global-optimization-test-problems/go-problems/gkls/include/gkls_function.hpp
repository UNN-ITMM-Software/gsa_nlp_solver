/******************************************************************************/
/*        GKLS-Generator of Classes of ND  (non-differentiable),              */
/*                                 D  (continuously differentiable), and      */
/*                                 D2 (twice continuously differentiable)     */
/*                     Test Functions for Global Optimization                 */
/*                                                                            */
/*   Authors:                                                                 */
/*                                                                            */
/*   M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev                         */
/*                                                                            */
/*   (C) 2002-2005                                                            */
/*                                                                            */
/*	 References:                                                              */
/*                                                                            */
/*   1. M.Gaviano, D.E.Kvasov, D.Lera, and Ya.D.Sergeyev (2003),              */
/*   Algorithm 829: Software for Generation of Classes of Test Functions      */
/*   with Known Local and Global Minima for Global Optimization.              */
/*   ACM Transactions on Mathematical Software, Vol. 29, no. 4, pp. 469-480.  */
/*                                                                            */
/*   2. D.Knuth (1997), The Art of Computer Programming, Vol. 2:              */
/*   Seminumerical Algorithms (Third Edition). Reading, Massachusetts:        */
/*   Addison-Wesley.                                                          */
/*                                                                            */
/*   The software constructs a convex quadratic function (paraboloid) and then*/
/*   systematically distorts randomly selected parts of this function         */
/*   by polynomials in order to introduce local minima and to construct test  */
/*   functions which are non-differentiable (ND-type), continuously           */
/*   differentiable (D-type), and twice continuously differentiable (D2-type) */
/*   at the feasible region.                                                  */
/*                                                                            */
/*   Each test class is defined by the following parameters:                  */
/*  (1) problem dimension                                                     */
/*  (2) number of local minima including the paraboloid min and the global min*/
/*  (3) global minimum value                                                  */
/*  (3) distance from the paraboloid vertex to the global minimizer           */
/*  (4) radius of the attraction region of the global minimizer               */
/******************************************************************************/

#ifndef __GKLS_FUNCTION_HPP__
#define __GKLS_FUNCTION_HPP__

#include "gkls_random.hpp"
#include "problem_interface.hpp"

#define PROPERTY(T, N)     \
  T Get ## N() const;     \
  void Set ## N(T value);

namespace gkls
{
  /* Penalty value of the generated function if x is not in D */
#define GKLS_MAX_VALUE        1E+100

  /* Value of the machine zero in the floating-point arithmetic */
#define GKLS_PRECISION        1.0E-10

  /* Default value of the paraboloid minimum */
#define GKLS_PARABOLOID_MIN   0.0

  /* Global minimum value: to be less than GKLS_PARABOLOID_MIN */
#define GKLS_GLOBAL_MIN_VALUE -1.0

  /* Max value of the parameter delta for the D2-type class function        */
  /* The parameter delta is chosen randomly from (0, GKLS_DELTA_MAX_VALUE ) */
#define GKLS_DELTA_MAX_VALUE  10.0

  /* Constant pi */
#ifndef PI
#define PI 3.14159265
#endif

  /* Error codes */
#define GKLS_OK                              0
#define GKLS_DIM_ERROR                       1
#define GKLS_NUM_MINIMA_ERROR                2
#define GKLS_FUNC_NUMBER_ERROR               3
#define GKLS_BOUNDARY_ERROR                  4
#define GKLS_GLOBAL_MIN_VALUE_ERROR          5
#define GKLS_GLOBAL_DIST_ERROR               6
#define GKLS_GLOBAL_RADIUS_ERROR             7
#define GKLS_MEMORY_ERROR                    8
#define GKLS_DERIV_EVAL_ERROR                9

  /* Reserved error codes */
#define GKLS_GREAT_DIM                      10
#define GKLS_RHO_ERROR                      11
#define GKLS_PEAK_ERROR                     12
#define GKLS_GLOBAL_BASIN_INTERSECTION      13

  /* Internal error codes */
#define GKLS_PARABOLA_MIN_COINCIDENCE_ERROR 14
#define GKLS_LOCAL_MIN_COINCIDENCE_ERROR    15
#define GKLS_FLOATING_POINT_ERROR           16

  typedef struct {
    double **local_min; /* list of local minimizers coordinates   */
    double *f;          /* list of local minima values            */
    double *w_rho;      /* list of radius weights                 */
    double *peak;       /* list of parameters gamma(i) =          */
    /*  = local minimum value(i) - paraboloid */
    /*    minimum within attraction regions   */
    /*    of local minimizer(i)               */
    double *rho;        /* list of attraction regions radii       */
  } T_GKLS_Minima;

  /* The structure of type T_GKLS_GlobalMinima contains information      */
  /* about the number of global minimizers and their                     */
  /* indices in the set of local minimizers                              */

  typedef struct {
    unsigned int num_global_minima; /* number of global minima    */
    unsigned int *gm_index;  /* list of indices of generated      */
    /* minimizers, which are the global ones (elements from 0     */
    /* to (num_global_minima - 1) of the list) and the local ones */
    /* (the resting elements of the list)                         */
  } T_GKLS_GlobalMinima;

  enum GKLSClass { Hard, Simple };
  enum GKLSFuncionType { TND, TD, TD2 };

  struct GKLSParameters
  {
    unsigned dimension;
    double globalMinimumValue;
    unsigned numberOfLocalMinima;
    double globalDistance;
    double globalRadius;
    GKLSFuncionType type;

    GKLSParameters() {}
    GKLSParameters(unsigned _dimension, double _globalMinimumValue,
      unsigned _numberOfLocalMinima, double _globalDistance,
      double _globalRadius, GKLSFuncionType _type) :
      dimension(_dimension),
      globalMinimumValue(_globalMinimumValue),
      numberOfLocalMinima(_numberOfLocalMinima),
      globalDistance(_globalDistance),
      globalRadius(_globalRadius),
      type(_type)
    {}
  };

  class GKLSFunction : public IGOProblem<double>
  {

  private:
    int mFunctionNumber;
    unsigned mDimension;
    bool mIsGeneratorMemoryAllocated;
    bool mIsDomainMemeoryAllocated;
    GKLSFuncionType mFunctionType;
    gkls::randomgenerator::GKLSRandomGenerator mRndGenerator;

    /*-------------- Variables accessible by the user --------------------- */
    double *GKLS_domain_left; /* left boundary vector of D  */
    /* D=[GKLS_domain_left; GKLS_domain_ight] */
    double *GKLS_domain_right;/* right boundary vector of D */

    unsigned int GKLS_dim;    /* dimension of the problem,        */
    /* 2<=test_dim<NUM_RND (see random) */
    unsigned int GKLS_num_minima; /* number of local minima, >=2  */

    double GKLS_global_dist;  /* distance from the paraboloid minimizer  */
    /* to the global minimizer                 */
    double GKLS_global_radius;/* radius of the global minimizer          */
    /* attraction region                       */
    double GKLS_global_value; /* global minimum value,                   */
    /* test_global_value < GKLS_PARABOLOID_MIN */
    T_GKLS_Minima GKLS_minima;
    /* see the structures type description     */
    T_GKLS_GlobalMinima GKLS_glob;

    /*--------------------------- Global variables ----------------------*/
    int isArgSet; /* isArgSet == 1 if all necessary parameters are set */

    double delta; /* parameter using in D2-type function generation;     */
    /* it is chosen randomly from the                      */
    /* open interval (0,GKLS_DELTA_MAX_VALUE)              */
    unsigned long rnd_counter; /* index of random array elements */

    double* rnd_num, *rand_condition;

    int GKLS_domain_alloc(); /* allocate boundary vectors   */
    void GKLS_domain_free(); /* deallocate boundary vectors */
    int GKLS_parameters_check() const;/* test the validity of the input parameters*/
    int GKLS_arg_generate(unsigned int nf, double* argmin = 0); /* test function generator */
    int GKLS_set_basins();
    void GKLS_free();        /* deallocate memory needed for the generator  */
    int GKLS_alloc();

    int GKLS_initialize_rnd(unsigned int, unsigned int, int);
    double GKLS_norm(const double *, const double *) const;
    int GKLS_coincidence_check() const;

  public:
    GKLSFunction();
    ~GKLSFunction();

    int SetFunctionNumber(int number);
    int GetFunctionNumber() const;
    PROPERTY(int, Dimension);
    PROPERTY(double, OptimumValue);
    PROPERTY(unsigned, NumberOfLocalMinima);
    PROPERTY(double, GlobalDistance);
    PROPERTY(double, GlobalRadius);
    PROPERTY(GKLSFuncionType, Type);
    PROPERTY(GKLSParameters, Parameters);

    void SetDefaultParameters();
    int CheckParameters() const;
    void SetFunctionClass(GKLSClass type, unsigned dimension);

    double Calculate(const double* x, int fNumber = 0) const override;

    double CalculateNDFunction(const double* x) const;
    double CalculateDFunction(const double* x) const;
    double CalculateD2Function(const double* x) const;

    double CalculateDFunctionDeriv(unsigned var_j, const double* x) const;
    double CalculateD2FunctionDeriv1(unsigned var_j, const double* x) const;
    double CalculateD2FunctionDeriv2(unsigned var_j, unsigned var_k, const double* x) const;

    int CalculateDFunctionGradient(const double* x, double* g) const;
    int CalculateD2FunctionGradient(const double* x, double* g) const;

    int CalculateD2FunctionHessian(const double* x, double** h) const;

    int GetOptimumPoint(double* argmin) const override;
    void GetBounds(double* lowerBound, double* upperBound) const override;
    int GetConstraintsNumber() const override;

    int SetGlobalMinimumPoint(double* argmin);
  };
}
#endif
