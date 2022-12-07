/*
Copyright (C) 2018 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#pragma once

#include "data_types.hpp"
#include "evolvent.hpp"
#include "problem_interface.hpp"
#include "local_optimizer.hpp"

#include <vector>
#include <memory>
#include <queue>
#include <set>
#include <limits>
#include <functional>

#ifdef USE_OpenCV
#include "opencv.hpp"
#include "highgui.hpp"
#include "ml/ml.hpp"
#endif

namespace ags
{

struct SolverParameters
{
  double eps = 0.01; //method tolerance. Less value -- better search precision, less probability of early stop.
  double stopVal = std::numeric_limits<double>::lowest(); //method stops after objective becomes less than this value
  double r = 2; //reliability parameter. Higher value of r -- slower convergence, higher chance to cache the global minima.
  unsigned numPoints = 6; //number of new points per iteration. > 1 is useless in current implementation.
  unsigned itersLimit = 20000; // max number of iterations.
  unsigned evolventDensity = 12; // density of evolvent. By default density is 2^-12 on hybercube [0,1]^N,
  // which means that maximum search accuracyis 2^-12. If search hypercube is large the density can be increased accordingly to achieve better accuracy.
  double epsR = 0.001; // parameter which prevents method from paying too much attention to constraints. Greater values of this parameter speed up convergence,
  // but global minima can be lost.
  bool refineSolution = false; //if true, the fibal solution will be refined with the HookJeves method.
  bool mixedFastMode = false; //if true, two versions of characteristics with different values of the r parameter are used.
  //It allows to speedup the method in case of overestimation io r

  bool useDecisionTree = false;

  SolverParameters() {}
  SolverParameters(double _eps, double _r,
      double epsR_, unsigned _trialsLimit) :
        eps(_eps), r(_r), itersLimit(_trialsLimit), epsR(epsR_)
  {}
};

class NLPSolver
{
protected:
  using PriorityQueue =
    std::priority_queue<Interval*, std::vector<Interval*>, CompareByR>;

  HookeJeevesOptimizer mLocalOptimizer;

  SolverParameters mParameters;
  std::shared_ptr<IGOProblem<double>> mProblem;
  Evolvent mEvolvent;

  std::vector<double> mHEstimations;
  std::vector<double> mZEstimations;
  std::vector<Trial> mNextPoints;
  PriorityQueue mQueue;
  std::set<Interval*, CompareIntervals> mSearchInformation;
  std::vector<Interval*> mNextIntervals;
  Trial mOptimumEstimation;

  std::vector<unsigned> mCalculationsCounters;
  unsigned mIterationsCounter;
  bool mNeedRefillQueue;
  bool mNeedStop;
  double mMinDelta;
  int mMaxIdx;
  double mLocalR;
  double mRho;

#ifdef USE_OpenCV
  //  ƒл€ многомерных деревьев решений=================================================================================================
  cv::Mat X;
  cv::Mat ff;
  // –авномерна€ сетка, использкюща€ при выборе интервала, хран€щего локальный минимум с использованием деревьев решений
  cv::Mat uniformPartition;
  cv::Ptr< cv::ml::DTrees > Mytree;
  //cv::Mat results;
  std::vector<Trial*> pointsForTree;
  std::vector<Trial*> pointsForLocalMethod;
  bool isFirst = true;
  int indexForTrainingMax = 0;
  std::vector<Trial*> localMins;
  int countOfRepetitions = 0;

  int DecisionTreesMaxDepth = 6;
  double DecisionTreesRegressionAccuracy = 0.01;
  int countPointInLocalMinimum = 5;
  double Localr = 16;
#endif

  void InitLocalOptimizer();
  void FirstIteration();
  void MakeTrials();
  void InsertIntervals();
  void CalculateNextPoints();
  void RefillQueue();
  void EstimateOptimum();

  void InitDataStructures();
  void ClearDataStructures();

  void UpdateAllH(std::set<Interval*>::iterator);
  void UpdateH(double newValue, int index);
  void UpdateR(Interval*);
  double CalculateR(const Interval*, const double) const;
  double GetNextPointCoordinate(const Interval*) const;

#ifdef USE_OpenCV
  void UpdateStatusDecisionTreesMultiDims(Trial* trial, Trial*& inflection);
  std::vector<Trial*> LocalS(Trial& point);
  bool UpdateStatusDecisionTrees(Trial* trial, Trial*& inflection);
  void CreateTree();
  void PrepareDataForDecisionTree(int N);
  void fillDataForDecisionTree(Trial* point);
  void FillTheSegment(int numPointPerDim);
  void recursiveFilling(int dim, const double* lb, const double* ub, int numPointPerDim, cv::Mat* uniformPartition, double*& value, int& index, int reset);
  int FindIndexOfNearestPoint(int numPointPerDim, Trial*& inflection);
  double FindDistance(cv::Mat point, int index, Trial*& inflection);
  bool FindAndCheckPointWithNeighbours(int numPointPerDim, int nearestPointIndex, cv::Mat results);
  std::vector<int> FindNeighbours(int numPointPerDim, int nearestPointIndex, int* masOfIndexes);
#endif
public:
  using FuncPtr = std::function<double(const double*)>;
  NLPSolver();

  void SetParameters(const SolverParameters& params);
  void SetProblem(std::shared_ptr<IGOProblem<double>> problem);
  void SetProblem(const std::vector<FuncPtr>& functions,
                  const std::vector<double>& leftBound, const std::vector<double>& rightBound);

  Trial Solve(std::function<bool(const Trial&)> external_stop);
  Trial Solve();
  std::vector<unsigned> GetCalculationsStatistics() const;
  std::vector<double> GetHolderConstantsEstimations() const;

#ifdef USE_OpenCV
  void UpdateStatus(Trial* trial);
  void NLPSolver::HookeJeevesMethod(Trial& point, std::vector<Trial*>& localPoints);
  void InsertLocalPoints(const std::vector<Trial*>& points);
  void UpdateOptimum(Trial& t);
#endif
};

namespace solver_utils
{
  bool checkVectorsDiff(const double* y1, const double* y2, size_t dim, double eps);
}

}
