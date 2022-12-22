/*
Copyright (C) 2018 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#include "solver.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>



using namespace ags;

namespace
{
    const double zeroHLevel = 1e-12;

    class ProblemInternal : public IGOProblem<double>
    {
    private:
      std::vector<NLPSolver::FuncPtr> mFunctions;
      std::vector<double> mLeftBound;
      std::vector<double> mRightBound;

      unsigned mDimension;
      unsigned mConstraintsNumber;

    public:
      ProblemInternal(const std::vector<NLPSolver::FuncPtr>& functions,
                      const std::vector<double>& leftBound, const std::vector<double>& rightBound)
      {
        mFunctions = functions;
        mConstraintsNumber = mFunctions.size() - 1;
        mDimension = leftBound.size();
        mLeftBound = leftBound;
        mRightBound = rightBound;
      }

      double Calculate(const double* y, int fNumber) const
      {
        return mFunctions[fNumber](y);
      }
      int GetConstraintsNumber() const
      {
        return mConstraintsNumber;
      }
      int GetDimension() const
      {
        return mDimension;
      }
      void GetBounds(double* left, double* right) const
      {
        for(size_t i = 0; i < mDimension; i++)
        {
          left[i] = mLeftBound[i];
          right[i] = mRightBound[i];
        }
      }
      int GetOptimumPoint(double* y) const {return 0;}
      double GetOptimumValue() const {return 0;}
    };
}

NLPSolver::NLPSolver() {}

void NLPSolver::SetParameters(const SolverParameters& params)
{
  mParameters = params;
}

void NLPSolver::SetProblem(std::shared_ptr<IGOProblem<double>> problem)
{
  mProblem = problem;
  NLP_SOLVER_ASSERT(mProblem->GetConstraintsNumber() <= solverMaxConstraints,
                    "Current implementation supports up to " + std::to_string(solverMaxConstraints) +
                    " nonlinear inequality constraints");
  InitLocalOptimizer();
}

void NLPSolver::SetProblem(const std::vector<FuncPtr>& functions,
                const std::vector<double>& leftBound, const std::vector<double>& rightBound)
{
  NLP_SOLVER_ASSERT(leftBound.size() == rightBound.size(), "Inconsistent dimensions of bounds");
  NLP_SOLVER_ASSERT(leftBound.size() > 0, "Zero problem dimension");
  NLP_SOLVER_ASSERT(functions.size() > 0, "Missing objective function");
  mProblem = std::make_shared<ProblemInternal>(functions, leftBound, rightBound);
  NLP_SOLVER_ASSERT(mProblem->GetConstraintsNumber() <= solverMaxConstraints,
                    "Current implementation supports up to " + std::to_string(solverMaxConstraints) +
                    " nonlinear inequality constraints");
  InitLocalOptimizer();
}

std::vector<unsigned> NLPSolver::GetCalculationsStatistics() const
{
  return mCalculationsCounters;
}

std::vector<double> NLPSolver::GetHolderConstantsEstimations() const
{
  return mHEstimations;
}

void NLPSolver::InitLocalOptimizer()
{
  std::vector<double> leftBound(mProblem->GetDimension());
  std::vector<double> rightBound(mProblem->GetDimension());
  mProblem->GetBounds(leftBound.data(), rightBound.data());

  double maxSize = 0;
  for(size_t i = 0; i < leftBound.size(); i++)
    maxSize = std::max(rightBound[i] - leftBound[i], maxSize);

  NLP_SOLVER_ASSERT(maxSize > 0, "Empty search domain");

  mLocalOptimizer.SetParameters(maxSize / 1000, maxSize / 100, 2);
}

void NLPSolver::InitDataStructures()
{
  double leftDomainBound[solverMaxDim], rightDomainBound[solverMaxDim];
  mProblem->GetBounds(leftDomainBound, rightDomainBound);
  mEvolvent = Evolvent(mProblem->GetDimension(), mParameters.evolventDensity,
    leftDomainBound, rightDomainBound);

  mNextPoints.resize(mParameters.numPoints);
  mOptimumEstimation.idx = -1;

  mZEstimations.resize(mProblem->GetConstraintsNumber() + 1);
  std::fill(mZEstimations.begin(), mZEstimations.end(),
            std::numeric_limits<double>::max());
  mNextIntervals.resize(mParameters.numPoints);
  mHEstimations.resize(mProblem->GetConstraintsNumber() + 1);
  std::fill(mHEstimations.begin(), mHEstimations.end(), 1.0);
  mCalculationsCounters.resize(mProblem->GetConstraintsNumber() + 1);
  std::fill(mCalculationsCounters.begin(), mCalculationsCounters.end(), 0);
  mQueue = PriorityQueue();
  mIterationsCounter = 0;
  mMinDelta = std::numeric_limits<double>::max();
  mMaxIdx = -1;
  mLocalR = fmax(mParameters.r / 2, 2.);
  mRho = pow((1. - 1. / mParameters.r) / (1 - 1. / mLocalR), 2);
  // mRho = pow((1. - 2. / mParameters.r) / (1 - 2. / mLocalR), 1);
}

void NLPSolver::ClearDataStructures()
{
  for (const auto& ptr : mSearchInformation)
    delete ptr;
  mSearchInformation.clear();
  mQueue = PriorityQueue();
}

Trial NLPSolver::Solve()
{
  return Solve([](const Trial&){ return false; });
}

Trial NLPSolver::Solve(std::function<bool(const Trial&)> external_stop)
{
  mNeedStop = false;
  InitDataStructures();
  FirstIteration();

  do {
    InsertIntervals();
    EstimateOptimum();

    if (mNeedRefillQueue || mQueue.size() < mParameters.numPoints)
      RefillQueue();
    CalculateNextPoints();
    MakeTrials();
    mNeedStop = mNeedStop || mMinDelta < mParameters.eps || external_stop(mOptimumEstimation);
    mIterationsCounter++;
  } while(mIterationsCounter < mParameters.itersLimit && !mNeedStop);

  ClearDataStructures();

  if (mParameters.refineSolution && mOptimumEstimation.idx == mProblem->GetConstraintsNumber())  {
    auto localTrial = mLocalOptimizer.Optimize(mProblem, mOptimumEstimation, mCalculationsCounters);
    int idx = mOptimumEstimation.idx;
    if (localTrial.idx == idx && localTrial.g[idx] < mOptimumEstimation.g[idx])
      mOptimumEstimation = localTrial;
  }

  return mOptimumEstimation;
}

void NLPSolver::FirstIteration()
{
  Trial leftBound = Trial(0);
  leftBound.idx = -1;
  Trial rightBound = Trial(1.);
  rightBound.idx = -1;

  for (size_t i = 1; i <= mParameters.numPoints; i++)
  {
    mNextPoints[i - 1] = Trial((double)i / (mParameters.numPoints + 1));
    mEvolvent.GetImage(mNextPoints[i - 1].x, mNextPoints[i - 1].y);
  }

  MakeTrials();
  EstimateOptimum();

  for (size_t i = 0; i <= mParameters.numPoints; i++)
  {
    Interval* pNewInterval;
    if (i == 0)
      pNewInterval = new Interval(leftBound, mNextPoints[i]);
    else if (i == mParameters.numPoints)
      pNewInterval = new Interval(mNextPoints[i - 1], rightBound);
    else
      pNewInterval = new Interval(mNextPoints[i - 1], mNextPoints[i]);
    pNewInterval->delta = pow(pNewInterval->pr.x - pNewInterval->pl.x,
                              1. / mProblem->GetDimension());
    mMinDelta = std::min(mMinDelta, pNewInterval->delta);
    auto insRes = mSearchInformation.insert(pNewInterval);
    UpdateAllH(insRes.first);
  }
  RefillQueue();
  CalculateNextPoints();
  MakeTrials();
  mIterationsCounter += 2;
}

void NLPSolver::MakeTrials()
{
  #pragma omp parallel for num_threads(mParameters.numPoints)//schedule(dynamic) 
  for (int i = 0; i < mNextPoints.size(); i++)
  {
    int idx = 0;
    while(idx < mProblem->GetConstraintsNumber())
    {
      mNextPoints[i].idx = idx;
      double val = mProblem->Calculate(mNextPoints[i].y, idx);
      mNextPoints[i].g[idx] = val;
      if (val > 0)
        break;
      idx++;
    }
    if (idx == mProblem->GetConstraintsNumber())
    {
        mNextPoints[i].idx = idx;
        mNextPoints[i].g[idx] = mProblem->Calculate(mNextPoints[i].y, idx);
    }
  }
  for (int i = 0; i < mNextPoints.size(); i++) {
      int idx = mNextPoints[i].idx;
      for (int j = 0; j <= idx; ++j)
          mCalculationsCounters[idx]++;
      if (idx > mMaxIdx)
      {
          mMaxIdx = idx;
          for (int i = 0; i < mMaxIdx; i++)
              mZEstimations[i] = -mParameters.epsR * mHEstimations[i];
          mNeedRefillQueue = true;
      }
      if (mNextPoints[i].idx == mMaxIdx &&
          mNextPoints[i].g[mMaxIdx] < mZEstimations[mMaxIdx])
      {
          mZEstimations[mMaxIdx] = mNextPoints[i].g[mMaxIdx];
          mNeedRefillQueue = true;
      }
  }
}

void NLPSolver::InsertIntervals()
{
  for (size_t i = 0; i < mParameters.numPoints; i++)
  {
    Interval* pOldInterval = mNextIntervals[i];
    Interval* pNewInterval = new Interval(mNextPoints[i], pOldInterval->pr);
    pOldInterval->pr = mNextPoints[i];
    pOldInterval->delta = pow(pOldInterval->pr.x - pOldInterval->pl.x,
                              1. / mProblem->GetDimension());
    pNewInterval->delta = pow(pNewInterval->pr.x - pNewInterval->pl.x,
                              1. / mProblem->GetDimension());
    mMinDelta = std::min(mMinDelta, pNewInterval->delta);
    mMinDelta = std::min(mMinDelta, pOldInterval->delta);

    auto insResult = mSearchInformation.insert(pNewInterval);
    bool wasInserted = insResult.second;
    if(!wasInserted)
      throw std::runtime_error("Error during interval insertion.");

    UpdateAllH(insResult.first);
    UpdateAllH(--insResult.first);

    if(!mNeedRefillQueue)
    {
      UpdateR(pNewInterval);
      UpdateR(mNextIntervals[i]);
      mQueue.push(pNewInterval);
      mQueue.push(pOldInterval);
    }

#ifdef USE_OpenCV
    if (mParameters.useDecisionTree == true)
      UpdateStatus(&(mNextPoints[i]));
#endif
  }
}

void NLPSolver::CalculateNextPoints()
{
  for(size_t i = 0; i < mParameters.numPoints; i++)
  {
    mNextIntervals[i] = mQueue.top();
    mQueue.pop();
    mNextPoints[i].x = GetNextPointCoordinate(mNextIntervals[i]);

    if (mNextPoints[i].x > mNextIntervals[i]->pr.x || mNextPoints[i].x < mNextIntervals[i]->pl.x)
      throw std::runtime_error("The next point is outside of the subdivided interval");
    else if (mNextPoints[i].x == mNextIntervals[i]->pr.x || mNextPoints[i].x == mNextIntervals[i]->pl.x) {
      mNeedStop = true;
      std::cout << "Warning: AGS stopped early! Two similar 1d points were generated." << std::endl;
    }

    mEvolvent.GetImage(mNextPoints[i].x, mNextPoints[i].y);
  }
}

void NLPSolver::RefillQueue()
{
  mQueue = PriorityQueue();
  for (const auto& pInterval : mSearchInformation)
  {
    UpdateR(pInterval);
    mQueue.push(pInterval);
  }
  mNeedRefillQueue = false;
}

void NLPSolver::EstimateOptimum()
{
  for (size_t i = 0; i < mNextPoints.size(); i++)
  {
    if (mOptimumEstimation.idx < mNextPoints[i].idx ||
        mOptimumEstimation.idx == mNextPoints[i].idx &&
        mOptimumEstimation.g[mOptimumEstimation.idx] > mNextPoints[i].g[mNextPoints[i].idx])
    {
      mOptimumEstimation = mNextPoints[i];
      if (mOptimumEstimation.idx == mProblem->GetConstraintsNumber() &&
          mOptimumEstimation.g[mOptimumEstimation.idx] < mParameters.stopVal)
        mNeedStop = true;
    }
  }
}

void NLPSolver::UpdateH(double newValue, int idx)
{
  if (newValue > mHEstimations[idx] || mHEstimations[idx] == 1.0 && newValue > zeroHLevel)
  {
    mHEstimations[idx] = newValue;
    mNeedRefillQueue = true;
  }
}

void NLPSolver::UpdateAllH(std::set<Interval*>::iterator iterator)
{
  Interval* pInterval = *iterator;
  if (pInterval->pl.idx < 0)
    return;

  if (pInterval->pl.idx == pInterval->pr.idx)
    UpdateH(fabs(pInterval->pr.g[pInterval->pr.idx] - pInterval->pl.g[pInterval->pl.idx]) /
                 pInterval->delta, pInterval->pl.idx);
  else
  {
    auto rightIterator = iterator;
    auto leftIterator = iterator;
    //right lookup
    ++rightIterator;
    while(rightIterator != mSearchInformation.end() && (*rightIterator)->pl.idx < pInterval->pl.idx)
      ++rightIterator;
    if (rightIterator != mSearchInformation.end() && (*rightIterator)->pl.idx >= pInterval->pl.idx)
    {
      int idx = pInterval->pl.idx;
      UpdateH(fabs((*rightIterator)->pl.g[idx] - pInterval->pl.g[idx]) /
              pow((*rightIterator)->pl.x - pInterval->pl.x, 1. / mProblem->GetDimension()), idx);
    }

    //left lookup
    --leftIterator;
    while(leftIterator != mSearchInformation.begin() && (*leftIterator)->pl.idx < pInterval->pl.idx)
      --leftIterator;
    if (leftIterator != mSearchInformation.begin() && (*leftIterator)->pl.idx >= pInterval->pl.idx)
    {
      int idx = pInterval->pl.idx;
      UpdateH(fabs((*leftIterator)->pl.g[idx] - pInterval->pl.g[idx]) /
              pow(pInterval->pl.x - (*leftIterator)->pl.x, 1. / mProblem->GetDimension()), idx);
    }
  }
}

void NLPSolver::UpdateR(Interval* i)
{
  i->R = CalculateR(i, mParameters.r);
  if(mParameters.mixedFastMode)
  {
    double localR = CalculateR(i, mLocalR)*mRho;
    if (localR > i->R)
    {
      i->R = localR;
      i->localR = true;
    }
  }
}

double NLPSolver::CalculateR(const Interval* i, const double r) const
{
  if(i->pl.idx == i->pr.idx)
  {
    const int v = i->pr.idx;
    return i->delta + pow((i->pr.g[v] - i->pl.g[v]) / (r * mHEstimations[v]), 2) / i->delta -
      2.*(i->pr.g[v] + i->pl.g[v] - 2*mZEstimations[v]) / (r * mHEstimations[v]);
  }
  else if(i->pl.idx < i->pr.idx)
    return 2*i->delta - 4*(i->pr.g[i->pr.idx] - mZEstimations[i->pr.idx]) / (r * mHEstimations[i->pr.idx]);
  else
    return 2*i->delta - 4*(i->pl.g[i->pl.idx] - mZEstimations[i->pl.idx]) / (r * mHEstimations[i->pl.idx]);
}

double NLPSolver::GetNextPointCoordinate(const Interval* i) const
{
  double x;
  double currentR = !i->localR ? mParameters.r : mLocalR;
  if(i->pr.idx == i->pl.idx)
  {
    const int v = i->pr.idx;
    double dg = i->pr.g[v] - i->pl.g[v];
    x = 0.5 * (i->pr.x + i->pl.x) -
      0.5*((dg > 0.) ? 1. : -1.) * pow(fabs(dg) / mHEstimations[v], mProblem->GetDimension()) / currentR;
  }
  else
    x = 0.5 * (i->pr.x + i->pl.x);

  return x;
}

bool solver_utils::checkVectorsDiff(const double* y1, const double* y2, size_t dim, double eps)
{
  for (size_t i = 0; i < dim; i++)
  {
    if (fabs(y1[i] - y2[i]) > eps)
      return true;
  }

  return false;
}

#ifdef USE_OpenCV
// ------------------------------------------------------------------------------------------------
void NLPSolver::UpdateStatus(Trial* trial)
{
  Trial* inflection = 0;

#ifdef USE_OpenCV
    if (mProblem->GetDimension() >= 2) 
    {
      UpdateStatusDecisionTreesMultiDims(trial, inflection);

      for (int i = 0; i < pointsForLocalMethod.size(); i++) 
      {
        LocalS(*pointsForLocalMethod[i]);
        if (this->mNeedStop)
          break;
      }
    }
    else 
    {
      bool isStartLocalMethod = false;
      isStartLocalMethod = UpdateStatusDecisionTrees(trial, inflection);
      if (isStartLocalMethod)
        LocalS(*inflection);
    }
#endif




  //PlotDecisionTrees();
}

void NLPSolver::CreateTree() 
{
  Mytree = cv::ml::DTrees::create();

  Mytree->setMinSampleCount(1);
  Mytree->setCVFolds(1);
  Mytree->setMaxDepth(DecisionTreesMaxDepth);
  Mytree->setRegressionAccuracy(DecisionTreesRegressionAccuracy);
}

void NLPSolver::PrepareDataForDecisionTree(int N) 
{
  X.create(N, mProblem->GetDimension(), CV_32F);
  ff.create(N, 1, CV_32FC1);

  indexForTrainingMax = 0;
  for (const auto& pInterval : mSearchInformation)
  {
    if (indexForTrainingMax >= N)
    {
      break;
    }

    fillDataForDecisionTree(&(pInterval->pl));
  }
}

void NLPSolver::fillDataForDecisionTree(Trial* point) 
{
  if (point->idx >= 0)
  {
    for (int j = 0; j < mProblem->GetDimension(); j++)
    {
      X.at<float>(indexForTrainingMax, j) = point->y[j];
    }
    ff.at<float>(indexForTrainingMax, 0) = point->g[point->idx];
    indexForTrainingMax++;
  }
}

void NLPSolver::FillTheSegment(int numPointPerDim)
{
  uniformPartition = cv::Mat(pow(numPointPerDim, (int)mProblem->GetDimension()), mProblem->GetDimension(), CV_32F);

  double lb[solverMaxDim];
  double ub[solverMaxDim];
  mProblem->GetBounds(lb, ub);
  double* value = new double[mProblem->GetDimension()];
  for (int j = 0; j < mProblem->GetDimension(); j++) 
  {
    value[j] = lb[j];
  }

  int zero = 0;
  recursiveFilling(zero, lb, ub, numPointPerDim, &uniformPartition, value, zero, zero);
}

void NLPSolver::recursiveFilling(int dim, const double* lb, const double* ub, int numPointPerDim, cv::Mat* uniformPartition, double*& value, int& idx, int reset) 
{
  double step = (ub[dim] - lb[dim]) / (numPointPerDim - 1);

  for (int r = reset + 1; r < mProblem->GetDimension(); r++) {
    value[r] = lb[r];
  }

  if (dim == mProblem->GetDimension() - 1) {

    for (int k = 0; k < numPointPerDim; k++) {
      for (int d = 0; d < mProblem->GetDimension(); d++) {
        (*uniformPartition).at<float>(k + numPointPerDim * idx, d) = value[d];
        if (d == dim)
          value[d] += step;
      }
    }
    idx++;
    return;
  }
  else {
    while (value[dim] < ub[dim]) {
      recursiveFilling(dim + 1, lb, ub, numPointPerDim, uniformPartition, value, idx, dim);
      value[dim] += step;
    }
  }
}

int NLPSolver::FindIndexOfNearestPoint(int numPointPerDim, Trial*& inflection) 
{
  int nearestPointIndex = 0;
  double distance = DBL_MAX;
  for (int j = 0; j < pow(numPointPerDim, (int)mProblem->GetDimension()); j++) 
  {
    double currentDistance = FindDistance(uniformPartition, j, inflection);
    if (currentDistance < distance) 
    {
      distance = currentDistance;
      nearestPointIndex = j;
    }
  }
  return nearestPointIndex;
}

double NLPSolver::FindDistance(cv::Mat point, int idx, Trial*& inflection) 
{
  double result = 0;
  for (int i = 0; i < mProblem->GetDimension(); i++) 
  {
    result += (inflection->y[i] - point.at<float>(idx, i)) * (inflection->y[i] - point.at<float>(idx, i));
  }
  return sqrt(result);
}

// Return true if we need to execute local method and false if we don't need to do so
bool NLPSolver::FindAndCheckPointWithNeighbours(int numPointPerDim, int nearestPointIndex, cv::Mat results) 
{
  int* masOfIndexes = new int[mProblem->GetDimension()];
  std::vector<int> checkVector;
  checkVector.push_back(nearestPointIndex);
  std::vector<int> errorVector;
  errorVector.push_back(nearestPointIndex);

  int j = 0;
  bool stopCheckingThisPoint = false;
  while (j < checkVector.size()) {
    if (stopCheckingThisPoint) {
      break;
    }
    int checkedValue = checkVector[j];
    for (int k = mProblem->GetDimension() - 1; k >= 0; k--) {
      masOfIndexes[k] = checkedValue % numPointPerDim;
      checkedValue = (int)(checkedValue / numPointPerDim);
    }
    //std::cout << "j = " << j << "; checkVector.size = " << checkVector.size() << "\n";

    // Теперь нужно найти соседей
    std::vector<int> neighbours = FindNeighbours(numPointPerDim, nearestPointIndex, masOfIndexes);

    // и осмотреть
    for (int k = 0; k < neighbours.size(); k++) {
      auto chk = std::find(errorVector.begin(), errorVector.end(), neighbours[k]);
      if (chk != errorVector.end()) {
        continue;
      }

      if (results.at<float>(neighbours[k], 0) < results.at<float>(checkVector[j], 0)) {
        //std::cout << "checked value = " << results.at<float>(checkVector[j], 0) << "; neighbour value = " << results.at<float>(neighbours[k], 0) << "\n";
        //std::cout << "\n";
        //return false;
        stopCheckingThisPoint = true;
        break;
      }
      else if (results.at<float>(neighbours[k], 0) == results.at<float>(checkVector[j], 0)) {
        checkVector.push_back(neighbours[k]);
      }
      errorVector.push_back(neighbours[k]);
    }
    j++;
  }
  return !stopCheckingThisPoint;
}

std::vector<int> NLPSolver::FindNeighbours(int numPointPerDim, int nearestPointIndex, int* masOfIndexes) 
{
  int* masOfIndexesForCalc = new int[mProblem->GetDimension()];
  bool flag = false;
  std::vector<int> neighbours;
  int mask[solverMaxDim]; // триарная маска, -1 - влево, +1 - в право, 0 - остаемся на месте.
  for (int i = 0; i < mProblem->GetDimension(); i++)
  {
    mask[i] = -1;
  }

  int maskIndex = 0; //индекс начиная с которого изменяется маска
  int startMaskIndex = 0; // куда сдвигаем maskIndex
  int maxMaskIndex = 0; // до куда дошли

  mask[0] = -2;

  for (int i = 0; i < pow(3, (int)mProblem->GetDimension()); i++)
  {
    for (int k = 0; k < mProblem->GetDimension(); k++) {
      masOfIndexesForCalc[k] = masOfIndexes[k];
    }

    maskIndex = startMaskIndex; //сдвигаем индекс маски
    while (1) //изменяем маску
    {

      if (maskIndex == mProblem->GetDimension())
      {
        break;
      }

      mask[maskIndex]++;
      if (mask[maskIndex] == 2)
      {
        mask[maskIndex] = -1;
        maskIndex++;
        if (maxMaskIndex < maskIndex)
          maxMaskIndex = maskIndex;
      }
      else
        break;
    }
    maskIndex = startMaskIndex; //сдвигаем индекс маски

    int neighbourIndex = 0;
    int N = mProblem->GetDimension();
    for (int k = 0; k < mProblem->GetDimension(); k++) //формируем координаты
    {
      masOfIndexes[k] = masOfIndexes[k] + mask[k];
      if (masOfIndexes[k] < 0 || masOfIndexes[k] >= numPointPerDim) {
        flag = true;
        break;
      }
      neighbourIndex += masOfIndexes[k] * pow(numPointPerDim, N - 1);
      N--;
    }
    if (flag) {
      continue;
    }
    if (neighbourIndex != nearestPointIndex)
      neighbours.push_back(neighbourIndex);
  }
  return neighbours;
}

void NLPSolver::UpdateStatusDecisionTreesMultiDims(Trial* trial, Trial*& inflection)
{
#ifdef USE_OpenCV
  //std::cout << indexForTrainingMax << "\n";
  //inflection - точка из которой хотим запуститиь метода
  inflection = trial;


  bool isStartLocalMethod = true;

  int N = mSearchInformation.size() - 1;


  if (N < 100 * mProblem->GetDimension())
    return;

  pointsForLocalMethod.clear();

  if (isFirst) 
  {
    PrepareDataForDecisionTree(N);

    CreateTree();

    Mytree->train(X, cv::ml::ROW_SAMPLE, ff); // тренирует дерево на TrainData

    // После того, как натренировали дерево на исходных данных
    // нужно организовать дискретную сетку (отображение наших точек на одномерный массив)
    // провести предсказание на новом одномерном массиве, найти точку, близкую к нашей пришедшей точке
    // и проверить ее соседей: находим значение меньшее -> возвращаем фолс; все значения больше -> возвращаем тру;
    // равное значение -> ее тоже проверяем

    // Равномерное заполнение отрезка (от левой до правой границы)
    int numPointPerDim = pow(300, 1.0 / mProblem->GetDimension());

    FillTheSegment(numPointPerDim);

    // На полученной сетке предсказываем значения функции
    cv::Mat results;
    Mytree->predict(uniformPartition, results);

    // На сетке ищем точку наиболее близкую к inflection
    int nearestPointIndex = FindIndexOfNearestPoint(numPointPerDim, inflection);

    //Надо как-то заполнить соседей и все что ниже завернуть в цикл, чтобы много раз прогонялось.
    //Вычислив индекс нужно посмотреть значение функции в соответствующей ячейке массива и сравнить с "ближайшим"
    //Дальше по ситуации, но нужен какой-то стек(?) куда я буду складывать точки, у которых совпадет значение функции
    //потому что их тоже надо проверять. Соответственно, нужно обернуть в цикл (по размеру этого стека?) проверку соседей
    //И подумать о том, как не зациклиться
    if (FindAndCheckPointWithNeighbours(numPointPerDim, nearestPointIndex, results)) {
      pointsForLocalMethod.push_back(inflection);
    }
    isFirst = false;
  }
  else {
    pointsForTree.push_back(inflection);

    if (pointsForTree.size() < 100 * mProblem->GetDimension())
      return;

    int sizeOfVector = pointsForTree.size();

    indexForTrainingMax = 0;
    for (int k = 0; k < sizeOfVector; ++k)
    {
      if (indexForTrainingMax >= sizeOfVector)
      {
        break;
      }

      fillDataForDecisionTree(pointsForTree[k]);
    }

    Mytree->train(X, cv::ml::ROW_SAMPLE, ff); // тренирует дерево на TrainData

    int numPointPerDim = pow(300, 1.0 / mProblem->GetDimension());

    cv::Mat results;
    Mytree->predict(uniformPartition, results);

    for (int i = 0; i < sizeOfVector; i++) {
      // На сетке ищем точку наиболее близкую к inflection
      int nearestPointIndex = FindIndexOfNearestPoint(numPointPerDim, inflection);

      if (FindAndCheckPointWithNeighbours(numPointPerDim, nearestPointIndex, results)) {
        pointsForLocalMethod.push_back(pointsForTree[i]);
      }
    }
    pointsForTree.clear();
  }

  return;
#else
  return;
#endif
}

std::vector<Trial*> NLPSolver::LocalS(Trial& point)
{

  //numberLocalMethodtStart++;

  std::vector<Trial*> localPoints;
  bool isStop = false;

  bool isLeastSquareMethod = false;


  HookeJeevesMethod(point, localPoints);


  int l = localPoints.size();

  if ((l == 0) || (isStop))
    return localPoints;


  if (isLeastSquareMethod)
    point.TypeColor = 1;
  point.pointStatus = local_min;

  if (l == 0)
    return localPoints;

  //Необходимо проверить метки у всех новых точек
  for (int i = 0; i < l; i++)
  {
    localPoints[i]->pointStatus = local_min;

    //TSearchInterval* li = localPoints[i]->leftInterval;
    //TSearchInterval* ri = localPoints[i]->rightInterval;

    //if (li != 0)
    //  li->status = TSearchInterval::educational_local_method;
    //if (ri != 0)
    //  ri->status = TSearchInterval::educational_local_method;
  }

  //if (localPoints[l - 1]->leftInterval == 0 || localPoints[l - 1]->rightInterval == 0)
  //  return localPoints;

  //Trial* lt = localPoints[l - 1]->leftInterval->pl;
  //Trial* rt = localPoints[l - 1]->rightInterval->RightPoint;

  //Trial* p1 = 0;
  //Trial* p2 = 0;

  //localPoints[l - 1]->lowAndUpPoints = 0;

  //p1 = lt;
  ////Обходим соседей и меняем метки
  //while ((p1->pointStatus == low) || (p1->pointStatus == low_inflection))
  //{
  //  TSearchInterval* i = p1->leftInterval;

  //  if ((p1->pointStatus == low_inflection))
  //    p1->pointStatus = low;

  //  if (i == 0)
  //    break;
  //  p1 = i->pl;
  //}

  //p2 = rt;

  //while ((p2->pointStatus == up) || (p2->pointStatus == low_inflection))
  //{
  //  TSearchInterval* i = p2->rightInterval;

  //  if ((p2->pointStatus == low_inflection))
  //    p2->pointStatus = up;

  //  if (i == 0)
  //    break;
  //  p2 = i->RightPoint;
  //}

  //PlotDecisionTrees();

  return localPoints;
}

void NLPSolver::HookeJeevesMethod(Trial& point, std::vector<Trial*>& localPoints)
{
  int addAllLocalPoints = 2;
  //TLocalMethod localMethod(&(pTask), point, addAllLocalPoints);

  //double initialStep = 0;
  //for (int i = 0; i < mProblem->GetDimension(); i++)
  //  initialStep += pTask.GetB()[i] - pTask.GetA()[i];
  //initialStep /= mProblem->GetDimension();
  //// начальный шаг равен среднему размеру стороны гиперкуба, умноженному на коэффициент
  //localMethod.SetEps(localVerificationEpsilon);
  //localMethod.SetInitialStep(0.07 * initialStep);
  //localMethod.SetMaxTrials(parameters.localVerificationIteration);
  auto newpoint = mLocalOptimizer.Optimize(mProblem, mOptimumEstimation, mCalculationsCounters);

  std::vector<Trial> points = mLocalOptimizer.GetSearchSequence();
  points.push_back(newpoint);

  int s = points.size();
  for (int i = 0; i < s; i++)
  {
    Trial* tmp = new Trial(points[i]);

    double newX[10];

    mEvolvent.GetAllPreimages(tmp->y, newX);

    tmp->x = newX[0];
    
    double z = mProblem->Calculate(tmp->y, 0);
    tmp->g[0] = z;

    tmp->TypeColor = 3;
    localPoints.push_back(tmp);
  }

  InsertLocalPoints(localPoints);
}

void NLPSolver::UpdateOptimum(Trial& t)
{
    if (mOptimumEstimation.idx < t.idx ||
      mOptimumEstimation.idx == t.idx &&
      mOptimumEstimation.g[mOptimumEstimation.idx] > t.g[t.idx])
    {
      mOptimumEstimation = t;
      if (mOptimumEstimation.idx == mProblem->GetConstraintsNumber() &&
        mOptimumEstimation.g[mOptimumEstimation.idx] < mParameters.stopVal)
        mNeedStop = true;
    }
  
}

void NLPSolver::InsertLocalPoints(const std::vector<Trial*>& points)
{
  for (size_t i = 0; i < points.size(); i++)
  {
    Trial* currentPoint = points[i];

    Interval* CoveringInterval = 0;

    for (const auto& pInterval : mSearchInformation)
    {
      if (pInterval->pl.x < points[i]->x && points[i]->x < pInterval->pr.x)
      {
        CoveringInterval = pInterval;
        break;
      }
    }
    
    if (!CoveringInterval)
      continue;
    if (!(currentPoint->x < CoveringInterval->pr.x || currentPoint->x > CoveringInterval->pl.x))
      continue;


    Interval* pOldInterval = CoveringInterval;
    Interval* pNewInterval = new Interval(*(points[i]), pOldInterval->pr);
    pOldInterval->pr = *(points[i]);
    pOldInterval->delta = pow(pOldInterval->pr.x - pOldInterval->pl.x,
      1. / mProblem->GetDimension());
    pNewInterval->delta = pow(pNewInterval->pr.x - pNewInterval->pl.x,
      1. / mProblem->GetDimension());
    mMinDelta = std::min(mMinDelta, pNewInterval->delta);
    mMinDelta = std::min(mMinDelta, pOldInterval->delta);

    auto insResult = mSearchInformation.insert(pNewInterval);
    bool wasInserted = insResult.second;
    if (!wasInserted)
      throw std::runtime_error("Error during interval insertion.");

    UpdateAllH(insResult.first);
    UpdateAllH(--insResult.first);


    UpdateOptimum(*currentPoint);
  }

  mNeedRefillQueue = true;

}

bool NLPSolver::UpdateStatusDecisionTrees(Trial* trial, Trial*& inflection)
{
#ifdef USE_OpenCV
  //inflection - точка из которой хотим запуститиь метода
  inflection = trial;

  //std::vector<Trial*>& leftLocalMinPoint, int& countPointLeft - точки слева от inflection
  //std::vector<Trial*>& RightLocalMinPoint, int& countPointRight - точки справа от inflection


  bool isStartLocalMethod = true;

  int N = mSearchInformation.size() - 1;

  std::vector<Trial*> leftLocalMinPoint; 
  std::vector<Trial*> RightLocalMinPoint;
  int countPointLeft; 
  int countPointRight;

  if (N < 10)
    return false;

  int flag = 0;

  cv::Mat X(N, 1, CV_32FC1);
  cv::Mat ff(N, 1, CV_32FC1);
  float h = 3.14 / N;

  //for (int i = 0; i < N; i++)
  int i = 0;
  int indexX = -1;
  std::vector< Trial*> allPoints(N);
  for (const auto& it : mSearchInformation)
  {
    if (i >= N)
    {
      std::cout << "Error point count!!!";
      break;
    }

    if (it->pl.idx >= 0)
    {
      X.at<float>(i, 0) = it->pl.x;
      ff.at<float>(i, 0) = it->pl.g[it->pl.idx];
      allPoints[i] = &(it->pl);

      i++;

      if ((it->pl.x) == trial->x)
        indexX = i;
    }
  }

  if (((N - indexX) <= 1) || (indexX <= 1))
    return false;


  cv::Ptr< cv::ml::DTrees > Mytree = cv::ml::DTrees::create();

  Mytree->setMinSampleCount(1);
  Mytree->setCVFolds(1);
  Mytree->setMaxDepth(DecisionTreesMaxDepth);
  Mytree->setRegressionAccuracy(DecisionTreesRegressionAccuracy);

  Mytree->train(X, cv::ml::ROW_SAMPLE, ff); // тренирует дерево на TrainData


  cv::Mat results;

  Mytree->predict(X, results);

  if (indexX == -1)
  {
    for (; indexX < results.rows; indexX++)
    {
      if (X.at<float>(indexX - 1, 0) <= trial->x && X.at<float>(indexX + 1, 0) >= trial->x)
        break;
    }
  }

  double pointFuncVal = results.at<float>(indexX, 0);
  double curFuncVal = results.at<float>(indexX, 0);
  double oldCurFuncVal = curFuncVal;
  int countLeftStep = 0;
  int lpi = indexX - 1;
  if (lpi <= 0 || lpi >= results.rows)
    std::cout << "ERR";
  for (; lpi > 0; lpi--)
  {
    curFuncVal = results.at<float>(lpi, 0);
    if (curFuncVal < oldCurFuncVal) // Значит слева убывает - конец
      break;
    else if (curFuncVal > oldCurFuncVal) // Значит слева возрастает - параболоид!
    {
      if (oldCurFuncVal != curFuncVal) // Перешагнули возрастающую ступеньку
      {
        oldCurFuncVal = curFuncVal;
        countLeftStep++;
      }
    }
    leftLocalMinPoint.push_back(allPoints[lpi]);
    countPointLeft++;
  }

  if (countLeftStep < 2) // мало ступенек слева
    return false;


  pointFuncVal = results.at<float>(indexX, 0);
  curFuncVal = results.at<float>(indexX, 0);
  oldCurFuncVal = curFuncVal;
  int countRightStep = 0;
  int rpi = indexX + 1;
  for (; rpi < results.rows; rpi++)
  {
    curFuncVal = results.at<float>(rpi, 0);
    if (curFuncVal < oldCurFuncVal) // Значит справа убывает - конец
      break;
    else if (curFuncVal > oldCurFuncVal)// Значит справа возрастает - параболоид!
    {
      if (oldCurFuncVal != curFuncVal) // Перешагнули возрастающую ступеньку
      {
        oldCurFuncVal = curFuncVal;
        countRightStep++;
      }
    }
    RightLocalMinPoint.push_back(allPoints[rpi]);
    countPointRight++;
  }

  if (countRightStep < 2) // мало ступенек справа
    return false;


  if ((countPointLeft + countPointRight + 1) < countPointInLocalMinimum)
    return false;

  double maxDelta = 1;
  if (RightLocalMinPoint.size() < 2 || leftLocalMinPoint.size() < 2)
    return false;
  double delta = RightLocalMinPoint[1]->x - leftLocalMinPoint[1]->x;
  double parR = 1.0 / Localr;
  if ((delta / maxDelta) < parR)
    isStartLocalMethod = true;
  else
    isStartLocalMethod = false;


  countPointLeft = 2;
  countPointRight = 2;

  return isStartLocalMethod;
#else
  return false;
#endif
}
#endif