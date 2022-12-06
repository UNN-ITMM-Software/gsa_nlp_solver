/*
Copyright (C) 2018 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#pragma once

#include <stdexcept>
#include <string>

#define NLP_SOLVER_ERROR(msg) throw std::runtime_error(std::string(msg))
#define NLP_SOLVER_ASSERT(expr, msg) if(!(expr)) NLP_SOLVER_ERROR(msg)

namespace ags
{

const unsigned solverMaxDim = 5;
const unsigned solverMaxConstraints = 10;

enum Status
{
  low_inflection,
  low,
  up,
  up_inflection,
  local_min
};

struct Trial
{
  double x;
  double y[solverMaxDim];
  double g[solverMaxConstraints + 1];
  int idx;
  /// Цвет рисования точки
  int TypeColor = 0;
  /// Метка точки: потенциальный локальный минимум, убывающая, возрастающая, потенциальный локальный максимум
  Status pointStatus = up_inflection;
  Trial() {}
  Trial(double _x) : x(_x) {}
};

struct Interval
{
  Trial pl;
  Trial pr;
  double R;
  double delta;
  bool localR;
  Interval() : localR(false) {}
  Interval(const Trial& _pl, const Trial& _pr) : pl(_pl), pr(_pr), localR(false) {}
};

struct CompareIntervals
{
  bool operator() (const Interval* i1, const Interval* i2) const
  {
    return i1->pl.x < i2->pl.x;
  }
};

class CompareByR
{
public:
  bool operator() (const Interval* i1, const Interval* i2) const
  {
    return i1->R < i2->R;
  }
};

}
