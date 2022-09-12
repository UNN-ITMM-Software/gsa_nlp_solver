#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "problem_wrapper.hpp"
#include "grishagin_function.hpp"
#include "gkls_function.hpp"

namespace py = pybind11;
using namespace vagrish;
using namespace gkls;

GOTestProblemWrapper<double> create_gkls_function(int idx, int dimension, GKLSClass cl, GKLSFuncionType type)
{
  auto f = std::make_shared<GKLSFunction>();
  f->SetFunctionClass(cl, dimension);
  f->SetType(type);
  f->SetFunctionNumber(idx);
  return GOTestProblemWrapper<double>(f);
}

GOTestProblemWrapper<double> create_grishagin_function(int idx)
{
  auto f = std::make_shared<GrishaginFunction>();
  f->SetFunctionNumber(idx);
  return GOTestProblemWrapper<double>(f);
}

PYBIND11_MODULE(go_problems, m)
{
  m.def("create_grishagin_function", &create_grishagin_function,
  "Create Grishagin problem", py::arg("function_number"));

  py::enum_<GKLSClass>(m, "GKLSClass")
    .value("Simple", Simple)
    .value("Hard", Hard)
    .export_values();

  py::enum_<GKLSFuncionType>(m, "GKLSFuncionType")
    .value("ND", TND)
    .value("D", TD)
    .value("D2", TD2)
    .export_values();

  m.def("create_gkls_function", &create_gkls_function,
  "Create GKLS problem", py::arg("function_number"),
                         py::arg("dimension") = 2,
                         py::arg("class") = Simple,
                         py::arg("type") = TD);

  py::class_<GOTestProblemWrapper<double>>(m, "GOTestProblemWrapper")
    .def("Calculate", (double (GOTestProblemWrapper<double>::*)(const std::vector<double>&, int) const)
                      &GOTestProblemWrapper<double>::Calculate, py::arg("y"), py::arg("fNumber") = 0)
    .def("GetDimension", &GOTestProblemWrapper<double>::GetDimension)
    .def("GetConstraintsNumber", &GOTestProblemWrapper<double>::GetConstraintsNumber)
    .def("GetOptimumPoint", (std::vector<double>(GOTestProblemWrapper<double>::*)() const)&GOTestProblemWrapper<double>::GetOptimumPoint)
    .def("GetOptimumValue", &GOTestProblemWrapper<double>::GetOptimumValue)
    .def("GetBounds", (std::pair<std::vector<double>, std::vector<double>>(GOTestProblemWrapper<double>::*)() const)
                      &GOTestProblemWrapper<double>::GetBounds)
    .def("GetCalculationsStatistics", &GOTestProblemWrapper<double>::GetCalculationsStatistics)
    .def("SetEps", &GOTestProblemWrapper<double>::SetEps)
    .def("IsStatsUpdateStopped", &GOTestProblemWrapper<double>::IsStatsUpdateStopped)
    .def("GetLoggedTrials", &GOTestProblemWrapper<double>::GetLoggedTrials)
    .def("EnableLogging", &GOTestProblemWrapper<double>::EnableLogging)
    ;
}
