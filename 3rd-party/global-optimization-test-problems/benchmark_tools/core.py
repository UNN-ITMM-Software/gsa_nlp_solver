import numpy as np
import go_problems

class Solver:
    def __init__(self, problem):
        pass
    def Solve(self, problem):
        return None

def solve_class(problems_set, solver, verbose=False, eps_check=0.01, relative_eps=True, external_solution_check=True):
    assert eps_check > 0
    all_calcs = []
    solved_satus = []
    for i, problem in enumerate(problems_set.get_problems()):
        if relative_eps:
            bounds = problem.GetBounds()
            eps = np.linalg.norm(bounds[0] - np.array(bounds[1]), np.inf)*eps_check
        else:
            eps = eps_check
        if external_solution_check:
            problem.SetEps(eps)

        opt_point, opt_val, calcs = solver.Solve(problem)

        if external_solution_check:
            solved_satus.append(problem.IsStatsUpdateStopped())
        else:
            ref_point = np.array(problem.GetOptimumPoint())
            diff = np.linalg.norm(opt_point - ref_point, np.inf)
            if diff < eps:
                solved_satus.append(True)
            else:
                solved_satus.append(False)

        all_calcs.append(calcs)

        if verbose:
            print('Problem # {}: '.format(i + 1) + ('solved' if solved_satus[-1] else 'not solved') +
                    ' after {} trials'.format(calcs[0]))

    return all_calcs, solved_satus

class GrishClass():
    def get_problems(self):
        for i in range(1, 101):
            yield go_problems.create_grishagin_function(i)

class GKLSClass():
    def __init__(self, dim=2, type=go_problems.GKLSClass.Simple):
        self.dimension = dim
        self.gen_type = type
    def get_problems(self, ):
        for i in range(1, 101):
            yield go_problems.create_gkls_function(i, self.dimension, self.gen_type)
