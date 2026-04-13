"""
Problem definitions wrapper for ZDT test suite using pymoo.
"""
from pymoo.problems import get_problem


def get_zdt_problem(name, n_var=30):
    """
    Return a pymoo ZDT problem instance.
    
    Parameters
    ----------
    name : str
        One of 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'.
    n_var : int
        Number of decision variables (default 30).
    
    Returns
    -------
    pymoo.core.problem.Problem
    """
    return get_problem(name, n_var=n_var)
