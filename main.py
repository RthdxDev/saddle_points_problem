import numpy as np
from typing import List

from src.problems import (
	SimpleVIProblem,
	BilinearSaddlePointProblem,
	OptimizationProblem,
	OptimizationResult,
	create_random_linear_problem,
	create_random_bilinear_problem
)
from src.algorithms import (
	ProjectionMethod,
	ExtragradientMethod,
	ExtragradientMethodWithRestarts,
	OptimizationAlgorithm,
	ball_projection,
	product_simplex_projection
)
from src.visualization import (
	plot_convergence,
	plot_convergence_comparison,
	print_comparison_table
)


def run_experiment(
	problem: OptimizationProblem,
	algorithms: List[OptimizationAlgorithm],
	x0: np.ndarray,
	max_iterations: int = 1000,
	eps: float = 1e-6,
) -> List[OptimizationResult]:
	results = []
	for algo in algorithms:
		result = algo.solve(
			problem=problem,
			x0=x0,
			max_iterations=max_iterations,
			eps=eps
		)
		results.append(result)
	
	return results


def experiment_simple_vi(
	dimension: int = 10,
	max_iterations: int = 100,
	eps: float = 1e-6
	) -> List[OptimizationResult]:

	"""
	Experiment: Simple VI problem with F(x) = x.
	"""
	problem = SimpleVIProblem(dimension=dimension)
	
	algorithms = [
		ProjectionMethod(proj=ball_projection),
		ExtragradientMethod(proj=ball_projection),
		ExtragradientMethodWithRestarts(proj=ball_projection)
	]
	
	x0 = np.ones(dimension)
	
	results = run_experiment(
		problem=problem,
		algorithms=algorithms,
		x0=x0,
		max_iterations=max_iterations,
		eps=eps,
	)

	plot_convergence(results, title="Simple VI: F(x) = x")
	print_comparison_table(results)
	return results


def experiment_linear_vi(
	dimension: int = 10,
	max_iterations: int = 100,
	eps: float = 1e-6,
	mu: float = 1) -> List[OptimizationResult]:
	"""
	Experiment: Linear VI problem with F(x) = Ax.
	"""
	problem = create_random_linear_problem(n=dimension, mu=mu, seed=4269)
	algorithms = [
		ProjectionMethod(proj=ball_projection),
		ExtragradientMethod(proj=ball_projection),
		ExtragradientMethodWithRestarts(proj=ball_projection)
	]
	x0 = np.ones(dimension)
	results = run_experiment(
		problem=problem,
		algorithms=algorithms,
		x0=x0,
		max_iterations=max_iterations,
		eps=eps,
	)
	plot_convergence(results, title="Linear VI Problem")
	print_comparison_table(results)
	return results


def experiment_bilinear_saddle_point(
	dimension: int = 10,
	max_iterations: int = 100,
	eps: float = 1e-6,
	) -> List[OptimizationResult]:
	"""
	Experiment: Bilinear saddle point problem (matrix game).
	"""
	problem = create_random_bilinear_problem(n=dimension,  seed=4269)
	
	algorithms = [
		ProjectionMethod(proj=product_simplex_projection),
		ExtragradientMethod(proj=product_simplex_projection),
	]
	
	x0 = np.zeros(2 * dimension)
	x0[0] = 1
	x0[dimension] = 1
	
	results = run_experiment(
		problem=problem,
		algorithms=algorithms,
		x0=x0,
		max_iterations=max_iterations,
		eps=eps,
	)
	
	plot_convergence(results, title="Bilinear Saddle Point Problem")
	print_comparison_table(results)

	return results


def main():
	results_exp1 = experiment_simple_vi()
	results_exp2 = experiment_linear_vi(max_iterations=500)
	results_exp3 = experiment_bilinear_saddle_point()

	all_results = [results_exp1, results_exp2, results_exp3]
	problem_names = ["Simple VI (F(x)=x)", "Linear VI (F(x)=Ax)", "Bilinear Saddle Point"]
	
	plot_convergence_comparison(
	    results_list=all_results,
	    problem_names=problem_names,
	    log_scale=True
	)


if __name__ == "__main__":
	main()
