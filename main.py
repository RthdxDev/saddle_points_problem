import numpy as np
from typing import List

from src.problems import (
	SimpleVIProblem,
	BilinearSaddlePointProblem,
	OptimizationProblem,
	OptimizationResult,
	create_random_affine_problem,
	create_rock_paper_scissors_game,
	create_doubly_stochastic_game,
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


def experiment_affine_vi(
	dimension: int = 10,
	max_iterations: int = 100,
	eps: float = 1e-6,
	mu: float = 1,
	radius: float = 1,
	) -> List[OptimizationResult]:
	"""
	Experiment: Affine VI problem with F(x) = Ax + b.
	"""
	problem = create_random_affine_problem(n=dimension, mu=mu, radius=radius, seed=4269)
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
	plot_convergence(results, title="Affine VI Problem")
	print_comparison_table(results)
	return results


def experiment_rock_paper_scissors(
	dimension: int = 3,
	max_iterations: int = 100,
	eps: float = 1e-6,
	) -> List[OptimizationResult]:
	"""
	Experiment: Rock-Paper-Scissors game as a bilinear saddle point problem.
	"""
	problem = create_rock_paper_scissors_game(n=dimension)
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
	plot_convergence(results, title="Rock-Paper-Scissors Game")
	print_comparison_table(results)
	return results


def experiment_doubly_stochastic_game(
	dimension: int = 10,
	max_iterations: int = 100,
	eps: float = 1e-6,
	) -> List[OptimizationResult]:
	"""
	Experiment: Bilinear saddle point problem (matrix game).
	"""
	problem = create_doubly_stochastic_game(n=dimension,  seed=4269)
	
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
	
	plot_convergence(results, title="Doubly Stochastic Game")
	print_comparison_table(results)

	return results


def main():
	results_exp1 = experiment_simple_vi()
	results_exp2 = experiment_affine_vi(max_iterations=200)
	results_exp3 = experiment_rock_paper_scissors(max_iterations=200)
	results_exp4 = experiment_doubly_stochastic_game(max_iterations=200)

	all_results = [results_exp1, results_exp2, results_exp3, results_exp4]
	problem_names = ["Simple VI (F(x)=x)", "Affine VI (F(x)=Ax + b)", "Rock-Paper-Scissors Game", "Doubly Stochastic Game"]
	
	plot_convergence_comparison(
	    results_list=all_results,
	    problem_names=problem_names,
	    log_scale=True
	)


if __name__ == "__main__":
	main()
