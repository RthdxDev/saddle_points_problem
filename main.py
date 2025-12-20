import numpy as np
from typing import List

from src.problems import (
    SimpleVIProblem,
    BilinearSaddlePointProblem,
    OptimizationProblem,
    OptimizationResult,
    create_random_affine_problem
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
    verbose: bool = True
) -> List[OptimizationResult]:
    """
    Run experiments with multiple algorithms on a single problem.
    
    Args:
        problem: Optimization problem to solve
        algorithms: List of algorithms to test
        x0: Initial point
        max_iterations: Maximum iterations per algorithm
        eps: Convergence tolerance
        verbose: Print progress information
        
    Returns:
        List of OptimizationResult objects
    """
    results = []
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running experiments for: {problem.get_name()}")
        print(f"{'=' * 80}")
        print(f"Initial point: {x0}")
        print(f"Maximum iterations: {max_iterations}")
        print(f"Tolerance: {eps}")
        print(f"Number of algorithms: {len(algorithms)}\n")
    
    for algo in algorithms:
        if verbose:
            print(f"Running {algo.name}...")
        
        result = algo.solve(
            problem=problem,
            x0=x0,
            max_iterations=max_iterations,
            eps=eps
        )
        
        results.append(result)
        
        if verbose:
            status = "  Converged" if result.converged else "   Not converged"
            print(f"  {status} in {result.iterations} iterations (Time: {result.computation_time:.4f}s)")
            print(f"  Final error: {result.convergence_errors[-1]:.6e}\n")
    
    return results


def experiment_simple_vi(dimension: int = 10):
    """
    Experiment: Simple VI problem with F(x) = x.
    """
    # print("\n" + "=" * 80)
    # print("Experiment: Simple Variational Inequality: F(x) = x")
    # print("=" * 80)
    
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
        max_iterations=100,
        eps=1e-6,
        verbose=True
    )

    plot_convergence(results, title="Simple VI: F(x) = x")
    print_comparison_table(results)
    
    return results


def experiment_affine_vi(dimension: int = 5, mu: float = 1, max_iterations: int = 100):
    problem = create_random_affine_problem(n=dimension, mu=mu, seed=4269)
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
        eps=1e-6,
        verbose=True
    )
    plot_convergence(results, title="Affine VI Problem")
    print_comparison_table(results)
    return results


def experiment_bilinear_saddle_point(dimension: int = 5):
    """
    Experiment: Bilinear saddle point problem (matrix game).
    """
    print("\n" + "=" * 80)
    print("Experiment: Bilinear Saddle Point Problem")
    print("=" * 80)
    
    np.random.seed(42)
    A = np.random.randn(dimension, dimension)
    A = (A - A.T) / 2
    
    problem = BilinearSaddlePointProblem(A=A, x_star=np.zeros(2 * dimension))
    
    algorithms = [
        ProjectionMethod(proj=product_simplex_projection),
        ExtragradientMethod(proj=product_simplex_projection),
        ExtragradientMethodWithRestarts(proj=product_simplex_projection)
    ]
    
    x0 = np.ones(2 * dimension)
    
    results = run_experiment(
        problem=problem,
        algorithms=algorithms,
        x0=x0,
        max_iterations=1000,
        eps=1e-6,
        verbose=True
    )
    
    plot_convergence(results, title="Bilinear Saddle Point Problem")
    print_comparison_table(results)
    
    return results


def main():
    """
    Run all experiments.
    """
    # print("\n" + "="*80)
    # print(" "*20 + "OPTIMIZATION EXPERIMENTS")
    # print("="*80)
    # print("Testing various algorithms on VI and saddle point problems\n")
    
    results_exp1 = experiment_simple_vi()
    results_exp2 = experiment_affine_vi(max_iterations=500)
    # results_exp2 = experiment_bilinear_saddle_point()
    
    # print("\n" + "="*80)
    # print("COMPREHENSIVE SUMMARY")
    # print("="*80)
    
    # all_results = [results_exp1]
    # problem_names = ["Simple VI (F(x)=x)"]
    
    # plot_convergence_comparison(
    #     results_list=all_results,
    #     problem_names=problem_names,
    #     log_scale=True
    # )
    
    print("\n" + "="*80)
    print("END OF EXPERIMENTS")

if __name__ == "__main__":
    main()
