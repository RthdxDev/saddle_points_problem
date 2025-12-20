import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from tabulate import tabulate

from problems import OptimizationResult


def plot_convergence(
    results: List[OptimizationResult],
    title: Optional[str] = None,
    log_scale: bool = True,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot convergence curves for multiple algorithms.
    
    Plots ||x* - x_k||_2 vs iteration number for each algorithm.
    
    Args:
        results: List of OptimizationResult objects
        title: Plot title
        log_scale: Use logarithmic foy y
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for result in results:
        iterations = range(len(result.convergence_errors))
        plt.plot(
            iterations,
            result.convergence_errors,
            marker='o',
            markersize=3,
            label=result.algorithm_name,
            linewidth=2
        )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'$\|x^* - x_k\|_2$', fontsize=12)
    
    if title is None:
        if len(results) > 0:
            title = f"Convergence Comparison: {results[0].problem_name}"
        else:
            title = "Convergence Comparison"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_convergence_comparison(
    results_list: List[List[OptimizationResult]],
    problem_names: List[str],
    log_scale: bool = True,
    figsize: tuple = (15, 5)
) -> None:
    """
    Plot convergence comparison for several problems.
    
    Args:
        results_list: List of result lists, one per problem
        problem_names: Names of the problems
        log_scale: Use logarithmic for y
        figsize: Figure size
    """
    n_problems = len(results_list)
    fig, axes = plt.subplots(1, n_problems, figsize=figsize)
    
    if n_problems == 1:
        axes = [axes]
    
    for idx, (results, problem_name) in enumerate(zip(results_list, problem_names)):
        ax = axes[idx]
        
        for result in results:
            iterations = range(len(result.convergence_errors))
            ax.plot(
                iterations,
                result.convergence_errors,
                marker='o',
                markersize=3,
                label=result.algorithm_name,
                linewidth=2
            )
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(r'$\|x^* - x_k\|_2$', fontsize=11)
        ax.set_title(problem_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def print_comparison_table(
    results: List[OptimizationResult],
) -> None:
    """
    Print a comparison table of optimization results.
    
    Args:
        results: List of OptimizationResult objects
    """
    if len(results) == 0:
        print("No results.")
        return
    
    headers = [
        "Algorithm",
        "Iterations",
        "Final Error",
        "Converged",
        "Time (s)"
    ]
    
    table_data = []
    
    for result in results:
        final_error = result.convergence_errors[-1] if len(result.convergence_errors) > 0 else np.nan
        
        row = [
            result.algorithm_name,
            result.iterations,
            f"{final_error:.6e}",
            "Yes" if result.converged else "No",
            f"{result.computation_time:.4f}"
        ]
        
        table_data.append(row)
    
    print("\n" + "="*80)
    print(f"Comparision Table: {results[0].problem_name}")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")


def print_detailed_results(result: OptimizationResult) -> None:
    """
    Print detailed information about a single optimization result.
    
    Args:
        result: OptimizationResult object
    """
    print("\n" + "="*80)
    print(f"Detailed results: {result.algorithm_name}")
    print("="*80)
    print(f"Problem: {result.problem_name}")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {'Yes' if result.converged else 'No'}")
    print(f"Computation time: {result.computation_time:.4f} seconds")
    print(f"\nFinal solution shape: {result.final_solution.shape if result.final_solution is not None else 'N/A'}")
    print(f"Final error ||x* - x_k||_2: {result.convergence_errors[-1]:.6e}")
    
    if result.parameters:
        print(f"\nAlgorithm parameters:")
        for key, value in result.parameters.items():
            print(f"  {key}: {value}")
    
    errors = np.array(result.convergence_errors)
    print(f"\nConvergence statistics:")
    print(f"  Initial error: {errors[0]:.6e}")
    print(f"  Final error: {errors[-1]:.6e}")
    print(f"  Error reduction: {errors[0]/errors[-1]:.2e}x")
    print("="*80 + "\n")
