import numpy as np
from abc import ABC, abstractmethod
import time

from src.problems import OptimizationProblem, OptimizationResult

def ball_projection(x: np.ndarray, **kwargs) -> np.ndarray:
    """
    Project point x onto the Euclidean ball of given radius.
    
    Args:
        x: Point to project
        radius: Radius of the ball
        
    Returns:
        Projected point
    """
    radius = kwargs.get("radius", 1.0)
    norm_x = np.linalg.norm(x)
    if norm_x <= radius:
        return x
    else:
        return (radius / norm_x) * x
    

def simplex_projection(x: np.ndarray, a = 1.0) -> np.ndarray:
    """
    Project point x onto the simplex {y | y >= 0, sum(y) = a}.
    
    Args:
        x: Point to project
        a: Sum constraint
        
    Returns:
        Projected point
    """
    n = len(x)
    y = np.sort(x)[::-1]
    
    K = 1
    for k in range(1, n + 1):
        sum_k = np.sum(y[:k])
        mu_k = (sum_k - a) / k
        
        if y[k-1] > mu_k:
            K = k
    
    mu = (np.sum(y[:K]) - a) / K
    return np.maximum(x - mu, 0)


def product_simplex_projection(z: np.ndarray, **kwargs) -> np.ndarray:
    """
    Project onto product of simplices: delta_n x delta_n y
    
    For bilinear saddle point problem where z = [x; y]:
    - x ∈ delta_n (first n components)
    - y ∈ delta_n (last n components)
    
    Args:
        z: Point to project (dimension 2n)
        a: Sum constraint for each simplex
        
    Returns:
        Projected point [proj_delta_n(x); proj_delta_n(y)]
    """
    n = z.shape[0] // 2
    x = z[:n]
    y = z[n:]
    
    a = kwargs.get("a", 1.0)
    x_proj = simplex_projection(x, a=a)
    y_proj = simplex_projection(y, a=a)
    
    return np.concatenate([x_proj, y_proj])

class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimization algorithms must implement the solve() method.
    """
    def __init__(self, name: str, proj = None):
        self.proj = proj
        self.name = name
    
    @abstractmethod
    def solve(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray,
        max_iterations: int = 1000,
        eps: float = 1e-6,
        **kwargs: dict
    ) -> OptimizationResult:
        """
        Solve the optimization problem.
        
        Args:
            problem: Optimization problem instance
            x0: Initial point
            max_iterations: Maximum number of iterations
            eps: Convergence tolerance
            **kwargs: Algorithm-specific parameters
            
        Returns:
            OptimizationResult
        """
        pass
    
    def _compute_convergence_error(
        self,
        x_k: np.ndarray,
        x_star: np.ndarray
    ) -> float:
        """
        Compute ||x* - x_k||_2.
        
        Args:
            x_k: Current iterate
            x_star: Exact solution
            
        Returns:
            L2 norm of the error
        """
        return float(np.linalg.norm(x_star - x_k))
    
    def _check_convergence(
        self,
        x_k: np.ndarray,
        x_star: np.ndarray,
        eps: float
    ) -> bool:
        """
        Check if algorithm has converged.
        
        Args:
            x_k: Current iterate
            x_star: Exact solution
            eps: Convergence tolerance
            
        Returns:
            True if ||x* - x_k||_2 < eps, False otherwise
        """
        return self._compute_convergence_error(x_k, x_star) < eps


class ProjectionMethod(OptimizationAlgorithm):
    """
    Projection method for VIs.
    
    Update rule:
    x_{k+1} = proj(x_k - a F(x_k))
    
    Requires strong monotonicity for convergence.
    """
    
    def __init__(self, proj = None):
        super().__init__("Projection Method", proj=proj)
    
    def solve(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray,
        max_iterations: int = 1000,
        eps: float = 1e-6,
        **kwargs
    ) -> OptimizationResult:
        start_time = time.time()

        L = problem.L
        a = 1 / (2 * L)
        
        x_k = x0.copy()
        x_star = problem.get_exact_solution()
        
        x_history = [x_k.copy()]
        convergence_errors = [self._compute_convergence_error(x_k, x_star)]
        
        converged = False
        
        for k in range(max_iterations):
            F_k = problem.operator(x_k)
            # Update: x_{k+1} = proj(x_k - a F(x_k))
            if self.proj is not None:
                x_k = self.proj(x_k - a * F_k, **kwargs)
            else:
                x_k = x_k - a * F_k
            
            x_history.append(x_k.copy())
            error = self._compute_convergence_error(x_k, x_star)
            convergence_errors.append(error)
            
            if error < eps:
                converged = True
                break
        
        computation_time = time.time() - start_time
        
        return OptimizationResult(
            algorithm_name=self.name,
            problem_name=problem.get_name(),
            x_history=x_history,
            iterations=len(x_history) - 1,
            convergence_errors=convergence_errors,
            final_solution=x_k,
            computation_time=computation_time,
            converged=converged,
            parameters={"step_size": a}
        )


class ExtragradientMethod(OptimizationAlgorithm):
    """
    Extragradient method (Korpelevich, 1976).
    
    Two-step update:
    1) y_k = proj(x_k - a F(x_k))
    2) x_{k+1} = proj(x_k - a F(y_k))
    
    Requires only monotonicity and Lipschitz continuity.
    """
    
    def __init__(self, proj = None):
        super().__init__("Extragradient Method", proj=proj)
    
    def solve(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray,
        max_iterations: int = 1000,
        eps: float = 1e-6,
        **kwargs
    ) -> OptimizationResult:
        start_time = time.time()

        L = problem.L
        a = 1 / (2 * L)
        
        x_k = x0.copy()
        x_star = problem.get_exact_solution()
        
        x_history = [x_k.copy()]
        convergence_errors = [self._compute_convergence_error(x_k, x_star)]
        
        converged = False
        
        for k in range(max_iterations):
            F_k = problem.operator(x_k)
            if self.proj is not None:
                y_k = self.proj(x_k - a * F_k, **kwargs)
            else:
                y_k = x_k - a * F_k
            
            F_y = problem.operator(y_k)
            if self.proj is not None:
                x_k = self.proj(x_k - a * F_y, **kwargs)
            else:
                x_k = x_k - a * F_y
            
            x_history.append(x_k.copy())
            error = self._compute_convergence_error(x_k, x_star)
            convergence_errors.append(error)
            
            if error < eps:
                converged = True
                break
        
        computation_time = time.time() - start_time
        
        return OptimizationResult(
            algorithm_name=self.name,
            problem_name=problem.get_name(),
            x_history=x_history,
            iterations=len(x_history) - 1,
            convergence_errors=convergence_errors,
            final_solution=x_k,
            computation_time=computation_time,
            converged=converged,
            parameters={"step_size": a}
        )
    
class ExtragradientMethodWithRestarts(OptimizationAlgorithm):
    """
    Restarted Averaged Extragradient Method for strongly convex problems.

    Applies the extragradient method in fixed-length blocks.
    Inside each block, auxiliary points y_k are averaged.
    Each restart is initialized from the averaged y.

    Requires Lipschitz continuity and known strong convexity parameter μ.
    """

    def __init__(self, proj=None):
        super().__init__("Extragradient Method With Restarts", proj=proj)

    def solve(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray,
        max_iterations: int = 1000,
        eps: float = 1e-6,
        **kwargs
    ) -> OptimizationResult:
        start_time = time.time()

        x_k = x0.copy()
        x_star = problem.get_exact_solution()

        x_history = [x_k.copy()]
        convergence_errors = [self._compute_convergence_error(x_k, x_star)]

        converged = False

        L = float(problem.L)
        mu = float(problem.mu)

        N_restart = int(np.ceil(L / mu))

        a = 1 / (2 * L)

        total_iterations = 0

        while total_iterations < max_iterations:
            y_sum = np.zeros_like(x_k)
            inner_steps = 0

            for _ in range(N_restart):
                if total_iterations >= max_iterations:
                    break

                F_k = problem.operator(x_k)

                if self.proj is not None:
                    y_k = self.proj(x_k - a * F_k, **kwargs)
                else:
                    y_k = x_k - a * F_k

                F_y = problem.operator(y_k)

                if self.proj is not None:
                    x_next = self.proj(x_k - a * F_y, **kwargs)
                else:
                    x_next = x_k - a * F_y

                x_k = x_next
                x_history.append(x_k.copy())

                error = self._compute_convergence_error(x_k, x_star)
                convergence_errors.append(error)

                y_sum += y_k
                inner_steps += 1

                total_iterations += 1

                if error < eps:
                    converged = True
                    break

            if converged or inner_steps == 0:
                break

            x_k = y_sum / inner_steps
            x_history.append(x_k.copy())
            convergence_errors.append(
                self._compute_convergence_error(x_k, x_star)
            )

        computation_time = time.time() - start_time

        return OptimizationResult(
            algorithm_name=self.name,
            problem_name=problem.get_name(),
            x_history=x_history,
            iterations=len(x_history) - 1,
            convergence_errors=convergence_errors,
            final_solution=x_k,
            computation_time=computation_time,
            converged=converged,
            parameters={
                "step_size": a
            }
        )

