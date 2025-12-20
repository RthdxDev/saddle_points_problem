import numpy as np
from abc import ABC, abstractmethod
import time

from problems import OptimizationProblem, OptimizationResult


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimization algorithms must implement the solve() method.
    """
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def solve(
        self,
        problem: OptimizationProblem,
        x0: np.ndarray,
        max_iterations: int = 1000,
        eps: float = 1e-6,
        **kwargs
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

    @abstractmethod
    def proj(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Projection operator onto the feasible set.
        
        Args:
            x: Point to project
            
        Returns:
            Projected point
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
    
    def __init__(self, step_size: float = 0.1):
        super().__init__("Projection Method")
        self.step_size = step_size
    
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
        
        for k in range(max_iterations):
            F_k = problem.operator(x_k)
            
            # Update: x_{k+1} = proj(x_k - a F(x_k))
            x_k = self.proj(x_k - self.step_size * F_k, **kwargs)
            
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
            parameters={"step_size": self.step_size}
        )
    
    def proj(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Projection onto the Euclidian ball.
        
        Args:
            x: Point to project
            
        Returns:
            Projected point
        """
        radius = kwargs.get("radius", 1.0)
        return x / np.linalg.norm(x) if np.linalg.norm(x) > radius else x


class ExtragradientMethod(OptimizationAlgorithm):
    """
    Extragradient method (Korpelevich, 1976).
    
    Two-step update:
    1) y_k = proj(x_k - a F(x_k))
    2) x_{k+1} = proj(x_k - a F(y_k))
    
    Requires only monotonicity and Lipschitz continuity.
    """
    
    def __init__(self, step_size: float = 0.1):
        super().__init__("Extragradient Method")
        self.step_size = step_size
    
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
        
        for k in range(max_iterations):
            F_k = problem.operator(x_k)
            y_k = self.proj(x_k - self.step_size * F_k, **kwargs)
            
            F_y = problem.operator(y_k)
            x_k = self.proj(x_k - self.step_size * F_y, **kwargs)
            
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
            parameters={"step_size": self.step_size}
        )
    
    def proj(self, x: np.ndarray, **kwargs) -> np.ndarray:
        a = kwargs.get('radius', 1.0)
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
