import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from abc import ABC, abstractmethod


@dataclass
class OptimizationResult:
    algorithm_name: str
    problem_name: str
    x_history: List[np.ndarray]
    iterations: int
    convergence_errors: List[float]
    function_values: Optional[List[float]] = None
    final_solution: Optional[np.ndarray] = None
    computation_time: float = 0.0
    converged: bool = False
    parameters: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.final_solution is None and len(self.x_history) > 0:
            self.final_solution = self.x_history[-1]


class OptimizationProblem(ABC):
    @abstractmethod
    def operator(self, z: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_exact_solution(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    def get_dimension(self) -> int:
        return len(self.get_exact_solution())


class SimpleVIProblem(OptimizationProblem):
    """
    Variational Inequality problem with F(x) = x.

    The solution is x* = 0.

    Strongly monotone, Lipschitz continuous operator.
    """
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.x_star = np.zeros(dimension)
    
    def operator(self, z: np.ndarray) -> np.ndarray:
        return z.copy()
    
    def get_exact_solution(self) -> np.ndarray:
        return self.x_star.copy()
    
    @property
    def L(self) -> float:
        return 1.0
    
    @property
    def mu(self) -> float: 
        return 1.0
    
    def get_name(self) -> str:
        return f"Simple VI (F(x)=x, dim={self.dimension})"
    

class AffineVIProblem(OptimizationProblem):
    """
    Affine Variational Inequality problem with F(x) = A x + b.

    The solution is x* = 0.

    Strongly monotone, Lipschitz continuous operator.
    """
    def __init__(self, A: np.ndarray, b: Optional[np.ndarray] = None, x_star: Optional[np.ndarray] = None):
        self.A = A
        self.b = b if b is not None else np.zeros(A.shape[0])
        self.dimension = A.shape[0]
        self.x_star = x_star if x_star is not None else np.zeros(self.dimension)
        self.eigvals = np.linalg.eigvalsh(A)
    
    def operator(self, z: np.ndarray) -> np.ndarray:
        return self.A @ z + self.b
    
    def get_exact_solution(self) -> np.ndarray:
        return self.x_star.copy()
    
    @property
    def L(self) -> float:
        return self.eigvals.max()
    
    @property
    def mu(self) -> float:
        return self.eigvals.min()
    
    def get_name(self) -> str:
        return f"Affine VI (dim={self.dimension})"


class StronglyConvexGradientProblem(OptimizationProblem):
    """
    Gradient of strongly convex function with quartic term.
    
    f(x) = (mu/2)||x||^2 + (1/4) sum(x_i^4) + c^T x
    F(x) = nabla f(x) = mu * x + x^3 (element-wise) + c
    
    For c = 0, the exact solution is x* = 0.
    
    This is a strongly monotone operator with constant mu.
    The Lipschitz constant depends on the domain size.
    """
    
    def __init__(self, dimension: int = 10, mu: float = 1, c: Optional[np.ndarray] = None, radius: float = 1.0):
        """
        Args:
            dimension: Problem dimension
            mu: Strong monotonicity parameter (must be > 0)
            c: Linear term (if None, use zeros, so x* = 0)
            radius: Radius of constraint ball (used for computing L)
        """
        self.dimension = dimension
        self.__mu = mu
        self.c = c if c is not None else np.zeros(dimension)
        self.radius = radius
        
        # For c = 0, solution is x* = 0
        if np.allclose(self.c, 0):
            self.x_star = np.zeros(dimension)
        else:
            # Solve mu*x + x^3 + c = 0 using Newton's method
            self.x_star = self._solve_for_solution()
    
    def _solve_for_solution(self) -> np.ndarray:
        """
        Solve mu*x + x^3 + c = 0 using Newton's method (element-wise).
        
        For each component i:
        mu*x_i + x_i^3 + c_i = 0
        
        Newton update: x_new = x_old - f(x_old) / f'(x_old)
        where f(x) = mu*x + x^3 + c, f'(x) = mu + 3*x^2
        """
        x = np.zeros(self.dimension)
        
        for _ in range(100):  # Max iterations
            f_val = self.mu * x + x**3 + self.c
            f_prime = self.mu + 3 * x**2
            
            x_new = x - f_val / f_prime
            
            if np.linalg.norm(x_new - x) < 1e-12:
                break
            x = x_new
        
        return x
    
    def operator(self, z: np.ndarray) -> np.ndarray:
        """
        F(x) = mu * x + x^3 (element-wise) + c
        
        Args:
            x: Point at which to evaluate the operator
            
        Returns:
            F(x) = nabla f(x)
        """
        return self.mu * z + z**3 + self.c
    
    def get_exact_solution(self) -> np.ndarray:
        """Return the exact solution x*"""
        return self.x_star.copy()
    
    @property
    def L(self) -> float:
        """
        Lipschitz constant of F on ball of radius R.
        
        Jacobian: nabla F(x) = mu * I + 3 * diag(x^2)
        
        For ||x|| <= R, largest eigenvalue: mu + 3 * R^2
        
        Returns:
            Lipschitz constant L
        """
        return self.mu + 3 * self.radius ** 2
    
    @property
    def mu(self) -> float:
        """
        Strong monotonicity constant.
        
        The operator is strongly monotone with constant mu because:
        <F(x) - F(y), x - y> >= mu ||x - y||^2
        
        Returns:
            Strong monotonicity constant mu
        """
        return self.__mu
    
    def get_name(self) -> str:
        """Get problem name for display"""
        return f"Strongly Convex Gradient (mu={self.__mu:.2f}, dim={self.dimension})"
    
    def get_dimension(self) -> int:
        """Get problem dimension"""
        return self.dimension
    
    def verify_strong_monotonicity(self, num_tests: int = 100, seed: int = 42) -> bool:
        """
        Verify that the operator is indeed strongly monotone.
        
        Tests: <F(x) - F(y), x - y> >= mu ||x - y||^2
        
        Args:
            num_tests: Number of random tests
            seed: Random seed
            
        Returns:
            True if all tests pass
        """
        np.random.seed(seed)
        
        for _ in range(num_tests):
            # Sample random points in the ball
            x = np.random.randn(self.dimension)
            x = (self.radius / np.linalg.norm(x)) * x * np.random.rand()
            
            y = np.random.randn(self.dimension)
            y = (self.radius / np.linalg.norm(y)) * y * np.random.rand()
            
            # Compute F(x) - F(y)
            F_diff = self.operator(x) - self.operator(y)
            xy_diff = x - y
            
            # Check strong monotonicity
            lhs = np.dot(F_diff, xy_diff)
            rhs = self.__mu * np.linalg.norm(xy_diff)**2
            
            if lhs < rhs - 1e-10:  # Allow small numerical error
                return False
        
        return True


class BilinearSaddlePointProblem(OptimizationProblem):
    """
    Bilinear saddle point problem (matrix game).
    
    Problem:
    min_x max_y x^T A y
        
    The operator is:
    F(z) = [A y; -A^T x], z = [x; y]

    Consider the case where A is an n×n matrix.
    """
    
    def __init__(self, A: np.ndarray, x_star: Optional[np.ndarray] = None):
        self.A = A
        self.n = A.shape[0]
        self.x_star = x_star if x_star is not None else np.zeros(2 * self.n)
    
    def operator(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the operator F(z) = [A y; -A^T x].
        
        Args:
            z = [x; y]
            
        Returns:
            F(z) = [A y; -A^T x]
        """
        x = z[:self.n]
        y = z[self.n:]
        
        F_x = self.A @ y
        F_y = -self.A.T @ x
        
        return np.concatenate([F_x, F_y])

    @property
    def L(self) -> float:
        return np.linalg.norm(self.A, 2).item()
    
    @property
    def mu(self) -> float:
        # Not strongly monotone
        return 0.0
    
    def get_exact_solution(self) -> np.ndarray:
        return self.x_star.copy()
    
    def get_name(self) -> str:
        return f"Bilinear Saddle Point (dim={self.n}×{self.n})"
    
    def get_objective(self, z: np.ndarray) -> float:
        """
        Compute the objective value x^T A y.
        
        Args:
            z: Current point z = [x; y]
            
        Returns:
            Objective value x^T A y
        """
        x = z[:self.n]
        y = z[self.n:]
        return float(x.T @ self.A @ y)
    

def create_random_affine_problem(
    n: int = 10, 
    radius: float = 1,
    mu: float = 1,
    seed: Optional[int] = 4269) -> AffineVIProblem:
    """
    Create a random linear variational inequality problem.
    
    Args:
        n: Problem dimension
        mu: Strong monotonicity constant
        seed: Random seed
        radius: Radius for scaling x_star
        
    Returns:
        LinearVIProblem instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    M = np.random.randn(n, n)
    A = M.T @ M
    A = A / np.linalg.norm(A, 2) + mu * np.eye(n)

    x_star = np.random.randn(n)
    x_star = x_star / np.linalg.norm(x_star) * radius if np.linalg.norm(x_star) > radius else x_star

    b = -A @ x_star
    
    return AffineVIProblem(A, b, x_star)


def create_rock_paper_scissors_game(n: int = 3) -> BilinearSaddlePointProblem:
    """
    Create a cyclic zero-sum game where uniform distribution is the Nash equilibrium.
    
    For n=3: Rock-Paper-Scissors
    For general n: Cyclic game where strategy i beats strategy (i + 1) mod n
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i - j) % n == 1:
                A[i, j] = 1  # i beats j
            elif (j - i) % n == 1:
                A[i, j] = -1  # j beats i
    
    x_star = np.ones(2 * n) / n

    return BilinearSaddlePointProblem(A, x_star)


def create_doubly_stochastic_game(n: int = 10, seed: Optional[int] = 4269) -> BilinearSaddlePointProblem:
    """
    Generate a doubly stochastic matrix using Sinkhorn's algorithm.
    """
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.rand(n, n) + 0.1
    
    for _ in range(100):
        A = A / A.sum(axis=1, keepdims=True)
        A = A / A.sum(axis=0, keepdims=True)
    
    x_star = np.ones(2 * n) / n
    print('Condition number:', np.linalg.cond(A))
    return BilinearSaddlePointProblem(A, x_star)


def create_well_conditioned_doubly_stochastic(n: int = 10, epsilon: float = 0.1) -> np.ndarray:
    A = (1 - epsilon) * np.eye(n) + (epsilon / n) * np.ones((n, n))
    print('Condition number:', np.linalg.cond(A))
    return A