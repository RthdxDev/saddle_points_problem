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
    seed: Optional[int] = None) -> AffineVIProblem:
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


def create_doubly_stochastic_game(n: int = 10, seed: Optional[int] = None) -> BilinearSaddlePointProblem:
    """
    Generate a game where uniform distribution is Nash equilibrium.
    
    Create a matrix where all row sums and column sums are equal.
    """
    if seed is not None:
        np.random.seed(seed)
    
    A_raw = np.random.randn(n, n)
    A = A_raw - A_raw.mean(axis=1, keepdims=True)
    A = A - A.mean(axis=0, keepdims=True)

    x_star = np.ones(2 * n) / n

    return BilinearSaddlePointProblem(A, x_star)