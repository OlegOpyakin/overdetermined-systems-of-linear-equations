import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Union
import warnings
from scipy.special import eval_legendre
from scipy.linalg import solve, LinAlgError
from numpy.linalg import cond


class LeastSquaresSolver:
    """
    Solver for least squares approximation with various basis functions.
    
    Solves the overdetermined system: A^T * B * A * x = A^T * B * y
    where A is the design matrix, B is the weight matrix, and x are coefficients.
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, 
                 weight_matrix: Optional[np.ndarray] = None):
        """
        Initialize the least squares solver.
        
        Args:
            x_data (np.ndarray): x coordinates of data points
            y_data (np.ndarray): y coordinates of data points  
            weight_matrix (np.ndarray, optional): Weight matrix B (identity by default)
        """
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")
        
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.n_points = len(x_data)
        
        # Set weight matrix (identity by default)
        if weight_matrix is None:
            self.weight_matrix = np.eye(self.n_points)
        else:
            if weight_matrix.shape != (self.n_points, self.n_points):
                raise ValueError("Weight matrix dimensions must match number of data points")
            self.weight_matrix = weight_matrix
    
    def set_weight_matrix(self, weight_matrix: np.ndarray):
        """Set or update the weight matrix B"""
        if weight_matrix.shape != (self.n_points, self.n_points):
            raise ValueError("Weight matrix dimensions must match number of data points")
        self.weight_matrix = weight_matrix
    
    def create_polynomial_basis(self, degree: int) -> np.ndarray:
        """
        Create design matrix for polynomial basis: [1, x, x^2, ..., x^degree]
        
        Args:
            degree (int): Degree of polynomial basis
            
        Returns:
            np.ndarray: Design matrix A of shape (n_points, degree+1)
        """
        A = np.zeros((self.n_points, degree + 1))
        for j in range(degree + 1):
            A[:, j] = self.x_data ** j
        return A
    
    def create_legendre_basis(self, degree: int) -> np.ndarray:
        """
        Create design matrix for Legendre polynomial basis: [P_0(x), P_1(x), ..., P_degree(x)]
        
        Args:
            degree (int): Maximum degree of Legendre polynomials
            
        Returns:
            np.ndarray: Design matrix A of shape (n_points, degree+1)
        """
        # Normalize x_data to [-1, 1] for Legendre polynomials
        x_min, x_max = np.min(self.x_data), np.max(self.x_data)
        x_normalized = 2 * (self.x_data - x_min) / (x_max - x_min) - 1
        
        A = np.zeros((self.n_points, degree + 1))
        for j in range(degree + 1):
            A[:, j] = eval_legendre(j, x_normalized)
        
        return A
    
    def create_trigonometric_basis(self, n_terms: int) -> np.ndarray:
        """
        Create design matrix for trigonometric basis: [1, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ...]
        
        Args:
            n_terms (int): Number of trigonometric terms (excluding constant)
            
        Returns:
            np.ndarray: Design matrix A of shape (n_points, 2*n_terms+1)
        """
        A = np.ones((self.n_points, 2 * n_terms + 1))  # Start with constant term
        
        col = 1
        for k in range(1, n_terms + 1):
            # Add sin(2πkx) and cos(2πkx) terms
            A[:, col] = np.sin(2 * np.pi * k * self.x_data)
            A[:, col + 1] = np.cos(2 * np.pi * k * self.x_data)
            col += 2
            
        return A
    
    def create_custom_trigonometric_basis(self, frequencies: List[float], 
                                        include_constant: bool = True) -> np.ndarray:
        """
        Create design matrix for custom trigonometric basis with specified frequencies.
        For LSQ3.DAT: sin(2πx) and sin(4πx)
        
        Args:
            frequencies (List[float]): List of frequencies for trigonometric functions
            include_constant (bool): Whether to include constant term
            
        Returns:
            np.ndarray: Design matrix A
        """
        n_cols = len(frequencies) + (1 if include_constant else 0)
        A = np.zeros((self.n_points, n_cols))
        
        col = 0
        if include_constant:
            A[:, 0] = 1.0
            col = 1
            
        for freq in frequencies:
            if freq == 0:
                A[:, col] = 1.0  # DC component
            else:
                A[:, col] = np.sin(2 * np.pi * freq * self.x_data)
            col += 1
            
        return A
    
    def solve_normal_equations(self, A: np.ndarray, method: str = 'gaussian') -> Tuple[np.ndarray, dict]:
        """
        Solve the normal equations A^T * B * A * x = A^T * B * y
        
        Args:
            A (np.ndarray): Design matrix
            method (str): Solution method ('gaussian' or 'conjugate_gradient')
            
        Returns:
            Tuple[np.ndarray, dict]: Solution vector and info dictionary
        """
        # Form the normal equations
        AtBA = A.T @ self.weight_matrix @ A
        AtBy = A.T @ self.weight_matrix @ self.y_data
        
        # Calculate condition number
        condition_number = cond(AtBA)
        
        info = {
            'condition_number': condition_number,
            'method': method,
            'matrix_rank': np.linalg.matrix_rank(AtBA)
        }
        
        if condition_number > 1e12:
            warnings.warn(f"Matrix is ill-conditioned (cond = {condition_number:.2e}). Results may be unreliable.")
        
        try:
            if method.lower() == 'gaussian':
                coefficients = solve(AtBA, AtBy)
                info['solver'] = 'Direct Gaussian elimination'
                
            elif method.lower() == 'conjugate_gradient':
                coefficients, cg_info = self._conjugate_gradient_solve(AtBA, AtBy)
                info.update(cg_info)
                
            else:
                raise ValueError("Method must be 'gaussian' or 'conjugate_gradient'")
                
        except LinAlgError as e:
            raise LinAlgError(f"Failed to solve normal equations: {e}")
        
        return coefficients, info
    
    def _conjugate_gradient_solve(self, A: np.ndarray, b: np.ndarray, 
                                max_iterations: int = 1000, 
                                tolerance: float = 1e-10) -> Tuple[np.ndarray, dict]:
        """
        Solve Ax = b using Conjugate Gradient method for symmetric positive definite A.
        
        Args:
            A (np.ndarray): Coefficient matrix (must be SPD)
            b (np.ndarray): Right-hand side vector
            max_iterations (int): Maximum iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            Tuple[np.ndarray, dict]: Solution and convergence info
        """
        n = len(b)
        x = np.zeros(n)
        r = b - A @ x  # Initial residual
        p = r.copy()   # Initial search direction
        
        residual_norms = [np.linalg.norm(r)]
        
        for iteration in range(max_iterations):
            Ap = A @ p
            alpha = (r @ r) / (p @ Ap)
            
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            residual_norm = np.linalg.norm(r_new)
            residual_norms.append(residual_norm)
            
            if residual_norm < tolerance:
                info = {
                    'solver': 'Conjugate Gradient',
                    'iterations': iteration + 1,
                    'final_residual': residual_norm,
                    'residual_norms': residual_norms,
                    'converged': True
                }
                return x, info
            
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new
        
        warnings.warn(f"Conjugate gradient did not converge in {max_iterations} iterations")
        info = {
            'solver': 'Conjugate Gradient',
            'iterations': max_iterations,
            'final_residual': np.linalg.norm(r),
            'residual_norms': residual_norms,
            'converged': False
        }
        return x, info
    
    def fit_polynomial(self, degree: int, method: str = 'gaussian') -> Tuple[np.ndarray, dict]:
        """
        Fit polynomial of specified degree to data.
        
        Args:
            degree (int): Degree of polynomial
            method (str): Solution method
            
        Returns:
            Tuple[np.ndarray, dict]: Coefficients and fitting info
        """
        A = self.create_polynomial_basis(degree)
        coefficients, info = self.solve_normal_equations(A, method)
        
        info['basis_type'] = 'polynomial'
        info['degree'] = degree
        info['design_matrix_shape'] = A.shape
        
        return coefficients, info
    
    def fit_legendre(self, degree: int, method: str = 'gaussian') -> Tuple[np.ndarray, dict]:
        """
        Fit Legendre polynomial of specified degree to data.
        
        Args:
            degree (int): Maximum degree of Legendre polynomials
            method (str): Solution method
            
        Returns:
            Tuple[np.ndarray, dict]: Coefficients and fitting info
        """
        A = self.create_legendre_basis(degree)
        coefficients, info = self.solve_normal_equations(A, method)
        
        info['basis_type'] = 'legendre'
        info['degree'] = degree
        info['design_matrix_shape'] = A.shape
        
        return coefficients, info
    
    def fit_trigonometric(self, n_terms: int, method: str = 'gaussian') -> Tuple[np.ndarray, dict]:
        """
        Fit trigonometric polynomial to data.
        
        Args:
            n_terms (int): Number of trigonometric terms
            method (str): Solution method
            
        Returns:
            Tuple[np.ndarray, dict]: Coefficients and fitting info
        """
        A = self.create_trigonometric_basis(n_terms)
        coefficients, info = self.solve_normal_equations(A, method)
        
        info['basis_type'] = 'trigonometric'
        info['n_terms'] = n_terms
        info['design_matrix_shape'] = A.shape
        
        return coefficients, info
    
    def fit_custom_trigonometric(self, frequencies: List[float], method: str = 'gaussian',
                               include_constant: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Fit custom trigonometric basis to data (for LSQ3.DAT).
        
        Args:
            frequencies (List[float]): List of frequencies
            method (str): Solution method
            include_constant (bool): Include constant term
            
        Returns:
            Tuple[np.ndarray, dict]: Coefficients and fitting info
        """
        A = self.create_custom_trigonometric_basis(frequencies, include_constant)
        coefficients, info = self.solve_normal_equations(A, method)
        
        info['basis_type'] = 'custom_trigonometric'
        info['frequencies'] = frequencies
        info['include_constant'] = include_constant
        info['design_matrix_shape'] = A.shape
        
        return coefficients, info
    
    def evaluate_polynomial(self, coefficients: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at given points"""
        result = np.zeros_like(x_eval)
        for i, coef in enumerate(coefficients):
            result += coef * (x_eval ** i)
        return result
    
    def evaluate_legendre(self, coefficients: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """Evaluate Legendre polynomial at given points"""
        # Normalize x_eval to [-1, 1] using same transformation as fitting
        x_min, x_max = np.min(self.x_data), np.max(self.x_data)
        x_normalized = 2 * (x_eval - x_min) / (x_max - x_min) - 1
        
        result = np.zeros_like(x_eval)
        for i, coef in enumerate(coefficients):
            result += coef * eval_legendre(i, x_normalized)
        return result
    
    def evaluate_trigonometric(self, coefficients: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
        """Evaluate trigonometric polynomial at given points"""
        result = coefficients[0] * np.ones_like(x_eval)  # Constant term
        
        col = 1
        k = 1
        while col < len(coefficients):
            result += coefficients[col] * np.sin(2 * np.pi * k * x_eval)
            if col + 1 < len(coefficients):
                result += coefficients[col + 1] * np.cos(2 * np.pi * k * x_eval)
            col += 2
            k += 1
            
        return result
    
    def evaluate_custom_trigonometric(self, coefficients: np.ndarray, x_eval: np.ndarray,
                                    frequencies: List[float], include_constant: bool = True) -> np.ndarray:
        """Evaluate custom trigonometric fit at given points"""
        result = np.zeros_like(x_eval)
        
        col = 0
        if include_constant:
            result += coefficients[0]
            col = 1
        
        for freq in frequencies:
            if freq == 0:
                result += coefficients[col]
            else:
                result += coefficients[col] * np.sin(2 * np.pi * freq * x_eval)
            col += 1
            
        return result
    
    def calculate_residuals(self, coefficients: np.ndarray, basis_type: str, **kwargs) -> np.ndarray:
        """Calculate residuals for the fit"""
        if basis_type == 'polynomial':
            y_fitted = self.evaluate_polynomial(coefficients, self.x_data)
        elif basis_type == 'legendre':
            y_fitted = self.evaluate_legendre(coefficients, self.x_data)
        elif basis_type == 'trigonometric':
            y_fitted = self.evaluate_trigonometric(coefficients, self.x_data)
        elif basis_type == 'custom_trigonometric':
            frequencies = kwargs.get('frequencies', [])
            include_constant = kwargs.get('include_constant', True)
            y_fitted = self.evaluate_custom_trigonometric(coefficients, self.x_data, 
                                                        frequencies, include_constant)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
        
        return self.y_data - y_fitted
    
    def calculate_rms_error(self, coefficients: np.ndarray, basis_type: str, **kwargs) -> float:
        """Calculate root mean square error"""
        residuals = self.calculate_residuals(coefficients, basis_type, **kwargs)
        return np.sqrt(np.mean(residuals**2))


def load_data_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from file (LSQ1.DAT, LSQ2.DAT, etc.)
    
    Args:
        filepath (str): Path to data file
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y data arrays
    """
    try:
        data = np.loadtxt(filepath)
        if data.shape[1] != 2:
            raise ValueError(f"Data file must have exactly 2 columns, found {data.shape[1]}")
        return data[:, 0], data[:, 1]
    except Exception as e:
        raise IOError(f"Failed to load data from {filepath}: {e}")


def create_diagonal_weight_matrix(n: int, weights: List[float]) -> np.ndarray:
    """
    Create diagonal weight matrix with specified diagonal elements.
    
    Args:
        n (int): Matrix size
        weights (List[float]): Diagonal weights
        
    Returns:
        np.ndarray: Diagonal weight matrix
    """
    if len(weights) != n:
        raise ValueError("Number of weights must match matrix size")
    
    return np.diag(weights)


# Example usage and testing functions
if __name__ == "__main__":
    # This section can be used for basic testing
    print("Least Squares Solver - Core Implementation")
    print("Use least_squares_analysis.py for full analysis and examples")