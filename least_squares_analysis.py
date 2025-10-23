import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import os
from least_squares_solver import (
    LeastSquaresSolver, 
    load_data_file, 
    create_diagonal_weight_matrix
)


def analyze_lsq1_data():
    """
    Analyze LSQ1.DAT - Linear relationship approximation
    Compare polynomial basis vs Legendre basis, and different weight matrices
    """
    print("=" * 60)
    print("ANALYSIS OF LSQ1.DAT - Linear Relationship")
    print("=" * 60)
    
    # Load data
    x_data, y_data = load_data_file('DATA/LSQ1.DAT')
    print(f"Loaded {len(x_data)} data points from LSQ1.DAT")
    
    # Create solver with identity weight matrix
    solver_identity = LeastSquaresSolver(x_data, y_data)
    
    # Fit with polynomial basis (degree 1 for linear)
    coeffs_poly, info_poly = solver_identity.fit_polynomial(degree=1, method='gaussian')
    rms_poly = solver_identity.calculate_rms_error(coeffs_poly, 'polynomial')
    
    print(f"\nPolynomial Basis (y = b₀ + b₁x):")
    print(f"  Coefficients: b₀ = {coeffs_poly[0]:.6f}, b₁ = {coeffs_poly[1]:.6f}")
    print(f"  RMS Error: {rms_poly:.6f}")
    print(f"  Condition Number: {info_poly['condition_number']:.2e}")
    
    # Fit with Legendre basis (degree 1)
    coeffs_legendre, info_legendre = solver_identity.fit_legendre(degree=1, method='gaussian')
    rms_legendre = solver_identity.calculate_rms_error(coeffs_legendre, 'legendre')
    
    print(f"\nLegendre Basis (y = b₀P₀(x) + b₁P₁(x)):")
    print(f"  Coefficients: b₀ = {coeffs_legendre[0]:.6f}, b₁ = {coeffs_legendre[1]:.6f}")
    print(f"  RMS Error: {rms_legendre:.6f}")
    print(f"  Condition Number: {info_legendre['condition_number']:.2e}")
    
    # Create arbitrary weight matrix
    weights = [1.0, 2.0, 1.5, 0.8, 1.2]  # Different weights for each point
    weight_matrix = create_diagonal_weight_matrix(len(x_data), weights)
    solver_weighted = LeastSquaresSolver(x_data, y_data, weight_matrix)
    
    coeffs_weighted, info_weighted = solver_weighted.fit_polynomial(degree=1, method='gaussian')
    rms_weighted = solver_weighted.calculate_rms_error(coeffs_weighted, 'polynomial')
    
    print(f"\nWeighted Polynomial Basis (arbitrary weights):")
    print(f"  Weights: {weights}")
    print(f"  Coefficients: b₀ = {coeffs_weighted[0]:.6f}, b₁ = {coeffs_weighted[1]:.6f}")
    print(f"  RMS Error: {rms_weighted:.6f}")
    print(f"  Condition Number: {info_weighted['condition_number']:.2e}")
    
    # Visualization
    plot_lsq1_analysis(solver_identity, coeffs_poly, coeffs_legendre, 
                       solver_weighted, coeffs_weighted)


def analyze_lsq2_data():
    """
    Analyze LSQ2.DAT - Higher degree polynomial approximation
    Compare different polynomial degrees and solution methods
    """
    print("\n" + "=" * 60)
    print("ANALYSIS OF LSQ2.DAT - Higher Degree Polynomial")
    print("=" * 60)
    
    # Load data
    x_data, y_data = load_data_file('DATA/LSQ2.DAT')
    print(f"Loaded {len(x_data)} data points from LSQ2.DAT")
    
    solver = LeastSquaresSolver(x_data, y_data)
    
    # Test different polynomial degrees
    degrees = [1, 3, 5, 7, 9]
    results = {}
    
    print(f"\nPolynomial Basis Comparison:")
    print(f"{'Degree':<8} {'RMS Error':<12} {'Condition':<12} {'Method'}")
    print("-" * 50)
    
    for degree in degrees:
        # Gaussian elimination
        coeffs_gauss, info_gauss = solver.fit_polynomial(degree, method='gaussian')
        rms_gauss = solver.calculate_rms_error(coeffs_gauss, 'polynomial')
        
        print(f"{degree:<8} {rms_gauss:<12.6f} {info_gauss['condition_number']:<12.2e} Gaussian")
        
        results[degree] = {
            'gaussian': {'coeffs': coeffs_gauss, 'info': info_gauss, 'rms': rms_gauss},
        }
        
        # Conjugate gradient for higher degrees
        if degree >= 5:
            coeffs_cg, info_cg = solver.fit_polynomial(degree, method='conjugate_gradient')
            rms_cg = solver.calculate_rms_error(coeffs_cg, 'polynomial')
            print(f"{degree:<8} {rms_cg:<12.6f} {info_cg['condition_number']:<12.2e} Conj. Grad.")
            results[degree]['cg'] = {'coeffs': coeffs_cg, 'info': info_cg, 'rms': rms_cg}
    
    # Compare with Legendre basis
    print(f"\nLegendre vs Polynomial Basis (degree 9):")
    coeffs_leg, info_leg = solver.fit_legendre(degree=9, method='gaussian')
    rms_leg = solver.calculate_rms_error(coeffs_leg, 'legendre')
    
    print(f"Legendre:   RMS = {rms_leg:.6f}, Condition = {info_leg['condition_number']:.2e}")
    print(f"Polynomial: RMS = {results[9]['gaussian']['rms']:.6f}, "
          f"Condition = {results[9]['gaussian']['info']['condition_number']:.2e}")
    
    # Visualization
    plot_lsq2_analysis(solver, results, coeffs_leg, info_leg)


def analyze_lsq3_data():
    """
    Analyze LSQ3.DAT - Trigonometric signal recovery
    Find c₁, c₂ for y(x) = c₁ sin(2πx) + c₂ sin(4πx)
    """
    print("\n" + "=" * 60)
    print("ANALYSIS OF LSQ3.DAT - Trigonometric Signal Recovery")
    print("=" * 60)
    
    # Load data
    x_data, y_data = load_data_file('DATA/LSQ3.DAT')
    print(f"Loaded {len(x_data)} data points from LSQ3.DAT")
    print("Expected signal: y(x) = c₁ sin(2πx) + c₂ sin(4πx) + noise")
    
    solver = LeastSquaresSolver(x_data, y_data)
    
    # Method 1: Custom trigonometric basis with specific frequencies
    frequencies = [1.0, 2.0]  # For sin(2πx) and sin(4πx)
    coeffs_custom, info_custom = solver.fit_custom_trigonometric(
        frequencies=frequencies, method='gaussian', include_constant=True
    )
    rms_custom = solver.calculate_rms_error(coeffs_custom, 'custom_trigonometric',
                                          frequencies=frequencies, include_constant=True)
    
    print(f"\nCustom Trigonometric Basis:")
    print(f"  y(x) = c₀ + c₁ sin(2πx) + c₂ sin(4πx)")
    print(f"  Coefficients: c₀ = {coeffs_custom[0]:.6f}")
    print(f"                c₁ = {coeffs_custom[1]:.6f}")
    print(f"                c₂ = {coeffs_custom[2]:.6f}")
    print(f"  RMS Error: {rms_custom:.6f}")
    print(f"  True values: c₁ = 3.0, c₂ = 1.5 (from data generation)")
    
    # Method 2: General trigonometric basis
    coeffs_trig, info_trig = solver.fit_trigonometric(n_terms=3, method='gaussian')
    rms_trig = solver.calculate_rms_error(coeffs_trig, 'trigonometric')
    
    print(f"\nGeneral Trigonometric Basis (3 terms):")
    print(f"  y(x) = c₀ + c₁sin(2πx) + c₂cos(2πx) + c₃sin(4πx) + c₄cos(4πx) + c₅sin(6πx) + c₆cos(6πx)")
    print(f"  Key coefficients: c₀ = {coeffs_trig[0]:.6f}")
    print(f"                    c₁ = {coeffs_trig[1]:.6f} (sin(2πx))")
    print(f"                    c₂ = {coeffs_trig[2]:.6f} (cos(2πx))")
    print(f"                    c₃ = {coeffs_trig[3]:.6f} (sin(4πx))")
    print(f"                    c₄ = {coeffs_trig[4]:.6f} (cos(4πx))")
    print(f"  RMS Error: {rms_trig:.6f}")
    
    # Visualization
    plot_lsq3_analysis(solver, coeffs_custom, coeffs_trig, frequencies)


def analyze_lsq4_data():
    """
    Analyze LSQ4.DAT - Trigonometric polynomial approximation comparison
    Compare trigonometric polynomials of degrees m and m+r
    """
    print("\n" + "=" * 60)
    print("ANALYSIS OF LSQ4.DAT - Trigonometric Polynomial Comparison")
    print("=" * 60)
    
    # Load data
    x_data, y_data = load_data_file('DATA/LSQ4.DAT')
    print(f"Loaded {len(x_data)} data points from LSQ4.DAT")
    
    solver = LeastSquaresSolver(x_data, y_data)
    
    # Choose m and m+r where m+r < N
    N = len(x_data)
    m = 4
    r = 3
    m_plus_r = m + r
    
    print(f"N (data points) = {N}")
    print(f"m = {m}, m+r = {m_plus_r}")
    
    # Fit with degree m
    coeffs_m, info_m = solver.fit_trigonometric(n_terms=m, method='gaussian')
    rms_m = solver.calculate_rms_error(coeffs_m, 'trigonometric')
    
    print(f"\nTrigonometric polynomial (m = {m} terms):")
    print(f"  Number of coefficients: {len(coeffs_m)}")
    print(f"  RMS Error: {rms_m:.6f}")
    print(f"  Condition Number: {info_m['condition_number']:.2e}")
    
    # Fit with degree m+r
    coeffs_mr, info_mr = solver.fit_trigonometric(n_terms=m_plus_r, method='gaussian')
    rms_mr = solver.calculate_rms_error(coeffs_mr, 'trigonometric')
    
    print(f"\nTrigonometric polynomial (m+r = {m_plus_r} terms):")
    print(f"  Number of coefficients: {len(coeffs_mr)}")
    print(f"  RMS Error: {rms_mr:.6f}")
    print(f"  Condition Number: {info_mr['condition_number']:.2e}")
    
    # Compare first m coefficients
    print(f"\nComparison of first {m*2+1} coefficients:")
    print(f"{'Index':<6} {'m terms':<12} {'m+r terms':<12} {'Difference':<12}")
    print("-" * 48)
    
    for i in range(min(len(coeffs_m), len(coeffs_mr))):
        diff = abs(coeffs_m[i] - coeffs_mr[i])
        print(f"{i:<6} {coeffs_m[i]:<12.6f} {coeffs_mr[i]:<12.6f} {diff:<12.6f}")
    
    # Visualization
    plot_lsq4_analysis(solver, coeffs_m, coeffs_mr, m, m_plus_r)


def plot_lsq1_analysis(solver_identity, coeffs_poly, coeffs_legendre, solver_weighted, coeffs_weighted):
    """Plot analysis results for LSQ1.DAT"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate evaluation points
    x_eval = np.linspace(np.min(solver_identity.x_data), np.max(solver_identity.x_data), 100)
    
    # Plot 1: Data and fits
    ax1 = axes[0]
    ax1.scatter(solver_identity.x_data, solver_identity.y_data, color='red', s=60, zorder=5, 
                label='Data points', alpha=0.8)
    
    y_poly = solver_identity.evaluate_polynomial(coeffs_poly, x_eval)
    y_legendre = solver_identity.evaluate_legendre(coeffs_legendre, x_eval)
    y_weighted = solver_weighted.evaluate_polynomial(coeffs_weighted, x_eval)
    
    ax1.plot(x_eval, y_poly, 'b-', linewidth=2, label='Polynomial basis')
    ax1.plot(x_eval, y_legendre, 'g--', linewidth=2, label='Legendre basis')
    ax1.plot(x_eval, y_weighted, 'm:', linewidth=2, label='Weighted polynomial')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('LSQ1.DAT - Linear Approximation Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals comparison
    ax2 = axes[1]
    residuals_poly = solver_identity.calculate_residuals(coeffs_poly, 'polynomial')
    residuals_legendre = solver_identity.calculate_residuals(coeffs_legendre, 'legendre')
    residuals_weighted = solver_weighted.calculate_residuals(coeffs_weighted, 'polynomial')
    
    x_pos = np.arange(len(solver_identity.x_data))
    width = 0.25
    
    ax2.bar(x_pos - width, residuals_poly, width, label='Polynomial', alpha=0.8)
    ax2.bar(x_pos, residuals_legendre, width, label='Legendre', alpha=0.8)
    ax2.bar(x_pos + width, residuals_weighted, width, label='Weighted', alpha=0.8)
    
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residuals Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/LSQ1_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_lsq2_analysis(solver, results, coeffs_legendre, info_legendre):
    """Plot analysis results for LSQ2.DAT"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    x_eval = np.linspace(np.min(solver.x_data), np.max(solver.x_data), 200)
    
    # Plot 1: Different polynomial degrees
    ax1 = axes[0, 0]
    ax1.scatter(solver.x_data, solver.y_data, color='red', s=40, zorder=5, 
                label='Data points', alpha=0.8)
    
    for degree in [1, 3, 5, 9]:
        coeffs = results[degree]['gaussian']['coeffs']
        y_eval = solver.evaluate_polynomial(coeffs, x_eval)
        ax1.plot(x_eval, y_eval, linewidth=2, label=f'Degree {degree}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('LSQ2.DAT - Polynomial Degree Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMS Error vs Degree
    ax2 = axes[0, 1]
    degrees = list(results.keys())
    rms_errors = [results[d]['gaussian']['rms'] for d in degrees]
    condition_numbers = [results[d]['gaussian']['info']['condition_number'] for d in degrees]
    
    ax2.semilogy(degrees, rms_errors, 'bo-', linewidth=2, label='RMS Error')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('RMS Error (log scale)')
    ax2.set_title('Approximation Quality vs Degree')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Condition Number vs Degree
    ax3 = axes[1, 0]
    ax3.semilogy(degrees, condition_numbers, 'ro-', linewidth=2)
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Condition Number (log scale)')
    ax3.set_title('Matrix Conditioning vs Degree')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Polynomial vs Legendre (degree 9)
    ax4 = axes[1, 1]
    ax4.scatter(solver.x_data, solver.y_data, color='red', s=40, zorder=5, 
                label='Data points', alpha=0.8)
    
    y_poly = solver.evaluate_polynomial(results[9]['gaussian']['coeffs'], x_eval)
    y_legendre = solver.evaluate_legendre(coeffs_legendre, x_eval)
    
    ax4.plot(x_eval, y_poly, 'b-', linewidth=2, label='Polynomial basis')
    ax4.plot(x_eval, y_legendre, 'g--', linewidth=2, label='Legendre basis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Polynomial vs Legendre Basis (Degree 9)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/LSQ2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_lsq3_analysis(solver, coeffs_custom, coeffs_trig, frequencies):
    """Plot analysis results for LSQ3.DAT"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    x_eval = np.linspace(0, 1, 200)
    
    # Plot 1: Data and custom trigonometric fit
    ax1 = axes[0, 0]
    ax1.scatter(solver.x_data, solver.y_data, color='red', s=40, zorder=5, 
                label='Noisy data', alpha=0.7)
    
    y_custom = solver.evaluate_custom_trigonometric(coeffs_custom, x_eval, frequencies, True)
    ax1.plot(x_eval, y_custom, 'b-', linewidth=2, label='Custom trig. fit')
    
    # Plot true signal for comparison
    y_true = 3.0 * np.sin(2 * np.pi * x_eval) + 1.5 * np.sin(4 * np.pi * x_eval)
    ax1.plot(x_eval, y_true, 'g--', linewidth=2, label='True signal', alpha=0.8)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('LSQ3.DAT - Signal Recovery')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: General trigonometric fit
    ax2 = axes[0, 1]
    ax2.scatter(solver.x_data, solver.y_data, color='red', s=40, zorder=5, 
                label='Noisy data', alpha=0.7)
    
    y_trig = solver.evaluate_trigonometric(coeffs_trig, x_eval)
    ax2.plot(x_eval, y_trig, 'm-', linewidth=2, label='General trig. fit')
    ax2.plot(x_eval, y_true, 'g--', linewidth=2, label='True signal', alpha=0.8)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('General Trigonometric Basis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient comparison
    ax3 = axes[1, 0]
    labels_custom = ['Constant', 'sin(2πx)', 'sin(4πx)']
    ax3.bar(range(len(coeffs_custom)), coeffs_custom, alpha=0.8, 
            color=['blue', 'orange', 'green'])
    ax3.set_xticks(range(len(coeffs_custom)))
    ax3.set_xticklabels(labels_custom)
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Custom Basis Coefficients')
    ax3.grid(True, alpha=0.3)
    
    # Add true values as horizontal lines
    ax3.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='True c₁=3.0')
    ax3.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='True c₂=1.5')
    ax3.legend()
    
    # Plot 4: Residuals
    ax4 = axes[1, 1]
    residuals_custom = solver.calculate_residuals(coeffs_custom, 'custom_trigonometric',
                                                frequencies=frequencies, include_constant=True)
    residuals_trig = solver.calculate_residuals(coeffs_trig, 'trigonometric')
    
    x_indices = range(len(solver.x_data))
    ax4.scatter(x_indices, residuals_custom, alpha=0.7, label='Custom basis', s=30)
    ax4.scatter(x_indices, residuals_trig, alpha=0.7, label='General basis', s=30)
    ax4.set_xlabel('Data Point Index')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residuals Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/LSQ3_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_lsq4_analysis(solver, coeffs_m, coeffs_mr, m, m_plus_r):
    """Plot analysis results for LSQ4.DAT"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    x_eval = np.linspace(0, 2*np.pi, 300)
    
    # Plot 1: Data and approximations
    ax1 = axes[0, 0]
    ax1.scatter(solver.x_data, solver.y_data, color='red', s=40, zorder=5, 
                label='Data points', alpha=0.8)
    
    y_m = solver.evaluate_trigonometric(coeffs_m, x_eval)
    y_mr = solver.evaluate_trigonometric(coeffs_mr, x_eval)
    
    ax1.plot(x_eval, y_m, 'b-', linewidth=2, label=f'{m} terms')
    ax1.plot(x_eval, y_mr, 'g--', linewidth=2, label=f'{m_plus_r} terms')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('LSQ4.DAT - Trigonometric Approximation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient comparison
    ax2 = axes[0, 1]
    n_common = min(len(coeffs_m), len(coeffs_mr))
    x_pos = np.arange(n_common)
    width = 0.35
    
    ax2.bar(x_pos - width/2, coeffs_m[:n_common], width, label=f'{m} terms', alpha=0.8)
    ax2.bar(x_pos + width/2, coeffs_mr[:n_common], width, label=f'{m_plus_r} terms', alpha=0.8)
    
    ax2.set_xlabel('Coefficient Index')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title(f'First {n_common} Coefficients Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coefficient differences
    ax3 = axes[1, 0]
    differences = np.abs(coeffs_m[:n_common] - coeffs_mr[:n_common])
    ax3.semilogy(x_pos, differences, 'ro-', linewidth=2)
    ax3.set_xlabel('Coefficient Index')
    ax3.set_ylabel('|Difference| (log scale)')
    ax3.set_title('Coefficient Differences')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals comparison
    ax4 = axes[1, 1]
    residuals_m = solver.calculate_residuals(coeffs_m, 'trigonometric')
    residuals_mr = solver.calculate_residuals(coeffs_mr, 'trigonometric')
    
    x_indices = range(len(solver.x_data))
    ax4.scatter(x_indices, residuals_m, alpha=0.7, label=f'{m} terms', s=30)
    ax4.scatter(x_indices, residuals_mr, alpha=0.7, label=f'{m_plus_r} terms', s=30)
    ax4.set_xlabel('Data Point Index')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residuals Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/LSQ4_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table():
    """Create a comprehensive summary table of all analyses"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY - LEAST SQUARES METHOD ANALYSIS")
    print("=" * 80)
    
    # Load all data files and create summary
    data_files = ['LSQ1.DAT', 'LSQ2.DAT', 'LSQ3.DAT', 'LSQ4.DAT']
    
    print(f"{'Dataset':<12} {'Points':<8} {'Best Method':<20} {'RMS Error':<12} {'Notes'}")
    print("-" * 80)
    
    for data_file in data_files:
        try:
            x_data, y_data = load_data_file(f'DATA/{data_file}')
            n_points = len(x_data)
            
            solver = LeastSquaresSolver(x_data, y_data)
            
            if data_file == 'LSQ1.DAT':
                coeffs, _ = solver.fit_polynomial(1)
                rms = solver.calculate_rms_error(coeffs, 'polynomial')
                method = "Linear Polynomial"
                notes = "Linear relationship"
                
            elif data_file == 'LSQ2.DAT':
                coeffs, _ = solver.fit_polynomial(5)
                rms = solver.calculate_rms_error(coeffs, 'polynomial')
                method = "Polynomial deg 5"
                notes = "Higher degree needed"
                
            elif data_file == 'LSQ3.DAT':
                coeffs, _ = solver.fit_custom_trigonometric([1.0, 2.0])
                rms = solver.calculate_rms_error(coeffs, 'custom_trigonometric',
                                               frequencies=[1.0, 2.0], include_constant=True)
                method = "Custom Trigonometric"
                notes = "Signal recovery"
                
            elif data_file == 'LSQ4.DAT':
                coeffs, _ = solver.fit_trigonometric(4)
                rms = solver.calculate_rms_error(coeffs, 'trigonometric')
                method = "Trigonometric"
                notes = "Function table"
            
            print(f"{data_file:<12} {n_points:<8} {method:<20} {rms:<12.6f} {notes}")
            
        except Exception as e:
            print(f"{data_file:<12} {'ERROR':<8} {str(e):<20}")


def main():
    """Main analysis function"""
    print("Least Squares Method - Comprehensive Analysis")
    print("Laboratory Work Week 3")
    print("Computational Mathematics")
    
    # Create graphs directory
    os.makedirs('graphs', exist_ok=True)
    
    # Perform all analyses
    analyze_lsq1_data()
    analyze_lsq2_data() 
    analyze_lsq3_data()
    analyze_lsq4_data()
    
    # Create summary
    create_summary_table()
    
    print(f"\n{'-'*60}")
    print("Analysis completed successfully!")
    print("Generated visualizations saved in 'graphs/' directory:")
    print("- LSQ1_analysis.png (Linear approximation)")
    print("- LSQ2_analysis.png (Polynomial approximation)")
    print("- LSQ3_analysis.png (Signal recovery)")
    print("- LSQ4_analysis.png (Trigonometric comparison)")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()