import numpy as np
import os

def generate_lsq1_data():
    """
    Generate LSQ1.DAT - Linear relationship data
    Points that approximately follow y = b0 + b1*x (linear relationship)
    5 points as mentioned in the task
    """
    # True parameters for linear relationship y = b0 + b1*x
    b0_true, b1_true = 2.5, 1.8
    
    x_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    
    np.random.seed(42)
    y_values = b0_true + b1_true * x_values + np.random.normal(0, 0.1, len(x_values))
    
    # Save to file
    data = np.column_stack((x_values, y_values))
    np.savetxt('DATA/LSQ1.DAT', data, fmt='%.6f', 
               header='# Linear relationship data: y ≈ b0 + b1*x\n# x_k, y_k values', 
               comments='')
    
    print(f"Generated LSQ1.DAT with {len(x_values)} points")
    print(f"True parameters: b0 = {b0_true}, b1 = {b1_true}")

def generate_lsq2_data():
    """
    Generate LSQ2.DAT - Higher degree polynomial data
    More complex relationship requiring polynomial of higher degree
    """
    # True polynomial coefficients for y = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    true_coeffs = [1.0, -2.0, 1.5, -0.3, 0.05]
    
    x_values = np.linspace(-2, 3, 15)
    y_true = np.polyval(true_coeffs[::-1], x_values)
    
    np.random.seed(123)
    noise_level = 0.3
    y_values = y_true + np.random.normal(0, noise_level, len(x_values))
    
    data = np.column_stack((x_values, y_values))
    np.savetxt('DATA/LSQ2.DAT', data, fmt='%.6f', 
               header='# Higher degree polynomial data\n# Requires polynomial of degree ~4-5 for good approximation\n# x_k, y_k values', 
               comments='')
    
    print(f"Generated LSQ2.DAT with {len(x_values)} points")
    print(f"True polynomial degree: {len(true_coeffs) - 1}")

def generate_lsq3_data():
    """
    Generate LSQ3.DAT - Trigonometric data
    Noisy signal: y(x) = c1*sin(2πx) + c2*sin(4πx)
    """
    # True coefficients
    c1_true, c2_true = 3.0, 1.5
    
    x_values = np.linspace(0, 1, 20)
    y_true = c1_true * np.sin(2 * np.pi * x_values) + c2_true * np.sin(4 * np.pi * x_values)
    
    np.random.seed(456)
    noise_level = 0.2
    y_values = y_true + np.random.normal(0, noise_level, len(x_values))
    
    data = np.column_stack((x_values, y_values))
    np.savetxt('DATA/LSQ3.DAT', data, fmt='%.6f', 
               header='# Trigonometric signal data: y = c1*sin(2πx) + c2*sin(4πx) + noise\n# True coefficients: c1=3.0, c2=1.5\n# x_k, y_k values', 
               comments='')
    
    print(f"Generated LSQ3.DAT with {len(x_values)} points")
    print(f"True coefficients: c1 = {c1_true}, c2 = {c2_true}")

def generate_lsq4_data():
    """
    Generate LSQ4.DAT - Function table for trigonometric approximation comparison
    Dense sampling of a smooth function
    """
    # Generate a smooth function that can be approximated by trigonometric polynomials
    x_values = np.linspace(0, 2*np.pi, 32)  # 32 points for good FFT properties
    
    # Combination of different frequency components
    y_values = (2.0 * np.cos(x_values) + 
                1.5 * np.sin(2*x_values) + 
                0.8 * np.cos(3*x_values) + 
                0.4 * np.sin(4*x_values))
    
    np.random.seed(789)
    y_values += np.random.normal(0, 0.05, len(x_values))
    
    data = np.column_stack((x_values, y_values))
    np.savetxt('DATA/LSQ4.DAT', data, fmt='%.6f', 
               header='# Function table for trigonometric polynomial approximation\n# Smooth function with multiple harmonic components\n# x_k, y_k values', 
               comments='')
    
    print(f"Generated LSQ4.DAT with {len(x_values)} points")

def main():
    """Generate all data files"""
    print("Generating data files for Least Squares Method Lab...")
    
    os.makedirs('DATA', exist_ok=True)
    
    #os.chdir('/Users/oleg/Desktop/Python/computational_mathematics/lab_week_3')
    
    generate_lsq1_data()
    generate_lsq2_data()
    generate_lsq3_data()
    generate_lsq4_data()
    
    print("\nAll data files generated successfully!")
    print("Files created:")
    print("- DATA/LSQ1.DAT (linear relationship)")
    print("- DATA/LSQ2.DAT (polynomial relationship)")
    print("- DATA/LSQ3.DAT (trigonometric signal)")
    print("- DATA/LSQ4.DAT (function table)")

if __name__ == "__main__":
    main()