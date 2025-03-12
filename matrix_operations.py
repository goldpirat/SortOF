import numpy as np

def matrix_determinant(matrix):
    """Calculates the determinant of a square matrix using numpy."""
    return np.linalg.det(matrix)

def matrix_trace(matrix):
    """Calculates the trace of a square matrix (sum of diagonal elements) using numpy."""
    return np.trace(matrix)

def matrix_frobenius_norm(matrix):
    """Calculates the Frobenius norm of a matrix using numpy."""
    return np.linalg.norm(matrix, 'fro') # 'fro' specifies Frobenius norm

if __name__ == '__main__':
    # Example Usage and Test Prints
    example_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    np_matrix = np.array(example_matrix) # Convert to numpy array for linear algebra functions

    determinant = matrix_determinant(np_matrix)
    trace = matrix_trace(np_matrix)
    frobenius_norm = matrix_frobenius_norm(np_matrix)

    print("--- Example Matrix ---")
    for row in example_matrix:
        print(row)

    print(f"\nDeterminant: {determinant}")
    print(f"Trace: {trace}")
    print(f"Frobenius Norm: {frobenius_norm}")
