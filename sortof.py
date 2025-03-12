# sortof.py
import time
import random
import numpy as np
from matrix_operations import matrix_determinant, matrix_frobenius_norm, matrix_trace
from sorting_algorithms import ( # Explicitly import ONLY the functions you need
    bubble_sort_num, selection_sort_num, insertion_sort_num, median_sort_num,
    merge_sort_num, quick_sort_num, heap_sort_num, radix_sort_num,
    bubble_sort_words, selection_sort_words, insertion_sort_words, median_sort_words,
    merge_sort_words, heap_sort_words, quicksort_words,
    sort_matrix_by_determinant, sort_matrix_by_trace, sort_matrix_by_norm, get_algorithm_description
)
from report_generator import generate_pdf_report


default_test_case_sizes = [100, 500, 1000, 2000, 5000, 7000, 10000]
valid_algorithms = ["bubble", "insertion", "selection", "merge", "quick", "radix", "heap", "median"]
valid_matrix_criteria = ["determinant", "trace", "norm"] # Valid criteria for matrix sorting
supported_data_types = ["numbers", "words", "matrices"] # List of supported data types
supported_algorithms = valid_algorithms + valid_matrix_criteria # Combine all supported algorithm/criteria names for help messages


def generate_array(data_type, size):
    """Generates an array of a specified data type and size."""
    if data_type == 'numbers':
        return [random.randint(0, 1000) for _ in range(size)]
    elif data_type == 'words':
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(size)]

def generate_matrix(data_type='numbers', size=10, matrix_shape='square', number_range=(0, 100)):
    """Generates a square matrix of numbers or words.

    Args:
        data_type (str, optional): 'numbers' or 'words'. Defaults to 'numbers'.
        size (int, optional): Size of the matrix (size x size for square). Defaults to 10.
        matrix_shape (str, optional): 'square'.  (Future: 'rectangular'?). Defaults to 'square'.
        number_range (tuple, optional): Range of random integers (min, max) if data_type='numbers'. Defaults to (0, 100).

    Returns:
        list of lists: A square matrix.
    """
    if matrix_shape != 'square':
        raise ValueError("Only 'square' matrices are currently supported.")

    if data_type == 'numbers':
        min_val, max_val = number_range
        return [[random.randint(min_val, max_val) for _ in range(size)] for _ in range(size)]
    elif data_type == 'words': # For future expansion, currently only numbers are requested.
        raise NotImplementedError("Word matrices are not yet implemented.") # Placeholder for words matrices if needed later
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'numbers' for matrices currently.")


def sort(algorithm, data_type, size=100, order='normal', method='length', matrix_sort_method=None):
    """Main sorting function, with matrix sorting using Python's built-in sorted()."""
    sort_time = 0.0
    test_case_times_temp = [0.0]

    if data_type == 'numbers':
        arr = generate_array(data_type, size)
        if algorithm == 'bubble':
            sorted_arr, sort_time_val, test_case_times_temp_val = bubble_sort_num(arr, order=order)
        elif algorithm == 'selection':
            sorted_arr, sort_time_val, test_case_times_temp_val = selection_sort_num(arr, order=order)
        elif algorithm == 'insertion':
            sorted_arr, sort_time_val, test_case_times_temp_val = insertion_sort_num(arr, order=order)
        elif algorithm == 'median':
            sorted_arr, sort_time_val, test_case_times_temp_val = median_sort_num(arr, order=order)
        elif algorithm == 'merge':
            sorted_arr, sort_time_val, test_case_times_temp_val = merge_sort_num(arr, order=order)
        elif algorithm == 'quick':
            sorted_arr, sort_time_val, test_case_times_temp_val = quick_sort_num(arr, order=order)
        elif algorithm == 'heap':
            sorted_arr, sort_time_val, test_case_times_temp_val = heap_sort_num(arr, order=order)
        elif algorithm == 'radix':
            sorted_arr, sort_time_val, test_case_times_temp_val = radix_sort_num(arr)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm} for data type: {data_type}")
        return sorted_arr, sort_time_val, test_case_times_temp_val


    elif data_type == 'words':
        arr = generate_array(data_type, size)
        if algorithm == 'bubble':
            sorted_arr = bubble_sort_words(arr, order=order, method=method)
        elif algorithm == 'selection':
            sorted_arr = selection_sort_words(arr, order=order, method=method)
        elif algorithm == 'insertion':
            sorted_arr = insertion_sort_words(arr, order=order, method=method)
        elif algorithm == 'median':
            sorted_arr = median_sort_words(arr, order=order, method=method)
        elif algorithm == 'merge':
            sorted_arr = merge_sort_words(arr, order=order, method=method)
        elif algorithm == 'quick':
            sorted_arr = quicksort_words(arr, order=order, method=method)
        elif algorithm == 'heap':
            sorted_arr = heap_sort_words(arr, order=order, method=method)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm} for data type: {data_type}")
        return sorted_arr, sort_time, test_case_times_temp


    elif data_type == 'matrices': # --- MODIFIED: Matrix Data Type Handling - Using sorted() ---
        if not matrix_sort_method:
            raise ValueError("Must specify 'matrix_sort_method' (determinant, trace, or norm) when data_type is 'matrices'.")

        matrices = [generate_matrix(data_type='numbers', size=size) for _ in range(size)]
        start_time_sort = time.perf_counter()

        def get_matrix_metric(matrix): # Key function remains the same
            np_matrix = np.array(matrix)
            if matrix_sort_method == 'determinant':
                return matrix_determinant(np_matrix)
            elif matrix_sort_method == 'trace':
                return matrix_trace(np_matrix)
            elif matrix_sort_method == 'norm':
                return matrix_frobenius_norm(np_matrix)
            else:
                raise ValueError(f"Unknown matrix_sort_method: {matrix_sort_method}. Must be 'determinant', 'trace', or 'norm'.")


        if algorithm in ["bubble", "insertion", "selection", "merge", "quick", "heap", "median"]: # Valid algorithms for matrices - using built-in sorted()
            reverse_order = (order == 'reverse') # Determine reverse order for sorted()
            sorted_matrices = sorted(matrices, key=get_matrix_metric, reverse=reverse_order) # Use Python's sorted() with key
            sort_time_val = time.perf_counter() - start_time_sort # Capture sort time here for matrices
            test_case_times_temp_val = [sort_time_val] # Capture test case time here for matrices
            return sorted_matrices, sort_time_val, test_case_times_temp_val # Return with sort_time_val and test_case_times_temp_val

        elif algorithm == 'radix': # Radix sort is not applicable
            raise ValueError("Radix sort is not applicable for sorting matrices based on metrics.")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm} for data type: matrices. Choose from: bubble, insertion, selection, merge, quick, heap, median.")


    elif callable(algorithm):
        start_time_sort = time.perf_counter()
        sorted_arr = algorithm(arr, order=order, method=method)
        sort_time = time.perf_counter() - start_time_sort
        test_case_times_temp = [sort_time]
        return sorted_arr, sort_time, test_case_times_temp

    else:
        raise ValueError(f"Unknown data type: {data_type}. Must be 'numbers', 'words', or 'matrices'.")

def benchmark(algorithm, data_type, test_case_sizes=default_test_case_sizes, pdf_report=False, order='normal', method='length', matrix_sort_method=None):
    """Benchmarks a SINGLE sorting algorithm and optionally generates a PDF report.

    The `algorithm` parameter is now the SORTING ALGORITHM NAME ('bubble', 'insertion', etc.) or a user function.
    For matrices, you also need to specify 'matrix_sort_method' ('determinant', 'trace', 'norm').
    """

    print(f"DEBUG: Entering benchmark function. Algorithm parameter: {algorithm}, data_type: {data_type}, matrix_sort_method: {matrix_sort_method}, type: {type(algorithm)}") # DEBUG PRINT - ENTRY POINT - added matrix_sort_method

    if isinstance(algorithm, str): # Check if algorithm is a string (algorithm name)
        print("DEBUG: Algorithm is detected as a STRING (algorithm name).") # DEBUG PRINT - STRING PATH
        algo_base_name = algorithm # Initialize algo_base_name


        if data_type == 'matrices': # --- MODIFIED: Matrix Data Type Handling ---
            valid_matrix_sort_methods = ["determinant", "trace", "norm"] # Valid matrix sorting methods (metrics)
            valid_matrix_algorithms = ["bubble", "insertion", "selection", "merge", "quick", "heap", "median"] # Valid algorithms for matrices

            if matrix_sort_method is None: # Ensure matrix_sort_method is provided
                raise ValueError("For 'matrices' data_type, you must specify 'matrix_sort_method' (determinant, trace, or norm).")
            if matrix_sort_method not in valid_matrix_sort_methods: # Validate matrix_sort_method
                raise ValueError(f"Invalid matrix_sort_method: '{matrix_sort_method}'. Choose from {', '.join(valid_matrix_sort_methods)} (determinant, trace, norm).")
            if algorithm not in valid_matrix_algorithms: # Validate algorithm for matrices
                raise ValueError(f"Algorithm '{algorithm}' not recognized for matrices. Choose from {', '.join(valid_matrix_algorithms)} (bubble, insertion, selection, merge, quick, heap, median).")


            algorithm_name = f"{algorithm.capitalize()} Sort by {matrix_sort_method.capitalize()}" # Algorithm name: e.g., "Quick Sort by Determinant"
            print(f"DEBUG: Matrix sorting: Algorithm='{algorithm}', Sort Method='{matrix_sort_method}'. Algorithm Name: '{algorithm_name}', base name: '{algo_base_name}'") # DEBUG PRINT - MATRIX ALGO AND METHOD

        
        elif data_type in ['numbers', 'words']: # Existing number/word algorithms (no significant changes)
            if algorithm not in valid_algorithms:
                raise ValueError(f"Algorithm '{algorithm}' not recognized for {data_type}. Choose from {', '.join(valid_algorithms)}.")
            algorithm_name = algorithm.capitalize() + " Sort"
            print(f"DEBUG: Algorithm name set to: '{algorithm_name}', base name: '{algo_base_name}'") # DEBUG PRINT - ALGO NAMES
        else:
            raise ValueError(f"Invalid data type: {data_type}. Must be 'numbers', 'words', or 'matrices'.") # Invalid data type error

        user_provided_description = None # No user description for string algorithms


    elif callable(algorithm): # User-defined function (no changes)
        print("DEBUG: Algorithm is detected as CALLABLE (function).") # DEBUG PRINT - CALLABLE PATH
        algorithm_name = "User-Defined Sort Function"
        algo_base_name = "user_defined"
        user_provided_description = algorithm.__doc__
        print(f"DEBUG: User-defined function detected. Algorithm name set to: '{algorithm_name}', base name: '{algo_base_name}', description from docstring: {bool(user_provided_description)}") # DEBUG PRINT - USER FUNCTION INFO
    else:
        print("DEBUG: Algorithm is of UNEXPECTED TYPE!") # DEBUG PRINT - UNEXPECTED TYPE
        raise TypeError("Algorithm parameter must be either an algorithm name (string) or a sorting function (callable).")


    if algo_base_name == 'radix' and data_type == 'words': # Radix sort check (no changes)
        raise ValueError("Radix sort is only applicable to numbers.")
    if algo_base_name == 'radix' and data_type == 'strings': # Radix sort check (no changes)
        raise ValueError("Radix sort is only applicable to numbers.")


    test_case_times = []
    all_algorithms_test_case_times = {algorithm_name: []}

    print(f"\n--- {algorithm_name} ---") # DEBUG PRINT - ALGO NAME BEFORE BENCHMARKING

    for size in test_case_sizes:
        start_time = time.perf_counter()
        sorted_arr, sort_time, test_case_times_temp = sort(algorithm, data_type, size=size, order=order, method=method, matrix_sort_method=matrix_sort_method) # MODIFIED: Pass matrix_sort_method to sort()
        execution_time = time.perf_counter() - start_time
        test_case_times.append(execution_time)
    
    avg_time = sum(test_case_times) / len(test_case_sizes)
    print(f"Average Time: {avg_time:.6f} seconds")
    all_algorithms_test_case_times[algorithm_name] = test_case_times


    report_description = ""
    if user_provided_description: # User-provided description (no changes)
        report_description = user_provided_description
        print("Using user-provided algorithm description from docstring.")
    else:
        try: # Get description from get_algorithm_description - MODIFIED to pass matrix_sort_method
            report_description = get_algorithm_description(algo_base_name, data_type, method=method, matrix_sort_method=matrix_sort_method) # Pass matrix_sort_method
        except NameError:
            pass
    

    if pdf_report: # PDF report generation
        if data_type == 'matrices': # --- MODIFIED FILENAME for Matrices ---
            output_filename = algo_base_name.replace(" ", "_").lower() + "_by_" + matrix_sort_method.lower() + "_matrices_report"
            algo_base_name_for_report = algo_base_name.lower().replace(" ", "_") + "_by_" + matrix_sort_method.lower() # Also update base name for report to include sort method
        else: # Existing filename logic for numbers and words (no change)
            output_filename = algo_base_name.replace(" ", "_").lower() + "_" + data_type.lower() + "_report"
            algo_base_name_for_report = algo_base_name.lower().replace(" ", "_")


        if data_type == 'matrices': # --- DEBUG PRINTS (Keep these for now) ---
            print("DEBUG (PDF): Matrix Report - Algorithm Name:", algorithm_name)
            print("DEBUG (PDF): Matrix Report - Base Name:", algo_base_name)
            print("DEBUG (PDF): Matrix Report - Base Name for Report:", algo_base_name_for_report) # MODIFIED: check if this is updated correctly
            print("DEBUG (PDF): Matrix Report - Output Filename:", output_filename) # MODIFIED: Check if filename now includes matrix_sort_method
            print("DEBUG (PDF): Matrix Report - Matrix Sort Method:", matrix_sort_method)


        input_details = {
            "Algorithm": algorithm_name,
            "Data Type": data_type.capitalize(),
            "Array Size": "Varying",
            "Test Cases": len(test_case_sizes)
        }
        generate_pdf_report(algorithm_name, avg_time, report_description, input_details, test_case_times, output_filename=output_filename, test_case_sizes=test_case_sizes)
        
    print(f"DEBUG: Algorithm Name before final print: '{algorithm_name}'") # DEBUG PRINT - BEFORE FINAL PRINT STATEMENT
    print(f"Benchmark completed for {algorithm_name} on {data_type}.")
    return all_algorithms_test_case_times, algorithm_name

if __name__ == '__main__':
    # Example Usage and Test Prints (You can keep this at the end of sortof.py for testing)
    print("--- Number Array ---")
    number_array = generate_array('numbers', 5)
    print(number_array)

    print("\n--- Word Array ---")
    word_array = generate_array('words', 5)
    print(word_array)

    print("\n--- Number Matrix (5x5) ---")
    number_matrix = generate_matrix('numbers', 5, number_range=(-10, 10)) # Example with range -10 to 10
    for row in number_matrix:
        print(row)