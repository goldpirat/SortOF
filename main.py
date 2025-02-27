import time
import random
import subprocess  # For running latexmk
import os # For cleanup
import fnmatch # For more flexible file matching in cleanup

from sorting_algorithms import bubble_sort_num, insertion_sort_num, merge_sort_num, quick_sort_num, radix_sort_num, selection_sort_num,heap_sort_num  # Import sorting algorithms
from report_generator import generate_pdf_report, generate_benchmark_plot # Import report generation functions


# Descriptions for the algorithms (for reports)
algorithm_descriptions = {
    "Bubble Sort": "Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed, which indicates that the list is sorted. Although simple, Bubble Sort is not efficient for large datasets.",
    "Insertion Sort": "Insertion Sort is a simple sorting algorithm that builds the final sorted array (or list) one item at a time. It is much less efficient on large lists than more advanced algorithms such as quicksort, heapsort, or merge sort.",
    "Merge Sort": "Merge Sort is an efficient, general-purpose, divide-and-conquer, comparison-based sorting algorithm. Most implementations produce a stable sort, which means that the implementation preserves the input order of equal elements in the sorted output.",
    "Quick Sort": "Quicksort is a divide-and-conquer algorithm. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.",
    "Radix Sort": "Radix sort is a non-comparative integer sorting algorithm. It sorts data with integer keys by grouping keys by the individual digits which share the same significant position and value.",
    "Selection Sort": "Selection sort is an in-place comparison sorting algorithm. It has an O(n^2) time complexity, which makes it inefficient on large lists, and generally performs worse than the similar insertion sort."
}


default_test_case_sizes = [100, 500, 1000, 2000, 5000, 7000, 10000] # Default test case sizes


def generate_array(data_type, size):
    """Generates an array of a specified data type and size."""
    if data_type == 'numbers':
        return [random.randint(0, 1000) for _ in range(size)] # Numbers up to 1000 for better radix sort example
    elif data_type == 'strings':
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(size)] # Fixed length strings for simplicity in sorting


def run_and_test_sort(sort_func, algorithm_name, params, report_params): # Accept report_params
    print(f"\n--- {algorithm_name} ---")
    test_case_times = [] # To store execution times for each test case
    test_case_sizes = report_params.get('test_case_sizes') if report_params else None # Get test case sizes for reporting

    if test_case_sizes is None:
        test_case_sizes = default_test_case_sizes # Use default sizes if not in report_params

    for size in test_case_sizes:
        test_array = generate_array('numbers', size) # Generate array of specified size
        start_time = time.perf_counter()

        params_with_arr = params.copy() # Create a copy to avoid modifying original params
        params_with_arr['arr'] = test_array # Add the 'arr' key with the test array
        sorted_arr, avg_time, test_case_times_temp = sort_func(**params_with_arr) # Unpack the modified params


        end_time = time.perf_counter()
        execution_time = end_time - start_time
        test_case_times.append(execution_time) # Store time for this test case


    if test_case_sizes:
        avg_time = sum(test_case_times) / len(test_case_times) # Calculate average time only if test cases were run
    else:
        avg_time = 0 # If no test cases, average time is 0

    print(f"Sorted Array (from final test case): {sorted_arr[:min(len(sorted_arr), 200)]}") # Print sorted array head

    print(f"Average Time: {avg_time:.6f} seconds")


    if report_params and report_params.get('pdf_report'): # Generate PDF report if pdf_report flag is True
        output_filename = algorithm_name.replace(" ", "_").lower() + "_" + report_params.get('data_type', 'unknown').lower() + "_report" # Construct filename
        report_description = algorithm_descriptions.get(algorithm_name, "Description not available.") # Get description, default if missing
        input_details = { # Collect input details for report
            "Data Type": report_params.get('data_type', 'Numbers').capitalize(),
            "Array Size": "Varying", # Or get specific size if needed
            "Test Cases": len(test_case_sizes) if test_case_sizes else 1 # Indicate number of test cases
        }

        generate_pdf_report(algorithm_name, avg_time, report_description, input_details, test_case_times, output_filename=output_filename, test_case_sizes=test_case_sizes) # Pass test_case_sizes for plotting

    return sorted_arr, avg_time, test_case_times # Return execution times for test cases



def main():
    algorithms_to_run = { # Dictionary of algorithms to run, easily extendable
        "Bubble Sort": bubble_sort_num,
        "Insertion Sort": insertion_sort_num,
        "Selection Sort": selection_sort_num,
        "Merge Sort": merge_sort_num,
        "Quick Sort": quick_sort_num,
        "Radix Sort": radix_sort_num, # Radix Sort for numbers
        "Heap Sort": heap_sort_num, # ADD HEAP SORT HERE for numbers
        #"Radix Sort Strings": radix_sort_str # Example Radix Sort for strings - not yet implemented for strings, just numbers
    }

    import argparse # Import for command-line arguments

    parser = argparse.ArgumentParser(description="Benchmark sorting algorithms and generate reports.")

    parser.add_argument('--algorithm', type=str, default='all', help='Algorithm to run (all, bubble, insertion, selection, merge, quick, radix). Default is all.')
    parser.add_argument('--data_type', type=str, default='numbers', choices=['numbers', 'strings'], help='Data type to sort (numbers, strings). Default is numbers.')
    parser.add_argument('--pdf', action='store_true', help='Generate PDF reports for benchmarks.') # Flag for PDF report
    parser.add_argument('--test_sizes', nargs='+', type=int, default=default_test_case_sizes, help='List of test array sizes (e.g., --test_sizes 100 1000 5000).') # Option to override test sizes


    args = parser.parse_args()

    algorithms_to_benchmark = []
    if args.algorithm == 'all':
        algorithms_to_benchmark = algorithms_to_run
    elif args.algorithm in algorithms_to_run:
        algorithms_to_benchmark = {args.algorithm: algorithms_to_run[args.algorithm]}
    else:
        print(f"Warning: Algorithm '{args.algorithm}' not recognized. Running all algorithms.")
        algorithms_to_benchmark = algorithms_to_run


    common_report_params = { # Common parameters for reporting, now including data_type and test_case_sizes
        'pdf_report': args.pdf,
        'data_type': args.data_type,
        'test_case_sizes': args.test_sizes if args.test_sizes != default_test_case_sizes else default_test_case_sizes # Use command line sizes, or defaults
    }


    # Parameters for each sorting algorithm (can be extended)
    params_bubble = {} # Bubble Sort takes array 'arr'
    params_insertion = {} # Insertion Sort takes array 'arr'
    params_selection = {} # Selection Sort takes array 'arr'
    params_merge = {} # Merge Sort takes array 'arr'
    params_quick = {} # QuickSort takes array 'arr'
    params_radix = {} # Radix Sort for numbers takes array 'arr'


    report_params = common_report_params.copy() # Create a copy of common report params for each algorithm run


    for name, func in algorithms_to_benchmark.items(): # Iterate through selected algorithms
        if "Radix Sort" in name: # Special case for Radix Sort (numbers only for now)
             if args.data_type == 'numbers':
                run_and_test_sort(algorithms_to_run['Radix Sort'], "Radix Sort - Number Array", params_radix, report_params=common_report_params.copy()) # Use base params for Radix Sort, but pass report params separately
             else:
                print(f"Radix Sort is only implemented for 'numbers' data type. Skipping Radix Sort for '{args.data_type}'.")

        elif name in ["Bubble Sort", "Insertion Sort", "Selection Sort", "Merge Sort", "Quick Sort"]: # Handle other sorts for both numbers and strings if needed
            if args.data_type == 'numbers':
                test_params = {} # Default params are empty dictionaries, 'arr' is added in run_and_test_sort
                run_and_test_sort(func, f"{name} - Number Array", test_params, report_params=report_params) # Pass report_params


            elif args.data_type == 'strings':
                print(f"{name} for strings not yet fully implemented, running for numbers instead.") # Placeholder for future string sort implementation

                test_params = {} #  Placeholders, adjust if you create string versions
                run_and_test_sort(func, f"{name} - Number Array", test_params, report_params=report_params) # Still run for numbers as fallback, adjust as needed


if __name__ == "__main__":
    main()