# sortof.py
import time
import random
from sorting_algorithms import *
from report_generator import generate_pdf_report

default_test_case_sizes = [100, 500, 1000, 2000, 5000, 7000, 10000]

def generate_array(data_type, size):
    """Generates an array of a specified data type and size."""
    if data_type == 'numbers':
        return [random.randint(0, 1000) for _ in range(size)]
    elif data_type == 'words':
        return [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5)) for _ in range(size)]

def sort(algorithm, data_type, size=100, order='normal', method='length'):
    """Main sorting function, simplified parameters."""
    arr = generate_array(data_type, size)

    if data_type == 'numbers':
        if algorithm == 'bubble':
            return bubble_sort_num(arr, order=order)
        elif algorithm == 'selection':
            return selection_sort_num(arr, order=order)
        elif algorithm == 'insertion':
            return insertion_sort_num(arr, order=order)
        elif algorithm == 'median':
            return median_sort_num(arr, order=order)
        elif algorithm == 'merge':
            return merge_sort_num(arr, order=order)
        elif algorithm == 'quick':
            return quick_sort_num(arr, order=order)
        elif algorithm == 'heap':
            return heap_sort_num(arr, order=order)
        elif algorithm == 'radix':
            return radix_sort_num(arr)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    elif data_type == 'words':
        if algorithm == 'bubble':
            return bubble_sort_words(arr, order=order, method=method)
        elif algorithm == 'selection':
            return selection_sort_words(arr, order=order, method=method)
        elif algorithm == 'insertion':
            return insertion_sort_words(arr, order=order, method=method)
        elif algorithm == 'median':
            return median_sort_words(arr, order=order, method=method)
        elif algorithm == 'merge':
            return merge_sort_words(arr, order=order, method=method)
        elif algorithm == 'quick':
            return quicksort_words(arr, order=order, method=method)
        elif algorithm == 'heap':
            return heap_sort_words(arr, order=order, method=method)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def benchmark(algorithm, data_type, test_case_sizes=default_test_case_sizes, pdf_report=False, order='normal', method='length'):
    """Benchmarks a SINGLE sorting algorithm and optionally generates a PDF report."""

    valid_algorithms = ["bubble", "insertion", "selection", "merge", "quick", "radix", "heap", "median"] # Valid algorithms
    if algorithm not in valid_algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not recognized. Choose from {', '.join(valid_algorithms)}.")
    if algorithm == 'radix' and data_type == 'words':
        raise ValueError("Radix sort is only applicable to numbers.")
    if algorithm == 'radix' and data_type == 'strings':
        raise ValueError("Radix sort is only applicable to numbers.")

    algorithm_name = algorithm.capitalize() + " Sort"
    test_case_times = []
    all_algorithms_test_case_times = {algorithm_name: []}

    print(f"\n--- {algorithm_name} ---")

    for size in test_case_sizes:
        start_time = time.perf_counter()
        sorted_arr, sort_time, test_case_times_temp = sort(algorithm, data_type, size=size, order=order, method=method) # Directly call sort
        execution_time = time.perf_counter() - start_time
        test_case_times.append(execution_time)

    avg_time = sum(test_case_times) / len(test_case_times)
    print(f"Average Time: {avg_time:.6f} seconds")
    all_algorithms_test_case_times[algorithm_name] = test_case_times

    if pdf_report:
        output_filename = algorithm.replace(" ", "_").lower() + "_" + data_type.lower() + "_report"
        algo_base_name = algorithm.lower().replace(" ", "_")
        report_description = get_algorithm_description(algo_base_name, data_type)
        input_details = {
            "Data Type": data_type.capitalize(),
            "Array Size": "Varying",
            "Test Cases": len(test_case_sizes)
        }
        generate_pdf_report(algorithm_name, avg_time, report_description, input_details, test_case_times, output_filename=output_filename, test_case_sizes=test_case_sizes)

    return all_algorithms_test_case_times