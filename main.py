# main.py - Showcase example for using the SortOF (Sorting Open Framework) library

# --- Import the SortOF library ---
import sortof as stf  # Import SortOF and alias it as 'stf' for easier use

if __name__ == '__main__':
    print("--- SortOF - Sorting Open Framework Showcase ---")

    # --- 1. Number Benchmarking Example ---
    print("\n--- Number Benchmarking Example ---")
    number_algorithms_example = ["bubble", "insertion", "merge"] # Example algorithms for numbers
    number_test_sizes_example = [100, 500] # Example test sizes
    generate_pdf_report_numbers_example = True # Example PDF report flag

    print("\n--- Number Sorting Benchmarks ---")
    for algo in number_algorithms_example:
        algorithm_name = f"{algo.capitalize()} Sort"
        print(f"\n--- Benchmarking Number Sort: {algorithm_name} ---") # Clear output for each benchmark
        stf.benchmark(  # Call the benchmark function from the imported 'stf' library
            algorithm=algo,
            data_type='numbers',
            test_case_sizes=number_test_sizes_example,
            pdf_report=generate_pdf_report_numbers_example
        )

    print("\n--- Number Benchmarking Example Finished ---")


    # --- 2. Word Benchmarking Example ---
    print("\n--- Word Benchmarking Example ---")
    word_algorithms_example = ["selection", "insertion", "quick"] # Example algorithms for words
    word_methods_example = ["length", "lex"] # Example word sorting methods - CORRECTED to use 'lex' for alphabetical
    word_test_sizes_example = [50, 200] # Example test sizes
    generate_pdf_report_words_example = True # Example PDF report flag

    print("\n--- Word Sorting Benchmarks ---")
    for algo in word_algorithms_example:
        for method in word_methods_example:
            method_display_name = "Lexicographical" if method == "lex" else method.capitalize() # Display 'Lexicographical' for 'lex'
            algorithm_name = f"{algo.capitalize()} Sort by {method_display_name}"
            print(f"\n--- Benchmarking Word Sort: {algorithm_name} ---") # Clear output - Use method_display_name
            stf.benchmark(  # Call benchmark from 'stf' library
                algorithm=algo,
                data_type='words',
                method=method, # Use 'lex' method now
                test_case_sizes=word_test_sizes_example,
                pdf_report=generate_pdf_report_words_example
            )

    print("\n--- Word Benchmarking Example Finished ---")


    # --- 3. Matrix Benchmarking Example ---
    print("\n--- Matrix Benchmarking Example ---")
    matrix_algorithms_example = ["bubble", "merge", "heap"] # Example algorithms for matrices
    matrix_sort_methods_example = ["determinant", "trace", "norm"] # Example matrix sort methods
    matrix_test_sizes_example = [5, 10] # Example test sizes
    generate_pdf_report_matrix_example = True # Example PDF report flag

    print("\n--- Matrix Sorting Benchmarks ---")
    for algo in matrix_algorithms_example:
        for sort_method in matrix_sort_methods_example:
            algorithm_name = f"{algo.capitalize()} Sort by {sort_method.capitalize()}"
            print(f"\n--- Benchmarking Matrix Sort: {algorithm_name} ---") # Clear output
            stf.benchmark(  # Call benchmark from 'stf' library
                algorithm=algo,
                data_type='matrices',
                matrix_sort_method=sort_method,
                test_case_sizes=matrix_test_sizes_example,
                pdf_report=generate_pdf_report_matrix_example
            )

    print("\n--- Matrix Benchmarking Example Finished ---")


    print("\n--- SortOF Showcase - Benchmarking Completed ---")
    print("\n--- Please check the generated PDF reports in the 'reports' directory. ---")