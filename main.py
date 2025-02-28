# main.py
import sortof as stf

def main():
    # --- BENCHMARK CONFIGURATION ---
    algorithm_to_benchmark = 'bubble'   # Choose algorithm: bubble, insertion, selection, merge, quick, radix, heap
    data_type_to_sort = 'words'     # Choose data type: numbers, words
    generate_pdf_report = True        # True to generate PDF report, False otherwise
    custom_test_sizes = [100, 1000, 5000] # Set custom test sizes, or use default: None

    sort_order = 'normal'             # For algorithms that support order: 'normal', 'reverse'
    word_comparison_method = 'lex' # For word sorting: 'length', 'lex'

    # --- RUN BENCHMARK ---
    try:
        stf.benchmark(algorithm=algorithm_to_benchmark,
                      data_type=data_type_to_sort,
                      pdf_report=generate_pdf_report,
                      test_case_sizes=custom_test_sizes,
                      order=sort_order,              # Pass order parameter
                      method=word_comparison_method) # Pass method parameter (for words)

        print(f"Benchmark completed for {algorithm_to_benchmark} sort on {data_type_to_sort}.")
        if generate_pdf_report:
            print("PDF report generated.")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()