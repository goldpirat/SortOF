# Sorting Algorithm Benchmark

This Python script benchmarks the performance of various sorting algorithms and generates a PDF report summarizing the results. It supports benchmarking with different data types (numbers and words) and allows you to generate performance plots and detailed reports in PDF format using LaTeX.

## Features

*   **Benchmarks Multiple Sorting Algorithms:** Includes implementations of Bubble Sort, Insertion Sort, Selection Sort, Median Sort (QuickSort with median-of-three pivot), Merge Sort, Quick Sort (random pivot), Heap Sort, and Radix Sort (for numbers only).
*   **Data Type Flexibility:**  Benchmarks can be run on arrays of numbers or words. For word sorting, you can choose to sort by word length or lexicographically.
*   **Performance Plotting:** Generates benchmark plots showing execution time versus input size for each algorithm using Matplotlib.
*   **PDF Reporting:** Creates comprehensive PDF reports (using LaTeX) that include:
    *   Algorithm descriptions
    *   General performance metrics (average execution time)
    *   Input details (data type, array size, test cases)
    *   Benchmark plots
*   **Customizable Test Cases:** Allows you to specify custom test case sizes via command-line arguments.

## Requirements

*   **Python 3.x**
*   **LaTeX Installation:**  To generate PDF reports, you need to have a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX).
*   **Python Packages:**  Install the necessary Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    See the `requirements.txt` file for the list of packages.

## How to Use

1.  **Clone the repository** (if you haven't already).
2.  **Install Python package dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run `main.py`** with the desired arguments.

    **Basic Command:**

    ```bash
    python3 main.py
    ```

    This will run benchmarks for all sorting algorithms on number arrays using default test case sizes and will **not** generate a PDF report.

    **Generating PDF Reports:**

    To generate a PDF report for the benchmark, use the `--pdf` flag:

    ```bash
    python3 main.py --pdf
    ```

    This will generate PDF reports for all algorithms benchmarked.

    **Specifying Algorithms:**

    To benchmark specific algorithms (instead of all), use the `--algorithm` argument followed by the algorithm name. You can choose from: `bubble_sort`, `insertion_sort`, `selection_sort`, `median_sort`, `merge_sort`, `quick_sort`, `heap_sort`, `radix_sort` or `all` for all algorithms.

    ```bash
    python3 main.py --algorithm "Merge Sort" --pdf
    python3 main.py --algorithm quick_sort --pdf
    python3 main.py --algorithm all --pdf
    ```

    **Specifying Data Type:**

    Use the `--data_type` argument to specify the data type for sorting. Choose between `numbers` (default) and `strings`.

    ```bash
    python3 main.py --data_type numbers --pdf
    python3 main.py --data_type strings --pdf
    ```

    **Custom Test Case Sizes:**

    Use the `--test_sizes` argument followed by a space-separated list of integer sizes to define custom test cases.

    ```bash
    python3 main.py --test_sizes 100 500 1000 --pdf
    python3 main.py --test_sizes 5000 10000 --algorithm "Quick Sort" --pdf
    ```

    **Example Command to run all algorithms on numbers and generate PDF reports with custom test sizes:**

    ```bash
    python3 main.py --algorithm all --data_type numbers --pdf --test_sizes 100 1000 10000
    ```

    **Word Sorting Options (for `strings` data type):**

    When using `--data_type strings`, the script defaults to sorting words lexicographically.  Word length sorting is not implemented in this version.

## Script Files

*   `main.py`:  The main script to run benchmarks and generate reports.
*   `sorting_algorithms.py`:  Contains implementations of various sorting algorithms for numbers and words.
*   `report_generator.py`:  Handles the generation of benchmark plots and PDF reports using LaTeX (`pylatex`).
*   `default_config.py`:  Configuration file (e.g., default test case sizes, algorithm list).
*   `requirements.txt`:  Lists Python package dependencies.
*   `README.md`: This file, providing documentation for the project.

## Notes

*   Ensure you have LaTeX properly installed and configured for PDF report generation.
*   Benchmark results can vary based on your system's hardware and software environment.
*   The `Radix Sort` algorithm is only implemented for number arrays in this version.
*   Word sorting for `strings` data type is currently only implemented for lexicographical sorting.