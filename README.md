# SortOF (Sorting Open Framework)

**A Python library for benchmarking sorting algorithms across different data types.**

## What is SortOF?

SortOF is a lightweight Python library and **API (Application Programming Interface)** designed for benchmarking the performance of various sorting algorithms. It empowers you to easily compare the execution time of different algorithms when sorting various data types in your Python projects:

*   **Numbers:**  Benchmark standard sorting algorithms on numerical arrays.
*   **Words:**  Benchmark sorting algorithms on lists of words, with options for sorting by length or lexicographically (alphabetical order).
*   **Matrices:** Benchmark sorting algorithms on lists of matrices, ordering them based on matrix properties like determinant, trace, or Frobenius norm.

SortOF provides as a **library/API**:

*   **Collection of Sorting Algorithms:** Includes implementations of Bubble Sort, Insertion Sort, Selection Sort, Merge Sort, Quick Sort, Heap Sort, Median Sort, and Radix Sort.
*   **Benchmarking Functionality:**  Provides a simple `benchmark()` function to measure the average execution time of sorting algorithms across different test case sizes.
*   **Result Visualization:** Generates PNG plots of benchmark results to visually compare performance.
*   **Detailed Reporting:** Creates PDF reports summarizing benchmark findings, including algorithm descriptions, input details, and performance charts for documentation or sharing.
*   **Showcase Example (`main.py`):**  Offers a clear and practical example of how to integrate and use the SortOF library/API in your own Python code.

## Key Accomplishments:

*   **Versatile Data Type Support:**  Supports benchmarking for numbers, words, and matrices, making it adaptable to different sorting needs.
*   **Matrix Sorting Capabilities:** Enables sorting of matrices based on key linear algebra metrics (determinant, trace, norm).
*   **Streamlined Benchmarking API:** Simplifies the benchmarking process through the `stf.benchmark()` function.
*   **Comprehensive Reporting API:** Offers automated generation of both visual plots and detailed PDF reports for analysis and documentation.
*   **Modular Library Design:** Built as a reusable Python library (`sortof.py`) with a clear API for integration into other projects, alongside a demonstrative showcase script (`main.py`).

## How to Use (as a Library/API):

1.  **Install Dependencies:** Ensure `numpy`, `matplotlib`, and `reportlab` are installed:
    ```bash
    pip install numpy matplotlib reportlab
    ```
2.  **Import SortOF in your Python Project:**
    ```python
    import sortof as stf  # Import SortOF and alias it as 'stf'
    ```
3.  **Use the `stf.benchmark()` function:** Call `stf.benchmark()` in your code to benchmark sorting algorithms.  Refer to the `main.py` example for detailed usage:

    ```python
    import sortof as stf

    algorithms_to_test = ["bubble", "insertion"]
    test_sizes = [100, 1000]
    generate_reports = True

    for algo in algorithms_to_test:
        stf.benchmark(algorithm=algo, data_type='numbers', test_case_sizes=test_sizes, pdf_report=generate_reports)
    ```

4.  **Run your Python script.**
5.  **Find Reports:** Benchmark plots (PNG) and PDF reports will be generated in the `reports` subdirectory.

**Run the Showcase Example:**

To quickly see SortOF in action, you can run the provided `main.py` showcase script:

```bash
python3 main.py