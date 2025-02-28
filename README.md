# SortOF: Your Python Sorting Algorithm Benchmarking Library

[![Python Version](https://www.python.org/)](https://www.python.org/)
[![License](https://opensource.org/licenses/MIT)](https://opensource.org/licenses/MIT)

**SortOF** is a Python library designed to help you explore and benchmark the performance of various sorting algorithms **programmatically**.  It provides functions to easily benchmark sorting algorithms within your Python code and generate insightful reports.  While a basic script (`main.py`) is included for demonstration and quick tests, the primary focus is on using SortOF as a library in your own Python projects.

## Features

*   **Benchmark Multiple Sorting Algorithms:** Programmatically compare the execution time of popular sorting algorithms like Bubble Sort, Insertion Sort, Selection Sort, Merge Sort, Quick Sort, Heap Sort, Median Sort (QuickSort variant), and Radix Sort (for numbers) within your Python code.
*   **Support for Different Data Types:** Benchmark sorting with both numerical data and words.
*   **Generate Detailed PDF Reports (Optional):**  Programmatically create comprehensive PDF reports summarizing benchmark results, including average execution times, algorithm descriptions, and input details.
*   **Easy-to-Use Programmatic Interface:**  Designed to be imported and used directly within your Python scripts for flexible benchmarking and analysis.
*   **Customizable Test Sizes:**  Define your own array sizes for benchmarking programmatically to suit your specific needs.
*   **Clear and Concise Output:**  Get immediate feedback on benchmark performance printed to your console during programmatic execution.

## Sorting Algorithms Included

*   **Bubble Sort:** Simple comparison-based algorithm.
*   **Insertion Sort:** Efficient for small datasets and nearly sorted data.
*   **Selection Sort:**  Another simple algorithm, known for its minimal number of swaps.
*   **Merge Sort:**  Efficient and stable, based on the divide-and-conquer approach.
*   **Quick Sort:**  Generally very fast for average cases (Median Sort variant included for improved pivot selection).
*   **Heap Sort:**  Efficient sorting algorithm using a binary heap data structure.
*   **Median Sort:** A variation of Quick Sort using the median-of-three pivot strategy.
*   **Radix Sort:**  A fast, non-comparison-based algorithm efficient for sorting integers.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   Libraries listed in `requirements.txt` (install using pip, see Installation below)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository-url] # Replace with your repository URL
    cd sortof
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Programmatic Use (Recommended - Library Usage)

SortOF is designed to be used as a library within your Python scripts. Here's how to benchmark sorting algorithms programmatically:

```python
import sortof as stf

def analyze_sorting_performance():
    # Benchmark Quick Sort for numbers, generate PDF report
    stf.benchmark(algorithm='quick', data_type='numbers', pdf_report=True)

    # Benchmark Bubble Sort for words, different test sizes, no PDF report
    stf.benchmark(algorithm='bubble', data_type='words', test_case_sizes=[50, 250, 500], pdf_report=False)

    # Run Merge Sort on numbers and get results directly without benchmarking
    sorted_array, sort_time, _ = stf.sort('merge', 'numbers', size=200, order='reverse')
    print(f"\nMerge Sort (Size 200, Reverse Order) - Sorted array (first 10 elements): {sorted_array[:10]}")
    print(f"Merge Sort (Size 200, Reverse Order) - Sorting time: {sort_time:.6f} seconds.")


if __name__ == "__main__":
    analyze_sorting_performance()
```

  * **Example Script (`main.py`):** The `main.py` file in the repository provides a basic example of how to use the `sortof` library programmatically. You can modify it or create your own Python scripts to perform various benchmarks.  **It is not intended to be the primary interface for the library, but rather a demonstration and a quick way to run pre-configured benchmarks.**

#### (De-emphasized) Command-Line Script (`main.py`)

A basic script (`main.py`) is included for convenience to quickly run a pre-configured benchmark.  However, for most use cases, you will want to use the `sortof` library directly in your own Python code for greater flexibility.

To run the example script (after modifying benchmark parameters within the `main.py` file):

```bash
python main.py
```

**Note:**  For customized benchmarks and integration into larger projects, it is strongly recommended to use the programmatic interface as shown in the "Programmatic Use" section above, rather than relying on the `main.py` script.

## Project Structure

```
sortof/
├── main.py          # Example script for running a pre-configured benchmark (not intended as primary library interface)
├── sortof.py        # Core library file containing sort and benchmark functions (PRIMARY LIBRARY FILE)
├── sorting_algorithms.py # Implementations of various sorting algorithms
├── report_generator.py # Functions for generating PDF reports and benchmark plots
├── report_template.tex # LaTeX template for PDF report generation
├── venv/             # Virtual environment directory (if created)
├── requirements.txt # Project dependencies
└── README.md        # Project README (this file)
```

**Key Structure Clarifications:**

  * **`sortof.py` is now clearly marked as the "PRIMARY LIBRARY FILE".**
  * **`main.py` is described as an "Example script" and "not intended as primary library interface".**  The description emphasizes it's for demonstration and quick tests.
  * **Usage examples** in "Programmatic Use" are more prominent and descriptive.
  * **The "Command-Line Interface" section is de-emphasized** and renamed to clarify that `main.py` is just a script.

## Dependencies

  * **LaTeX:** Required for PDF report generation.
  * **pylatex:** Python library for generating PDF documents.
  * **MatPlotLib:** Python library for generating visualization in function.

Dependencies are listed in `requirements.txt`. Install them using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## Author

Flori Kusari - florikusari28@gmail.com

-----

**Start benchmarking sorting algorithms in your Python projects with SortOF Library!**
