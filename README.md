# Sorting Algorithm Simulator and Analysis Library

This Python library provides a comprehensive toolset for students studying algorithms and data structures, allowing them to simulate and analyze various sorting algorithms. It's designed to be easy to use yet highly informative, facilitating a deeper understanding of sorting techniques.

## Features

* **Algorithm Simulation:**
    * Simulate a wide range of sorting algorithms:
        * Bubble Sort
        * Selection Sort
        * Insertion Sort
        * Median Sort
        * Merge Sort
        * Quick Sort
* **Data Flexibility:**
    * Sort both numerical and textual data.
    * For textual data, choose between sorting by length or lexicographical order.
* **Customizable Test Cases:**
    * Define the number of test cases to run.
    * Provide custom arrays or generate random default arrays.
    * Choose between normal or reversed array order.
* **Performance Analysis:**
    * Measure and report the execution time for each sorting algorithm.
* **LaTeX PDF Report Generation:**
    * Generate detailed PDF reports using LaTeX, including:
        * Algorithm descriptions and notations (worst-case, best-case, average-case).
        * Test case configurations.
        * Performance results.
* **User-Friendly Interface:**
    * Designed for ease of use, making it accessible for students.

## Usage

### Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt # create this file by running pip freeze > requirements.txt
    ```

### Example

```python
from sorting_library import SortAnalyzer

analyzer = SortAnalyzer()

# Example: Sorting numbers
results_numbers = analyzer.run_sorting(
    data_type="numbers",
    algorithm="quicksort",
    num_tests=5,
    random_data=True,
    reverse_order=False
)

# Example: Sorting words by length
results_words_length = analyzer.run_sorting(
    data_type="words",
    algorithm="mergesort",
    num_tests=3,
    custom_data=["apple", "banana", "cat", "dog"],
    sort_by="length"
)

# Example: generate PDF report.
analyzer.generate_pdf_report(results_numbers, "numbers_report.pdf")
