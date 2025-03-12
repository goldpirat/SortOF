# sorting_algorithms.py
import time
import random
import string
import math  # for floor in heapify
from report_generator import generate_pdf_report # Changed import to generate_latex_report
import numpy as np
from matrix_operations import matrix_determinant, matrix_trace, matrix_frobenius_norm # Import matrix operation functions

# --- Helper Functions ---
def generate_random_word(min_length, max_length):
    """Generates a random word with length between min_length and max_length."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def sort_matrix_by_determinant(matrix, order='normal'):
    """Sorts a list of matrices based on their determinant."""
    return _sort_matrices_by_metric(matrix, matrix_determinant, order)

def sort_matrix_by_trace(matrix, order='normal'):
    """Sorts a list of matrices based on their trace."""
    return _sort_matrices_by_metric(matrix, matrix_trace, order)

def sort_matrix_by_norm(matrix, order='normal'):
    """Sorts a list of matrices based on their Frobenius norm."""
    return _sort_matrices_by_metric(matrix, matrix_frobenius_norm, order)


def _sort_matrices_by_metric(matrices, metric_function, order='normal'):
    """Helper function to sort matrices based on a given metric function (determinant, trace, norm)."""
    def get_metric_value(matrix):
        np_matrix = np.array(matrix) # Convert list of lists to numpy array for matrix operations
        return metric_function(np_matrix)

    sorted_matrices = sorted(matrices, key=get_metric_value, reverse=(order == 'reverse')) # Use Python's built-in sorted()
    return sorted_matrices

# Helper for median-of-three (numbers)
def _median_of_three_num(a, low, high):
    mid = (low + high) // 2
    if a[low] > a[mid]:
        a[low], a[mid] = a[mid], a[low]
    if a[low] > a[high]:
        a[low], a[high] = a[high], a[low]
    if a[mid] > a[high]:
        a[mid], a[high] = a[high], a[mid]
    return mid

# Helper for partition (numbers)
def _partition_num(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

# Helper for median-of-three (words)
def _median_of_three_words(a, low, high, compare_func):
    mid = (low + high) // 2
    if not compare_func(a[low], a[mid]):
        a[low], a[mid] = a[mid], a[low]
    if not compare_func(a[low], a[high]):
        a[low], a[high] = a[high], a[low]
    if not compare_func(a[mid], a[high]):
        a[mid], a[high] = a[high], a[mid]
    return mid

# Helper for partition (words)
def _partition_words(a, low, high, compare_func):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if compare_func(a[j], pivot):
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

# Helper for merge (numbers)
def _merge_num(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Helper for merge (words)
def _merge_words(left, right, compare_func):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if compare_func(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# --- Heap Sort Helpers ---
def _heapify(arr, n, i, compare_func):
    """
    Heapify a subtree rooted at node 'i' (index 'i' in 0-based array),
    which is an index in 'arr[]. n is size of heap.
    This function assumes that the left and right subtrees of 'i' are already heaps.
    It makes 'i' the root of a min-heap.
    """
    smallest = i  # Initialize smallest as root
    l = 2 * i + 1     # left child index
    r = 2 * i + 2     # right child index

    # If left child exists and is smaller than root
    if l < n and compare_func(arr[l], arr[smallest]):
        smallest = l

    # If the right child exists and is smaller than smallest so far
    if r < n and compare_func(arr[r], arr[smallest]):
        smallest = r

    # If smallest is not the root
    if smallest != i:
        # Swap the root with the smallest element
        arr[i], arr[smallest] = arr[smallest], arr[i]

        # Recursively heapify the affected sub-tree
        _heapify(arr, n, smallest, compare_func)


def build_heap(arr, compare_func):
    """
    Builds a min-heap from an unordered array.
    """
    n = len(arr)
    # Index of last non-leaf node is floor(n/2) - 1
    start_index = math.floor(n / 2) - 1

    # Perform reverse level order traversal from last non-leaf node and heapify each node
    for i in range(start_index, -1, -1):
        _heapify(arr, n, i, compare_func)

# --- Core Sorting Algorithms (Generic, Comparison-Based) ---

def bubble_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using bubble sort - numbers only.
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y
    n = len(a)
    start_time = time.perf_counter()
    for i in range(n):
        for j in range(0, n - i - 1):
            if not compare_func(a[j], a[j+1]):
                a[j], a[j+1] = a[j+1], a[j]
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time]  # Return sorted array, time, and test case times


def bubble_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using bubble sort - words only.
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y

    n = len(a)
    start_time = time.perf_counter()
    for i in range(n):
        for j in range(0, n - i - 1):
            if not compare_func(a[j], a[j+1]):
                a[j], a[j+1] = a[j+1], a[j]
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def selection_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using selection sort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y

    n = len(a)
    start_time = time.perf_counter()
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if compare_func(a[j], a[min_index]):
                min_index = j
        a[i], a[min_index] = a[min_index], a[i]
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def selection_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using selection sort - words
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y

    n = len(a)
    start_time = time.perf_counter()
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if compare_func(a[j], a[min_index]):
                min_index = j
        a[i], a[min_index] = a[min_index], a[i]
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def insertion_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using insertion sort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y

    start_time = time.perf_counter()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and compare_func(key, a[j]):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def insertion_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using insertion sort - words
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y

    start_time = time.perf_counter()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and compare_func(key, a[j]):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def median_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using median sort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y
    is_numeric_comparison = compare_func == (lambda x, y: x <= y)

    def _median_sort_helper(a, low, high, compare_func, is_numeric_comparison):
        if low < high:
            m_index = _median_of_three_num(a, low, high) if is_numeric_comparison else _median_of_three_words(a, low, high, compare_func)
            a[m_index], a[high] = a[high], a[m_index]
            p = _partition_num(a, low, high) if is_numeric_comparison else _partition_words(a, low, high, compare_func)
            _median_sort_helper(a, low, p - 1, compare_func, is_numeric_comparison)
            _median_sort_helper(a, p + 1, high, compare_func, is_numeric_comparison)

    start_time = time.perf_counter()
    _median_sort_helper(a, 0, len(a) - 1, compare_func, is_numeric_comparison)
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def median_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using median sort - words
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y


    def _median_sort_helper(a, low, high, compare_func): # Removed is_numeric_comparison
        if low < high:
            m_index = _median_of_three_words(a, low, high, compare_func) # Removed conditional median selection
            a[m_index], a[high] = a[high], a[m_index]
            p = _partition_words(a, low, high, compare_func) # Removed conditional partition selection
            _median_sort_helper(a, low, p - 1, compare_func) # Removed is_numeric_comparison
            _median_sort_helper(a, p + 1, high, compare_func) # Removed is_numeric_comparison

    start_time = time.perf_counter()
    _median_sort_helper(a, 0, len(a) - 1, compare_func) # Removed is_numeric_comparison
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def merge_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using merge sort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y

    start_time_total = time.perf_counter()
    def _merge_sort_helper(arr, compare_func): # Removed is_numeric_comparison
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = _merge_sort_helper(arr[:mid], compare_func) # Removed is_numeric_comparison
        right = _merge_sort_helper(arr[mid:], compare_func) # Removed is_numeric_comparison
        return _merge_num(left, right) # Removed conditional merge selection

    sorted_a = _merge_sort_helper(a, compare_func) # Removed is_numeric_comparison
    end_time_total = time.perf_counter()
    sort_time = end_time_total - start_time_total
    return sorted_a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def merge_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using merge sort - words
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y


    start_time_total = time.perf_counter()
    def _merge_sort_helper(arr, compare_func): # Removed is_numeric_comparison
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = _merge_sort_helper(arr[:mid], compare_func) # Removed is_numeric_comparison
        right = _merge_sort_helper(arr[mid:], compare_func) # Removed is_numeric_comparison
        return _merge_words(left, right, compare_func) # Removed conditional merge selection

    sorted_a = _merge_sort_helper(a, compare_func) # Removed is_numeric_comparison
    end_time_total = time.perf_counter()
    sort_time = end_time_total - start_time_total
    return sorted_a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def quick_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None):
    """
    Sorts the array using quicksort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y

    start_time = time.perf_counter()
    def _quicksort_helper(a, compare_func):
        if len(a) <= 1:
            return a
        # --- Random Pivot Selection ---
        pivot_index = random.randint(0, len(a) - 1) # Generate a random index within the array
        pivot = a[pivot_index] # Select the element at the random index as pivot
        a[0], a[pivot_index] = a[pivot_index], a[0] # Swap pivot with the first element (optional, for consistency with original logic)
        # --- End Random Pivot Selection ---

        less = [x for x in a[1:] if compare_func(x, pivot)]
        greater = [x for x in a[1:] if not compare_func(x, pivot)]
        return _quicksort_helper(less, compare_func) + [pivot] + _quicksort_helper(greater, compare_func)

    sorted_a = _quicksort_helper(a, compare_func)
    end_time = time.perf_counter()
    sort_time = end_time - start_time
    return sorted_a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


def quicksort_words(a, order='normal', method='length'):
    """Sorts an array of words using iterative quicksort (stack-based)."""

    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y) # Sort by length
    elif method == 'lex':
        compare_func = lambda x, y: x <= y # Lexicographical sort
    else:
        raise ValueError("Invalid method for word sorting.")

    stack = [(0, len(a) - 1)] # Initialize stack with initial array bounds

    while stack:
        low, high = stack.pop() # Get current partition bounds

        if low < high:
            # Partitioning (same as in recursive version, but now iterative)
            pivot_index = random.randint(low, high) # Randomized pivot selection
            pivot = a[pivot_index]
            a[low], a[pivot_index] = a[pivot_index], a[low] # Swap pivot to low index
            pivot = a[low]

            i = low + 1
            j = high
            while True:
                while i <= j and compare_func(a[i], pivot):
                    i += 1
                while i <= j and not compare_func(a[j], pivot):
                    j -= 1
                if i > j:
                    break
                a[i], a[j] = a[j], a[i] # Swap elements
            a[low], a[j] = a[j], a[low] # Swap pivot back to correct position (j)
            partition_index = j

            # Push sub-partitions onto the stack (instead of recursive calls)
            stack.append((low, partition_index - 1))  # Subarray to the left of pivot
            stack.append((partition_index + 1, high)) # Subarray to the right of pivot

    return a # Array is sorted in-place by iterative quicksort, return it

def heap_sort_num(arr, order='normal', testcase=1, select=100, min_val=0, max_val=1000, pdf=False, test_case_sizes=None, input_description=None, count_heapify=False):
    """
    Sorts the array using heap sort - numbers
    """
    a = arr.copy()
    compare_func = lambda x, y: x <= y
    if order == 'reverse':
        compare_func = lambda x, y: x >= y
    n = len(a)
    heapify_time = 0

    if count_heapify:
        start_heapify = time.perf_counter()
        build_heap(a, compare_func)
        end_heapify = time.perf_counter()
        heapify_time = end_heapify - start_heapify
    else:
        build_heap(a, compare_func)

    sorted_array = []

    start_sort = time.perf_counter()
    for _ in range(n):
        smallest_element = a[0]
        sorted_array.append(smallest_element)
        a[0] = a[-1]
        a.pop()
        if len(a) > 0:
            _heapify(a, len(a), 0, compare_func)
    end_sort = time.perf_counter()
    sort_time = end_sort - start_sort
    total_time = sort_time + heapify_time if count_heapify else sort_time
    return sorted_array, total_time, [total_time] # Returns sorted array, time, and test_case_times


def heap_sort_words(arr, order='normal', testcase=1, select=100, min_length=3, max_length=10, method='length', pdf=False, test_case_sizes=None, input_description=None, count_heapify=False):
    """
    Sorts the array using heap sort - words
    """
    a = arr.copy()
    if method == 'length':
        compare_func = lambda x, y: len(x) <= len(y)
    else: # method == 'lex'
        compare_func = lambda x, y: x <= y
    if order == 'reverse':
        if method == 'length':
            compare_func = lambda x, y: len(x) >= len(y)
        else: # method == 'lex'
            compare_func = lambda x, y: x >= y

    n = len(a)
    heapify_time = 0

    if count_heapify:
        start_heapify = time.perf_counter()
        build_heap(a, compare_func)
        end_heapify = time.perf_counter()
        heapify_time = end_heapify - start_heapify
    else:
        build_heap(a, compare_func)

    sorted_array = []
    start_sort = time.perf_counter()
    for _ in range(n):
        smallest_element = a[0]
        sorted_array.append(smallest_element)
        a[0] = a[-1]
        a.pop()
        if len(a) > 0:
            _heapify(a, len(a), 0, compare_func)
    end_sort = time.perf_counter()
    sort_time = end_sort - start_sort
    total_time = sort_time + heapify_time if count_heapify else sort_time
    return sorted_array, total_time, [total_time] # Returns sorted array, time, and test_case_times


def radix_sort_num(arr):
    """
    Sorts non-negative integers using Radix Sort in Ascending Order.
    """
    a = arr.copy()
    if not all(isinstance(x, int) and x >= 0 for x in a):
        raise ValueError("Radix Sort can only be used for non-negative integers.")
    if not a: # Handle empty array case
        return a

    max_val = max(a)
    num_digits = len(str(max_val))
    start_time = time.perf_counter() # Start timing here
    for digit_place in range(num_digits):
        buckets = [[] for _ in range(10)]  # 10 buckets for digits 0-9
        for num in a:
            digit = (num // (10**digit_place)) % 10
            buckets[digit].append(num)
        a = []
        for bucket in buckets:
            a.extend(bucket)
    end_time = time.perf_counter() # End timing here
    sort_time = end_time - start_time
    return a, sort_time, [sort_time] # Returns sorted array, time, and test_case_times


algorithm_descriptions = {
    'bubble_num': 'Bubble Sort: Compares adjacent elements and swaps them if they are in the wrong order, repeatedly.',
    'insertion_num': 'Insertion Sort: Builds the final sorted array one item at a time by comparisons.',
    'selection_num': 'Selection Sort: Repeatedly finds the minimum element from the unsorted part and puts it at the beginning.',
    'merge_num': 'Merge Sort: A divide-and-conquer algorithm that divides the array into halves, recursively sorts them, and then merges them.',
    'quick_num': 'Quick Sort: A highly efficient divide-and-conquer algorithm, using a pivot to partition the array.',
    'heap_num': 'Heap Sort:  A comparison-based sorting algorithm using a Binary Heap data structure.',
    'radix_num': 'Radix Sort: Sorts integers by processing digits of numbers.',
    'median_num': 'Median Sort: (Example Algorithm) Sorts based on comparing elements to the median.',

    'bubble_words': 'Bubble Sort: Compares adjacent words and swaps them if they are in the wrong order, repeatedly.',
    'insertion_words': 'Insertion Sort: Builds the final sorted array of words one item at a time by comparisons.',
    'selection_words': 'Selection Sort: For words, repeatedly finds the lexicographically smallest word and puts it in place.',
    'merge_words': 'Merge Sort: Sorts words using a divide-and-conquer approach, merging sorted sub-arrays.',
    'quick_words': 'Quick Sort: Sorts words efficiently using a divide-and-conquer approach with partitioning.',
    'heap_words': 'Heap Sort:  Sorts words using a heap data structure.',
    'median_words': 'Median Sort: (Example Algorithm) Sorts words based on comparisons to the median word.',

    # --- NEW: Matrix Sorting Algorithm + Criteria Descriptions ---
    'bubble_determinant_matrices': 'Bubble Sort for Matrices (by Determinant): Sorts a list of matrices using Bubble Sort algorithm, ordering them based on their determinant values.',
    'insertion_determinant_matrices': 'Insertion Sort for Matrices (by Determinant): Sorts matrices by Determinant using Insertion Sort.',
    'selection_determinant_matrices': 'Selection Sort for Matrices (by Determinant): Sorts matrices by Determinant using Selection Sort.',
    'merge_determinant_matrices': 'Merge Sort for Matrices (by Determinant): Sorts matrices by Determinant using Merge Sort.',
    'quick_determinant_matrices': 'Quick Sort for Matrices (by Determinant): Sorts matrices by Determinant using Quick Sort.',
    'heap_determinant_matrices': 'Heap Sort for Matrices (by Determinant): Sorts matrices by Determinant using Heap Sort.',
    'median_determinant_matrices': 'Median Sort for Matrices (by Determinant): Sorts matrices by Determinant using Median Sort.',

    'bubble_trace_matrices': 'Bubble Sort for Matrices (by Trace): Sorts matrices using Bubble Sort, ordered by their trace.',
    'insertion_trace_matrices': 'Insertion Sort for Matrices (by Trace): Sorts matrices by Trace using Insertion Sort.',
    'selection_trace_matrices': 'Selection Sort for Matrices (by Trace): Sorts matrices by Trace using Selection Sort.',
    'merge_trace_matrices': 'Merge Sort for Matrices (by Trace): Sorts matrices by Trace using Merge Sort.',
    'quick_trace_matrices': 'Quick Sort for Matrices (by Trace): Sorts matrices by Trace using Quick Sort.',
    'heap_trace_matrices': 'Heap Sort for Matrices (by Trace): Sorts matrices by Trace using Heap Sort.',
    'median_trace_matrices': 'Median Sort for Matrices (by Trace): Sorts matrices by Trace using Median Sort.',

    'bubble_norm_matrices': 'Bubble Sort for Matrices (by Frobenius Norm): Sorts matrices using Bubble Sort, ordered by Frobenius norm.',
    'insertion_norm_matrices': 'Insertion Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Insertion Sort.',
    'selection_norm_matrices': 'Selection Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Selection Sort.',
    'merge_norm_matrices': 'Merge Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Merge Sort.',
    'quick_norm_matrices': 'Quick Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Quick Sort.',
    'heap_norm_matrices': 'Heap Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Heap Sort.',
    'median_norm_matrices': 'Median Sort for Matrices (by Frobenius Norm): Sorts matrices by Frobenius Norm using Median Sort.',
}


# --- Helper Functions ---
def get_algorithm_description(algorithm_name, data_type, method='length', matrix_sort_method=None, count_heapify=False):
    """Retrieves algorithm descriptions based on algorithm name, data type, and optionally matrix_sort_method."""
    descriptions = algorithm_descriptions # Use the updated descriptions dictionary

    print(f"DEBUG: get_algorithm_description called with algorithm_name='{algorithm_name}', data_type='{data_type}', method='{method}', matrix_sort_method='{matrix_sort_method}'") # DEBUG print - added matrix_sort_method

    if data_type == 'matrices': # --- MODIFIED: Matrix Data Type ---
        if not matrix_sort_method: # Should not happen, but for robustness
            raise ValueError("matrix_sort_method must be specified for matrices.")
        description_key = f"{algorithm_name}_{matrix_sort_method}_{data_type}" # Key: e.g., 'quick_determinant_matrices'
    elif data_type == 'words':
        description_key = f"{algorithm_name}_words" # Key: e.g., 'bubble_words'
    elif data_type == 'numbers':
        description_key = f"{algorithm_name}_num" # Key: e.g., 'bubble_num'
    else:
        raise ValueError(f"Unsupported data_type: {data_type} for algorithm descriptions.")

    print(f"DEBUG: Constructed description_key='{description_key}'")
    description = descriptions.get(description_key, "Description not available.")
    print(f"DEBUG: Description found: '{description}'")


    if data_type == 'words': # Word-specific description additions (no changes)
        if method == 'length':
            description += " Words are compared based on their length."
        elif method == 'lex':
            description += " Words are compared lexicographically."

    if "heap" in algorithm_name: # Heap-specific description additions (no changes)
        if count_heapify:
            description += " Includes time to build the initial heap in total execution time."
        else:
            description += " Excludes time to build the initial heap from total execution time."

    return description