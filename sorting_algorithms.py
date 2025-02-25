import time
import random
import string
from report_generator import generate_pdf_report


def _compare(x, y, method):
    if method == 'length':
        return len(x) <= len(y)
    elif method == 'lex':
        return x <= y
    else:
        raise ValueError("Method not recognized. Use 'length' or 'lex'.")


def bubble_sort(arr):
    """
    Sorts the array using bubble sort.
    Returns the sorted array.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def bubble_sort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    """
    Sorts numbers using the bubble sort algorithm.

    Parameters:
      - testcase (int): Number of test cases (if unsorted_array is not provided).
      - order (str): 'normal' for ascending or 'reverse' for descending order.
      - pdf (bool): Whether to generate a PDF report.
      - unsorted_array (list, optional): A user-supplied array of numbers.
      - min_val (int, optional): Minimum value for random generation (required if unsorted_array is not provided).
      - max_val (int, optional): Maximum value for random generation (required if unsorted_array is not provided).
      - select (int, optional): Size of the array for random generation (required if unsorted_array is not provided).

    Returns:
      - sorted_result (list): The sorted array from the final test case.
      - average_time (float): The average time taken to sort.
    """
    # Use provided array if available; otherwise generate random arrays.
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("For random array generation, you must provide min_val, max_val, and select.")
        arrays = [
            [random.randint(min_val, max_val) for _ in range(select)]
            for _ in range(testcase)
        ]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None

    # Run bubble sort on each test case.
    for arr in arrays:
        arr_copy = arr.copy()  # work on a copy to keep the original intact
        start = time.perf_counter()
        sorted_arr = bubble_sort(arr_copy)
        end = time.perf_counter()

        # Adjust the order if needed.
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))

        total_time += (end - start)
        sorted_result = sorted_arr  # store the result from the last test case

    average_time = total_time / actual_testcase

    # Generate PDF report if requested.
    if pdf:
        description = (
            "Bubble Sort is a simple comparison-based algorithm.\n"
            "Worst-case and average-case time complexity: O(n²).\n"
            "Best-case (when the array is nearly sorted): O(n).\n"
            "It works by repeatedly swapping adjacent elements if they are in the wrong order."
        )
        generate_pdf_report("Bubble Sort (Numbers)", average_time, description, output_filename="bubble_sort_report.pdf")

    return sorted_result, average_time


def generate_random_word(min_length, max_length):
    """
    Generates a random word with length between min_length and max_length.
    """
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def bubble_sort_words(*, testcase, order, pdf, unsorted_array=None, min_length=3, max_length=10, select=10, method='length'):
    """
    Sorts words using the bubble sort algorithm.
    
    Parameters:
      - testcase (int): Number of test cases to run (if unsorted_array is not provided).
      - order (str): 'normal' for ascending or 'reverse' for descending order.
      - pdf (bool): Whether to generate a PDF report.
      - unsorted_array (list, optional): A user-supplied list of words.
      - min_length (int, optional): Minimum length for generating random words (if unsorted_array is not provided).
      - max_length (int, optional): Maximum length for generating random words.
      - select (int, optional): Number of words in each test case.
      - method (str, optional): 'length' to sort by word length (default) or 'lex' for lexicographical order.
    
    Returns:
      - sorted_result (list): The sorted list from the final test case.
      - average_time (float): The average time taken to sort.
    """
    
    # Determine test cases based on whether an unsorted_array is provided.
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        # Generate a list of random words for each test case.
        arrays = [
            [generate_random_word(min_length, max_length) for _ in range(select)]
            for _ in range(testcase)
        ]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None

    # Run bubble sort on each test case.
    for arr in arrays:
        arr_copy = arr.copy()  # Work on a copy to preserve the original list.
        start = time.perf_counter()
        n = len(arr_copy)
        # Bubble sort algorithm adapted for word comparisons.
        for i in range(n):
            for j in range(0, n - i - 1):
                if method == 'length':
                    if len(arr_copy[j]) > len(arr_copy[j+1]):
                        arr_copy[j], arr_copy[j+1] = arr_copy[j+1], arr_copy[j]
                elif method == 'lex':
                    if arr_copy[j] > arr_copy[j+1]:
                        arr_copy[j], arr_copy[j+1] = arr_copy[j+1], arr_copy[j]
                else:
                    raise ValueError("Method not recognized. Use 'length' or 'lex'.")
        end = time.perf_counter()
        total_time += (end - start)
        sorted_result = arr_copy  # Save the result from the last test case.

    average_time = total_time / actual_testcase

    # Reverse the sorted result if required.
    if order.lower() == 'reverse':
        sorted_result = list(reversed(sorted_result))  
        # The reason we can do this is because it does not affect our time as we hace stopped it already.

    # Generate PDF report if requested.
    if pdf:
        description = (
            "Bubble Sort is a simple comparison-based sorting algorithm.\n"
            "This version has been adapted to sort words.\n"
            "It can sort words either by length or lexicographically.\n"
            "Time Complexity: Worst-case and average-case O(n²), Best-case O(n) for nearly sorted arrays."
        )
        generate_pdf_report("Bubble Sort (Words)", average_time, description, output_filename="bubble_sort_words_report.pdf")

    return sorted_result, average_time



def selection_sort(arr):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if a[j] < a[min_index]:
                min_index = j
        a[i], a[min_index] = a[min_index], a[i]
    return a

def selection_sort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    # Prepare test arrays.
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("Provide min_val, max_val, and select for random generation.")
        arrays = [[random.randint(min_val, max_val) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = selection_sort(arr)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Selection Sort repeatedly finds the minimum element and moves it to the beginning.\n"
            "Time Complexity: O(n²) in all cases."
        )
        generate_pdf_report("Selection Sort (Numbers)", average_time, description,
                            output_filename="selection_sort_num_report.pdf")
    return sorted_result, average_time


def _selection_sort_words(arr, method='length'):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if not _compare(a[min_index], a[j], method):
                min_index = j
        a[i], a[min_index] = a[min_index], a[i]
    return a

def selection_sort_words(*, testcase, order, pdf, unsorted_array=None,
                         min_length=3, max_length=10, select=10, method='length'):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        arrays = [[generate_random_word(min_length, max_length) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = _selection_sort_words(arr, method=method)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Selection Sort (Words) repeatedly finds the minimum word (by length or lexicographical order) and moves it to the beginning.\n"
            "Time Complexity: O(n²) in all cases."
        )
        generate_pdf_report("Selection Sort (Words)", average_time, description,
                            output_filename="selection_sort_words_report.pdf")
    return sorted_result, average_time




def insertion_sort(arr):
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a

def insertion_sort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("Provide min_val, max_val, and select for random generation.")
        arrays = [[random.randint(min_val, max_val) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = insertion_sort(arr)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Insertion Sort builds the final sorted array one element at a time.\n"
            "Time Complexity: Worst-case O(n²), Best-case O(n) for nearly sorted arrays."
        )
        generate_pdf_report("Insertion Sort (Numbers)", average_time, description,
                            output_filename="insertion_sort_num_report.pdf")
    return sorted_result, average_time


def _insertion_sort_words(arr, method='length'):
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        if method == 'length':
            while j >= 0 and len(a[j]) > len(key):
                a[j + 1] = a[j]
                j -= 1
        elif method == 'lex':
            while j >= 0 and a[j] > key:
                a[j + 1] = a[j]
                j -= 1
        else:
            raise ValueError("Method not recognized.")
        a[j + 1] = key
    return a

def insertion_sort_words(*, testcase, order, pdf, unsorted_array=None,
                         min_length=3, max_length=10, select=10, method='length'):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        arrays = [[generate_random_word(min_length, max_length) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = _insertion_sort_words(arr, method=method)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Insertion Sort (Words) builds the final sorted list one word at a time.\n"
            "Time Complexity: Worst-case O(n²), Best-case O(n) for nearly sorted lists."
        )
        generate_pdf_report("Insertion Sort (Words)", average_time, description,
                            output_filename="insertion_sort_words_report.pdf")
    return sorted_result, average_time



# Helper for median-of-three
def median_of_three(a, low, high):
    mid = (low + high) // 2
    if a[low] > a[mid]:
        a[low], a[mid] = a[mid], a[low]
    if a[low] > a[high]:
        a[low], a[high] = a[high], a[low]
    if a[mid] > a[high]:
        a[mid], a[high] = a[high], a[mid]
    return mid

def partition(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def median_sort(arr):
    a = arr.copy()
    def _median_sort_helper(a, low, high):
        if low < high:
            m_index = median_of_three(a, low, high)
            a[m_index], a[high] = a[high], a[m_index]
            p = partition(a, low, high)
            _median_sort_helper(a, low, p - 1)
            _median_sort_helper(a, p + 1, high)
    _median_sort_helper(a, 0, len(a) - 1)
    return a

def median_sort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("Provide min_val, max_val, and select for random generation.")
        arrays = [[random.randint(min_val, max_val) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = median_sort(arr)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Median Sort is a variant of Quick Sort using median-of-three pivot selection.\n"
            "Average Time Complexity: O(n log n), Worst-case: O(n²)."
        )
        generate_pdf_report("Median Sort (Numbers)", average_time, description,
                            output_filename="median_sort_num_report.pdf")
    return sorted_result, average_time



def _median_of_three_words(a, low, high, method):
    mid = (low + high) // 2
    if not _compare(a[low], a[mid], method):
        a[low], a[mid] = a[mid], a[low]
    if not _compare(a[low], a[high], method):
        a[low], a[high] = a[high], a[low]
    if not _compare(a[mid], a[high], method):
        a[mid], a[high] = a[high], a[mid]
    return mid

def _partition_words(a, low, high, method):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if _compare(a[j], pivot, method):
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def _median_sort_words(a, low, high, method):
    if low < high:
        m_index = _median_of_three_words(a, low, high, method)
        a[m_index], a[high] = a[high], a[m_index]
        p = _partition_words(a, low, high, method)
        _median_sort_words(a, low, p - 1, method)
        _median_sort_words(a, p + 1, high, method)

def _median_sort_words_wrapper(arr, method='length'):
    a = arr.copy()
    _median_sort_words(a, 0, len(a) - 1, method)
    return a

def median_sort_words(*, testcase, order, pdf, unsorted_array=None,
                      min_length=3, max_length=10, select=10, method='length'):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        arrays = [[generate_random_word(min_length, max_length) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = _median_sort_words_wrapper(arr, method=method)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Median Sort (Words) is a variant of Quick Sort using median-of-three pivot selection with custom comparisons.\n"
            "Average Time Complexity: O(n log n), Worst-case: O(n²)."
        )
        generate_pdf_report("Median Sort (Words)", average_time, description,
                            output_filename="median_sort_words_report.pdf")
    return sorted_result, average_time


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
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

def merge_sort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("Provide min_val, max_val, and select for random generation.")
        arrays = [[random.randint(min_val, max_val) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = merge_sort(arr)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Merge Sort divides the array into halves, sorts them, and merges the sorted halves.\n"
            "Time Complexity: O(n log n) in all cases."
        )
        generate_pdf_report("Merge Sort (Numbers)", average_time, description,
                            output_filename="merge_sort_num_report.pdf")
    return sorted_result, average_time


def _merge_words(left, right, method):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if _compare(left[i], right[j], method):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def _merge_sort_words(arr, method):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = _merge_sort_words(arr[:mid], method)
    right = _merge_sort_words(arr[mid:], method)
    return _merge_words(left, right, method)

def merge_sort_words(*, testcase, order, pdf, unsorted_array=None,
                     min_length=3, max_length=10, select=10, method='length'):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        arrays = [[generate_random_word(min_length, max_length) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = _merge_sort_words(arr, method)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Merge Sort (Words) divides the list into halves, sorts each half using the chosen comparison, and merges them.\n"
            "Time Complexity: O(n log n) in all cases."
        )
        generate_pdf_report("Merge Sort (Words)", average_time, description,
                            output_filename="merge_sort_words_report.pdf")
    return sorted_result, average_time


def quicksort(arr):
    a = arr.copy()
    if len(a) <= 1:
        return a
    pivot = a[0]
    less = [x for x in a[1:] if x <= pivot]
    greater = [x for x in a[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)

def quicksort_num(*, testcase, order, pdf, unsorted_array=None, min_val=None, max_val=None, select=None):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        if None in (min_val, max_val, select):
            raise ValueError("Provide min_val, max_val, and select for random generation.")
        arrays = [[random.randint(min_val, max_val) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = quicksort(arr)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Quick Sort is a divide-and-conquer algorithm that partitions the list around a pivot.\n"
            "Average Time Complexity: O(n log n), Worst-case: O(n²)."
        )
        generate_pdf_report("Quick Sort (Numbers)", average_time, description,
                            output_filename="quicksort_num_report.pdf")
    return sorted_result, average_time


def _quicksort_words(arr, method):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if _compare(x, pivot, method)]
    greater = [x for x in arr[1:] if not _compare(x, pivot, method)]
    return _quicksort_words(less, method) + [pivot] + _quicksort_words(greater, method)

def quicksort_words(*, testcase, order, pdf, unsorted_array=None,
                    min_length=3, max_length=10, select=10, method='length'):
    if unsorted_array is not None:
        arrays = [unsorted_array]
        actual_testcase = 1
    else:
        arrays = [[generate_random_word(min_length, max_length) for _ in range(select)]
                  for _ in range(testcase)]
        actual_testcase = testcase

    total_time = 0
    sorted_result = None
    for arr in arrays:
        start = time.perf_counter()
        sorted_arr = _quicksort_words(arr.copy(), method)
        end = time.perf_counter()
        if order.lower() == 'reverse':
            sorted_arr = list(reversed(sorted_arr))
        total_time += (end - start)
        sorted_result = sorted_arr

    average_time = total_time / actual_testcase

    if pdf:
        description = (
            "Quick Sort (Words) is a divide-and-conquer algorithm that partitions the list around a pivot using custom comparisons.\n"
            "Average Time Complexity: O(n log n), Worst-case: O(n²)."
        )
        generate_pdf_report("Quick Sort (Words)", average_time, description,
                            output_filename="quicksort_words_report.pdf")
    return sorted_result, average_time
