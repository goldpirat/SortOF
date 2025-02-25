from sorting_algorithms import bubble_sort_num, bubble_sort_words, selection_sort_num, selection_sort_words, insertion_sort_num, insertion_sort_words
from sorting_algorithms import quicksort_num, quicksort_words, merge_sort_num, merge_sort_words, median_sort_num, median_sort_words

def main():
    # Example 1: Using a provided array.
    print("Example 1: Sorting a provided array")
    custom_array = [34, 7, 23, 32, 5, 62]
    sorted_arr, avg_time = bubble_sort_num(
        testcase=100, 
        order='normal', 
        pdf=True, 
        unsorted_array=custom_array
    )
    print("Sorted Array:", sorted_arr)
    print(f"Average Time: {avg_time:.6f} seconds\n")

    # Example 2: Generating random arrays.
    print("Example 2: Sorting random arrays")
    sorted_arr, avg_time = bubble_sort_num(
        testcase=100, 
        order='normal', 
        pdf=True, 
        min_val=1, 
        max_val=20, 
        select=35
    )
    print("Sorted Array (from final test case):", sorted_arr)
    print(f"Average Time: {avg_time:.6f} seconds\n\n")

    # Using a provided list of words:
    words = ["apple", "banana", "kiwi", "cherry", "date"]
    print("Example 3: Sorting word array in lexicograhical method.")
    sorted_words, avg_time = bubble_sort_words(
        testcase=100, 
        order='normal', 
        pdf=True, 
        unsorted_array=words, 
        method='lex'
    )
    print("Sorted Words:", sorted_words)
    print(f"Average Time: {avg_time:.6f} seconds \n\n")

    # Generating random words:
    print("Example 4: Sorting word array in length method.")
    sorted_words, avg_time = bubble_sort_words(
        testcase=100, 
        order='normal', 
        pdf=True, 
        min_length=3, 
        max_length=8, 
        select=10, 
        method='length'
    )
    print("Sorted Words (from final test case):", sorted_words)
    print(f"Average Time: {avg_time:.6f} seconds \n\n")


if __name__ == "__main__":
    main()
