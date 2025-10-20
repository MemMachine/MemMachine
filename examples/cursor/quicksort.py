def quicksort(arr):
    """
    Quicksort algorithm implementation in Python.
    
    Args:
        arr: List of comparable elements to be sorted
        
    Returns:
        List: New sorted list (does not modify original)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr, low=0, high=None):
    """
    In-place quicksort implementation using Lomuto partition scheme.
    
    Args:
        arr: List to be sorted (modified in place)
        low: Starting index (default: 0)
        high: Ending index (default: len(arr)-1)
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)


def partition(arr, low, high):
    """
    Lomuto partition scheme for in-place quicksort.
    
    Args:
        arr: List to partition
        low: Starting index
        high: Ending index
        
    Returns:
        int: Final position of pivot element
    """
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# Example usage and test cases
if __name__ == "__main__":
    # Test with functional quicksort
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_arr}")
    sorted_arr = quicksort(test_arr)
    print(f"Sorted array (functional): {sorted_arr}")
    
    # Test with in-place quicksort
    test_arr2 = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_arr2}")
    quicksort_inplace(test_arr2)
    print(f"Sorted array (in-place): {test_arr2}")
    
    # Test with edge cases
    print(f"Empty array: {quicksort([])}")
    print(f"Single element: {quicksort([42])}")
    print(f"Already sorted: {quicksort([1, 2, 3, 4, 5])}")
    print(f"Reverse sorted: {quicksort([5, 4, 3, 2, 1])}")
    print(f"Duplicates: {quicksort([3, 1, 4, 1, 5, 9, 2, 6, 5])}")
