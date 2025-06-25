from typing import Iterable, List, Tuple, Union

import numpy as np


def match_allowed_values(
    image_data: np.ndarray, allowed_set: Iterable[int]
) -> np.ndarray:
    """
    Optimize multiple (image_data == id) checks by using interval-based comparisons.

    For a given image_data array and a set of allowed values, this function replaces
    multiple equality checks with interval-based comparisons:
    - For a continuous interval [a, b], use (image_data > a-1) & (image_data < b+1)
    - If two intervals are separated by a single value c, they can be merged into one
      interval while excluding c: (image_data > a-1) & (image_data < d+1) & (image_data != c)

    Args:
        image_data: Numpy array containing values to check (e.g., uint8 image data)
        allowed_set: Iterable of allowed integer values

    Returns:
        np.ndarray: Boolean array indicating whether elements in image_data are in
                   the allowed_set (after interval optimization)
    """
    # Sort and deduplicate allowed_set
    allowed_values = sorted(set(allowed_set))
    if not allowed_values:
        return np.zeros(image_data.shape, dtype=bool)

    # Step 1: Group into continuous intervals
    intervals: List[Tuple[int, int]] = []
    start = allowed_values[0]
    end = allowed_values[0]
    for v in allowed_values[1:]:
        if v == end + 1:
            end = v
        else:
            intervals.append((start, end))
            start = v
            end = v
    intervals.append((start, end))

    # Step 2: Try to merge intervals
    # If two intervals have a gap of 1 (e.g., first interval [a, b], second interval [b+2, c]),
    # merge them into [a, c] while excluding b+1
    merged_intervals: List[Tuple[Tuple[int, int], Union[int, None]]] = []
    i = 0
    while i < len(intervals):
        if i < len(intervals) - 1:
            cur_start, cur_end = intervals[i]
            next_start, next_end = intervals[i + 1]
            if next_start - cur_end == 2:
                # Merge current and next intervals, record the value to exclude
                merged_intervals.append(((cur_start, next_end), cur_end + 1))
                i += 2
                continue
        # If not merged, add directly with no exclusion value
        merged_intervals.append(((intervals[i][0], intervals[i][1]), None))
        i += 1

    # Step 3: Generate conditions for each interval
    # For interval (a, b), use (image_data > a-1) & (image_data < b+1) to check if in [a,b]
    # If there's an exclusion value c, add (image_data != c)
    result_mask = None
    for (a, b), excl in merged_intervals:
        current_mask = (image_data > (a - 1)) & (image_data < (b + 1))
        if excl is not None:
            current_mask &= image_data != excl
        if result_mask is None:
            result_mask = current_mask
        else:
            result_mask |= current_mask
    return result_mask


if __name__ == "__main__":
    # Example 1: Continuous allowed values
    allowed_values1 = [1, 2, 3, 4]
    # Example 2: Allowed values with gap of 1 (missing 4)
    allowed_values2 = [1, 2, 3, 5, 6, 7]

    # Create test array (0~10)
    image_data = np.arange(11, dtype=np.uint8)

    res1 = match_allowed_values(image_data, allowed_values1)
    res2 = match_allowed_values(image_data, allowed_values2)

    print("Test array:", image_data)
    print("allowed_values1 =", allowed_values1)
    print("Result 1:", res1.astype(int))

    print("allowed_values2 =", allowed_values2)
    print("Result 2:", res2.astype(int)) 