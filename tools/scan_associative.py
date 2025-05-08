# Copyright 2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PyTorch version of utilities for parallel calculation of prefix sums."""

import functools
import torch
import numpy as np
from typing import Callable, List, Tuple, Union, Any


# def _slice_along_axis(x, start=0, stop=None, step=1, axis=0):
#     """Slices a Tensor along the given axis."""
#     # Convert None to appropriate index values
#     if stop is None:
#         stop = x.shape[axis]
#     if start < 0:
#         start = x.shape[axis] + start
#     if stop < 0:
#         stop = x.shape[axis] + stop
        
#     # Create slices for all dimensions
#     slices = [slice(None)] * x.dim()
#     slices[axis] = slice(start, stop, step)
    
#     return x[tuple(slices)]
def _slice_along_axis(x, start=0, stop=None, step=1, axis=0):
    """Slices a Tensor along the given axis."""
    # Convert None to appropriate index values
    if start is None:
        start = 0
    if stop is None:
        stop = x.shape[axis]
    
    # Now that start is guaranteed to be an integer, we can check if it's negative
    if start < 0:
        start = x.shape[axis] + start
    if stop < 0:
        stop = x.shape[axis] + stop
        
    # Create slices for all dimensions
    slices = [slice(None)] * x.dim()
    slices[axis] = slice(start, stop, step)
    
    return x[tuple(slices)]

def _interleave(a, b, axis):
    """Interleaves two Tensors along the given axis."""
    # [a b c ...] [d e f ...] -> [a d b e c f ...]
    num_elems_a = a.shape[axis]
    num_elems_b = b.shape[axis]
    
    # Handle special case where a has one more element than b
    if num_elems_a == num_elems_b + 1:
        # Interleave all but the last element of a with b
        a_prefix = _slice_along_axis(a, None, -1, axis=axis)
        a_suffix = _slice_along_axis(a, -1, None, axis=axis)
        interleaved = _interleave_with_b(a_prefix, b, axis)
        return torch.cat([interleaved, a_suffix], dim=axis)
    else:
        return _interleave_with_b(a, b, axis)
    

def _interleave_with_b(a, b, axis):
    """Helper function for interleaving equal length tensors."""
    shape = list(a.shape)
    
    # Create indices for reshaping
    shape_a = list(a.shape)
    shape_a.insert(axis + 1, 1)
    shape_b = list(b.shape)
    shape_b.insert(axis + 1, 1)
    
    # Expand dimensions to prepare for interleaving
    a_expanded = a.view(shape_a)
    b_expanded = b.view(shape_b)
    
    # Concat along the new dimension
    stacked = torch.cat([a_expanded, b_expanded], dim=axis + 1)
    
    # Set new shape for reshaping after interleaving
    final_shape = list(a.shape)
    final_shape[axis] = a.shape[axis] + b.shape[axis]
    
    # Reshape to get interleaved result
    return stacked.view(final_shape)


def _validate_elem_length(max_num_levels, elems_flat, axis):
    """Checks that elems all have the same length, and returns that length."""
    # Get the length of the first element along the specified axis
    elem_length = elems_flat[0].shape[axis]
    
    # Check that the length doesn't exceed what we can handle
    size_limit = 2**(max_num_levels + 1)
    if elem_length >= size_limit:
        raise ValueError(
            f'Input Tensors must have dimension less than '
            f'2**(max_num_levels + 1) along axis=={axis}. '
            f'(saw: {elem_length} which is not less than 2**{max_num_levels} == {size_limit})')
    
    # Check that all elements have the same length along the given axis
    for elem in elems_flat[1:]:
        if elem.shape[axis] != elem_length:
            raise ValueError(
                f'Inputs must have the same size along the given axis. '
                f'(saw: {[elem.shape for elem in elems_flat]})')
    
    return elem_length


def scan_associative(fn,
                    elems,
                    max_num_levels=48,
                    axis=0,
                    validate_args=False,
                    name=None):
    """Perform a scan with an associative binary operation, in parallel.

    The associative scan operation computes the cumulative sum, or
    [all-prefix sum](https://en.wikipedia.org/wiki/Prefix_sum), of a set of
    elements under an associative binary operation [1]. For example, using the
    ordinary addition operator `fn = lambda a, b: a + b`, this is equivalent to
    the ordinary cumulative sum `torch.cumsum` along axis 0. This method
    supports the general case of arbitrary associative binary operations operating
    on Tensors or structures of Tensors:

    ```python
    scan_associative(fn, elems) = torch.stack([
      elems[0],
      fn(elems[0], elems[1]),
      fn(elems[0], fn(elems[1], elems[2])),
      ...
      fn(elems[0], fn(elems[1], fn(..., fn(elems[-2], elems[-1]))),
    ], axis=0)
    ```

    The associative structure allows the computation to be decomposed
    and executed by parallel reduction. Where a naive sequential
    implementation would loop over all `N` elements, this method requires
    only a logarithmic number (`2 * ceil(log_2 N)`) of sequential steps, and
    can thus yield substantial performance speedups from hardware-accelerated
    vectorization. The total number of invocations of the binary operation
    (including those performed in parallel) is
    `2 * (N / 2 + N / 4 + ... + 1) = 2N - 2`
    --- i.e., approximately twice as many as a naive approach.

    [1] Blelloch, Guy E.
        [Prefix sums and their applications](
        https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
        Technical Report CMU-CS-90-190,
        School of Computer Science,
        Carnegie Mellon University, 1990.

    Args:
      fn: Python callable implementing an associative binary operation with
        signature `r = fn(a, b)`. This must satisfy associativity:
        `fn(a, fn(b, c)) == fn(fn(a, b), c)`. The inputs and result are
        (possibly nested structures of) Tensor(s), matching `elems`. Each
        Tensor has a batch dimension in place of `elem_length`; the `fn`
        is expected to map over this dimension. The result `r` has the same shape
        (and structure) as the two inputs `a` and `b`.
      elems: A (possibly nested structure of) Tensor(s), each with dimension
        `elem_length` along `axis`.
      max_num_levels: Python `int`. The `axis` of the tensors in `elems` must have
        size less than `2**(max_num_levels + 1)`. The default value is
        sufficiently large for most needs. Lowering this value can reduce
        graph-building time when `scan_associative` is used with inputs of unknown
        shape.
        Default value: `48`.
      axis: Tensor `int` axis along which to perform the scan.
      validate_args: Python `bool`. When `True`, runtime checks
        for invalid inputs are performed. This may carry a performance cost.
        Default value: `False`.
      name: Python `str` name (unused in PyTorch version, kept for API compatibility).
    Returns:
      result: A (possibly nested structure of) Tensor(s) of the same shape
        and structure as `elems`, in which the `k`th element is the result of
        recursively applying `fn` to combine the first `k` elements of
        `elems`. For example, given `elems = [a, b, c, ...]`, the result
        would be `[a, fn(a, b), fn(fn(a, b), c), ...]`.

    #### Examples

    ```python
    import torch
    import operator

    # Example 1: Partials sums of numbers.
    scan_associative(operator.add, torch.arange(0, 4))
    # ==> tensor([ 0, 1, 3, 6])

    # Example 2: Partial products of random matrices.
    matrices = torch.randn(100, 2, 2)
    def matrix_multiply(a, b):
        return torch.matmul(a, b)
    scan_associative(matrix_multiply, matrices)
    ```
    """
    slice_elem = functools.partial(_slice_along_axis, axis=axis)
    
    def flatten_nested_structure(structure):
        """Flatten a nested structure to a list."""
        if isinstance(structure, (list, tuple)):
            result = []
            for item in structure:
                result.extend(flatten_nested_structure(item))
            return result
        else:
            return [structure]
    
    def pack_structure_as(flattened, structure):
        """Pack a flat list according to the structure."""
        if isinstance(structure, (list, tuple)):
            result = []
            idx = 0
            for item in structure:
                if isinstance(item, (list, tuple)):
                    sublist_len = len(flatten_nested_structure(item))
                    result.append(pack_structure_as(flattened[idx:idx+sublist_len], item))
                    idx += sublist_len
                else:
                    result.append(flattened[idx])
                    idx += 1
            return type(structure)(result)
        else:
            return flattened[0]

    def lowered_fn(a, b):
        # Lower `fn` to operate on flattened sequences of elems.
        a_structure = pack_structure_as(a, elems)
        b_structure = pack_structure_as(b, elems)
        result = fn(a_structure, b_structure)
        return flatten_nested_structure(result)

    elems_flat = flatten_nested_structure(elems)
    
    # Convert all elements to torch tensors if they aren't already
    elems_flat = [torch.as_tensor(elem) for elem in elems_flat]
    
    if validate_args:
        elem_length = _validate_elem_length(max_num_levels, elems_flat, axis=axis)
    else:
        elem_length = elems_flat[0].shape[axis]
    
    # Handle empty or single-element case
    if elem_length < 2:
        return elems

    # Define the recursive scan function
    def _scan(level, elems):
        """Perform scan on `elems`."""
        elem_length = elems[0].shape[axis]

        # Base case of recursion: assumes `elem_length` is 2 or 3.
        if elem_length == 2:
            reduced_elems = lowered_fn(
                [slice_elem(elem, 0, 1) for elem in elems],
                [slice_elem(elem, 1, 2) for elem in elems])
                
            return [torch.cat([slice_elem(elem, 0, 1), reduced_elem], dim=axis)
                   for (reduced_elem, elem) in zip(reduced_elems, elems)]
                   
        elif elem_length == 3:
            reduced_elems = lowered_fn(
                [slice_elem(elem, 0, 1) for elem in elems],
                [slice_elem(elem, 1, 2) for elem in elems])
                
            reduced_reduced_elems = lowered_fn(
                reduced_elems,
                [slice_elem(elem, 2, 3) for elem in elems])
                
            return [
                torch.cat([slice_elem(elem, 0, 1), 
                          reduced_elem,
                          reduced_reduced_elem], dim=axis)
                for (reduced_reduced_elem, reduced_elem, elem)
                in zip(reduced_reduced_elems, reduced_elems, elems)]
        
        # For larger elem_length, use the recursive algorithm
        
        # Apply `fn` to reduce adjacent pairs to a single entry
        a = [slice_elem(elem, 0, -1, step=2) for elem in elems]
        b = [slice_elem(elem, 1, None, step=2) for elem in elems]
        reduced_elems = lowered_fn(a, b)
        
        # Recursive call on the reduced elements
        odd_elems = _scan(level - 1, reduced_elems)
        
        # Compute the even elements
        if elem_length % 2 == 0:  # Even length case
            even_results = lowered_fn(
                [slice_elem(odd_elem, 0, -1) for odd_elem in odd_elems],
                [slice_elem(elem, 2, None, 2) for elem in elems])
        else:  # Odd length case
            even_results = lowered_fn(
                [odd_elem for odd_elem in odd_elems],
                [slice_elem(elem, 2, None, 2) for elem in elems])
        
        # The first element of a scan is the same as the first element
        # of the original `elems`
        even_elems = [torch.cat([slice_elem(elem, 0, 1), result], dim=axis)
                     for (elem, result) in zip(elems, even_results)]
        
        # Interleave even and odd elements to get the final result
        return [_interleave(a, b, axis=axis) for a, b in zip(even_elems, odd_elems)]

    # Call the recursive function with appropriate level
    result_flat = _scan(max_num_levels - 1, elems_flat)
    
    # Pack the flat result according to the input structure
    return pack_structure_as(result_flat, elems)