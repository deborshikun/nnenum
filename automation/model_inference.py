#!/usr/bin/env python3
"""
A dedicated script to run inference on an ONNX model for a given input vector.

This script takes two command-line arguments:
1. The path to the ONNX model file.
2. A string representation of the input vector (e.g., "[-0.27, -0.39, ...]").

It loads the model, runs the input through it, and prints the output vector
to standard output in a parseable format.
"""
import argparse
import sys
import math
from typing import List, Sequence
import numpy as np
import onnxruntime as ort


def _infer_target_shape(expected_shape: Sequence, flat_len: int) -> List[int]:
    """Infer a concrete input shape given the model's expected shape and a flat vector length.
    - Keeps batch dimension at 1.
    - Fills unknown dims (None or symbolic) and tries to map the flat vector sensibly.
    - Handles common ranks 2, 3, 4.
    """
    # Convert to python list and normalize unknowns to None
    norm_shape = []
    for d in expected_shape:
        if isinstance(d, int):
            norm_shape.append(d)
        else:
            # symbolic or None
            norm_shape.append(None)

    rank = len(norm_shape)
    if rank == 0:
        # Scalar model input shouldn't happen; fall back to [1, flat_len]
        return [1, flat_len]

    # Always force batch dim to 1
    norm_shape[0] = 1

    if rank == 1:
        # Shape like [N]
        return [flat_len]

    if rank == 2:
        # [N, F]
        return [1, flat_len]

    if rank == 3:
        # [N, C, L]  or [N, L, C] (unknown). If one of dims is known, respect it.
        # If all non-batch dims known and product matches, use expected dims
        if all(d is not None for d in norm_shape[1:]):
            prod = math.prod([int(d) for d in norm_shape[1:]])
            if prod == flat_len:
                return [int(x) for x in norm_shape]
        # Compute known product of non-batch dims
        known = [1 if d is None else d for d in norm_shape[1:]]
        known_prod = math.prod(known)
        if known_prod == 0:
            known_prod = 1
        if flat_len % known_prod == 0:
            remaining = flat_len // known_prod
            # Fill unknown dims with 1, put remaining into the last unknown dim
            new_dims = list(norm_shape)
            placed = False
            for i in range(1, 3):
                if new_dims[i] is None and not placed:
                    new_dims[i] = remaining
                    placed = True
                elif new_dims[i] is None:
                    new_dims[i] = 1
            for i in range(1, 3):
                if new_dims[i] is None:
                    new_dims[i] = 1
            return [int(x) for x in new_dims]
        # Fallback: place everything in last dim
        return [1, 1, flat_len]

    if rank == 4:
        # Typical image models: [N, C, H, W] or [N, H, W, C]
        # Try to respect any known dims first.
        dims = list(norm_shape)
        # Replace N with 1
        dims[0] = 1
        # If all non-batch dims are known and product matches, use dims directly
        if all(d is not None for d in dims[1:]):
            prod = 1
            for d in dims[1:]:
                prod *= int(d)
            if prod == flat_len:
                return [int(x) for x in dims]
        # Known product among dims[1:]
        known_dims = [1 if (d is None or d == 0) else d for d in dims[1:]]
        known_prod = 1
        for v in known_dims:
            known_prod *= v
        if known_prod == 0:
            known_prod = 1
        if flat_len % known_prod == 0:
            remaining = flat_len // known_prod
            # Assign remaining to the last unknown dim among dims[1:]
            new_dims = dims[:]
            idxs = [1, 2, 3]
            placed = False
            for i in reversed(idxs):
                if new_dims[i] is None:
                    new_dims[i] = remaining
                    placed = True
                    break
            if not placed:
                # No unknowns; if all known are 1, place in W
                if new_dims[1] == 1 and new_dims[2] == 1 and new_dims[3] == 1:
                    new_dims[3] = remaining
                else:
                    # If known_prod == 1, place in W
                    if known_prod == 1:
                        new_dims[3] = remaining
                    else:
                        # Cannot map cleanly
                        raise ValueError(f"Cannot map flat vector of length {flat_len} into expected shape {expected_shape}")
            # Fill remaining unknowns as 1
            for i in idxs:
                if new_dims[i] is None:
                    new_dims[i] = 1
            return [int(x) for x in new_dims]
        # If all known are 1, we can use [1,1,1,flat_len]
        if known_prod == 1:
            return [1, 1, 1, flat_len]
        raise ValueError(f"Cannot map flat vector of length {flat_len} into expected shape {expected_shape}")

    # Generic fallback: [1, flat_len] while warning
    return [1, flat_len]


def main():
    parser = argparse.ArgumentParser(description="Run inference on an ONNX model.")
    parser.add_argument("onnx_model", help="Path to the ONNX model file.")
    parser.add_argument("input_vector", help="Input vector as a string, e.g., '[1.0, 2.0, 3.0]'")
    args = parser.parse_args()

    try:
        ort_session = ort.InferenceSession(args.onnx_model)
        input_info = ort_session.get_inputs()[0]
        input_name = input_info.name
        expected_shape = input_info.shape  # may include None or symbolic dims

        # Parse the input vector string into a list of floats
        str_vals = args.input_vector.strip().replace('[', '').replace(']', '').split(',')
        input_vals = [float(v.strip()) for v in str_vals if v.strip()]

        # Build input tensor with inferred shape
        target_shape = _infer_target_shape(expected_shape, len(input_vals))
        input_tensor = np.array(input_vals, dtype=np.float32).reshape(target_shape)

        outputs = ort_session.run(None, {input_name: input_tensor})
        output_arr = outputs[0]
        # Flatten output to 1D list for printing
        output_vector = np.asarray(output_arr).reshape(-1).tolist()

        print(f"Output: {output_vector}")

    except FileNotFoundError:
        print(f"Error: ONNX model not found at '{args.onnx_model}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
