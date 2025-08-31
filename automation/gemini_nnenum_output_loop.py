#!/usr/bin/env python3
"""
Automation loop (Gap Analysis Strategy):
 1) For each rule in an explanation file, calculate the "gaps" between the rule's bounds
    and the original, wider input bounds from a base vnnlib file.
 2) Create a temporary vnnlib file that includes the original safety property and an
    additional assertion that the input must lie within these gaps.
 3) Run nnenum to check if any input in these gaps leads to a safety violation (a counterexample).
 4) If a counterexample is found, calculate its output and immediately ask Gemini to refine the
    explanation.
 5) Repeat the entire process with the new, refined explanation.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import onnxruntime as ort

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Data Structures & Parsers ---
Interval = Tuple[float, float]
VarBounds = Dict[int, Interval]
BOUND_RE = re.compile(r"\(\s*([<>]=)\s+X_(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)")
INDEP_RE = re.compile(r"^\s*X_(?P<var>\d+)\s+belongs\s+to\s+(?P<intervals>.+?)\s*$", re.IGNORECASE)
INTERVAL_RE = re.compile(r"\[\s*(?P<a>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*(?P<b>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")


def parse_base_vnnlib_bounds(vnnlib_text: str) -> VarBounds:
    bounds: Dict[int, Dict[str, float]] = {}
    for match in BOUND_RE.finditer(vnnlib_text):
        op, var_idx_str, val_str = match.groups()
        var_idx = int(var_idx_str)
        val = float(val_str)
        if var_idx not in bounds:
            bounds[var_idx] = {'min': -float('inf'), 'max': float('inf')}
        if op == '>=':
            bounds[var_idx]['min'] = max(bounds[var_idx]['min'], val)
        elif op == '<=':
            bounds[var_idx]['max'] = min(bounds[var_idx]['max'], val)
    final_bounds: VarBounds = {}
    for var_idx, b in bounds.items():
        if b['min'] > -float('inf') and b['max'] < float('inf'):
            final_bounds[var_idx] = (b['min'], b['max'])
    return final_bounds

def parse_explanation_intervals(text: str) -> List[Interval]:
    intervals = []
    for m in INTERVAL_RE.finditer(text):
        a, b = float(m.group("a")), float(m.group("b"))
        intervals.append((min(a, b), max(a, b)))
    return intervals

def parse_explanation_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith('#') or line.startswith(';'):
        return None
    m = INDEP_RE.match(line)
    if m:
        var = int(m.group('var'))
        ivals = parse_explanation_intervals(m.group('intervals'))
        if not ivals:
            raise ValueError(f"No intervals parsed in independent bound line: '{line}'")
        return {'var': var, 'intervals': ivals, 'raw': line}
    if "when" in line.lower():
        print(f"Warning: Skipping dependent rule (not yet supported): {line}")
        return None
    return None

def parse_vector_from_file_content(text: str) -> Optional[List[float]]:
    m = re.search(r"\[([^\]]+)\]", text)
    if not m:
        return None
    try:
        return [float(p.strip()) for p in m.group(1).split(',')]
    except (ValueError, IndexError):
        return None

# --- Logic for Bound Difference ---
def calculate_bound_difference(original_bound: Interval, llm_intervals: List[Interval]) -> List[Interval]:
    orig_min, orig_max = original_bound
    if not llm_intervals:
        return [original_bound]
    llm_intervals.sort()
    merged = [llm_intervals[0]]
    for current_min, current_max in llm_intervals[1:]:
        last_min, last_max = merged[-1]
        if current_min <= last_max:
            merged[-1] = (last_min, max(last_max, current_max))
        else:
            merged.append((current_min, current_max))
    gaps: List[Interval] = []
    current_pos = orig_min
    for llm_min, llm_max in merged:
        if current_pos < llm_min:
            gaps.append((current_pos, llm_min))
        current_pos = max(current_pos, llm_max)
    if current_pos < orig_max:
        gaps.append((current_pos, orig_max))
    return [(a, b) for a, b in gaps if (b - a) > 1e-9]

def vnnlib_and_inside(var_index: int, interval: Interval) -> str:
    a, b = interval
    return f"(and (>= X_{var_index} {a}) (<= X_{var_index} {b}))"

# --- External Tool Wrappers ---
def run_nnenum(repo_root: Path, onnx_path: Path, vnnlib_path: Path, timeout: int, outfile: Path):
    ps_cmd = (
        '$env:OPENBLAS_NUM_THREADS="1";$env:OMP_NUM_THREADS="1";$env:MKL_NUM_THREADS="1";'
        '$env:NUMEXPR_NUM_THREADS="1";$env:PYTHONPATH=(Resolve-Path ./src).Path;'
        f'python -m nnenum.nnenum "{onnx_path}" "{vnnlib_path}" {timeout} "{outfile}"'
    )
    subprocess.run(['powershell', '-NoProfile', '-Command', ps_cmd], cwd=str(repo_root), capture_output=True, text=True, encoding='utf-8')

def ensure_gemini(api_key: Optional[str]):
    if genai is None:
        raise ImportError("google-generativeai is not installed. Run: pip install google-generativeai")
    if not api_key:
        raise ValueError("Gemini API key missing. Pass --api-key or set GEMINI_API_KEY env var.")
    genai.configure(api_key=api_key)

def call_gemini_to_refine_bounds(api_key: str, model_name: str, base_adv_lines: List[str], new_counterexample: List[float], prev_bounds_text: str) -> str:
    ensure_gemini(api_key)
    model = genai.GenerativeModel(model_name)
    header = (
        "You are an expert at refining safety explanations for neural networks.\n"
        "Your previous explanation of an UNSAFE region was found to be flawed because it "
        "missed the following UNSAFE point (a counterexample).\n"
        "Refine the 'Previous explanation' to be more general so it INCLUDES this new unsafe point, "
        "while still covering the original adversarial inputs.\n"
        "Preserve the output format exactly. Do not include any commentary."
    )
    cex_section = "Newly found UNSAFE point to include:\n" + ", ".join(f"X_{i}={v}" for i, v in enumerate(new_counterexample))
    
    parts = [header, "Previous explanation:\n" + prev_bounds_text.strip(), cex_section]
    if base_adv_lines:
        orig_adv_section = "Original UNSAFE inputs to explain (for context):\n" + "\n".join(base_adv_lines)
        parts.append(orig_adv_section)

    full_prompt = "\n\n".join(parts)
    resp = model.generate_content(full_prompt)
    try:
        return "".join(p.text for c in resp.candidates for p in c.content.parts).strip()
    except Exception:
        raise RuntimeError("Gemini response had no text content")

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Automate LLM-vs-nnenum loop using gap analysis.")
    parser.add_argument('--api-key', default=os.getenv('GEMINI_API_KEY'))
    parser.add_argument('--gemini-model', default='gemini-1.5-pro-latest')
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--base-vnnlib', required=True, type=Path)
    parser.add_argument('--explanation', required=True, type=Path)
    parser.add_argument('--adv-inputs', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output-dir', default='automation_output', type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    base_vnnlib_text = args.base_vnnlib.read_text(encoding='utf-8')
    original_input_bounds = parse_base_vnnlib_bounds(base_vnnlib_text)

    ort_session = ort.InferenceSession(str(args.onnx))
    input_name = ort_session.get_inputs()[0].name

    print("--- Parsed Original Input Bounds ---")
    for var, (vmin, vmax) in sorted(original_input_bounds.items()):
        print(f"  X_{var}: [{vmin}, {vmax}]")
    print("-" * 34)
    
    current_explanation_path = args.explanation

    for iteration in range(args.iterations):
        print(f"\n--- Starting Iteration {iteration}: Analyzing {current_explanation_path.name} ---")
        if not current_explanation_path.exists():
            sys.exit(f"ERROR: Explanation file not found: {current_explanation_path}.")
        
        explanation_text = current_explanation_path.read_text(encoding='utf-8')
        explanation_lines = explanation_text.splitlines()
        
        counterexample_found_this_iteration = False
        
        for i, line in enumerate(explanation_lines):
            parsed_rule = parse_explanation_line(line)
            if not parsed_rule:
                continue

            var_idx = parsed_rule['var']
            if var_idx not in original_input_bounds:
                print(f"  Skipping rule for X_{var_idx} (no original bounds found).")
                continue

            gaps = calculate_bound_difference(original_input_bounds[var_idx], parsed_rule['intervals'])
            if not gaps:
                print(f"  Rule {i+1} for X_{var_idx} has no gaps to test.")
                continue
            
            gap_conditions = " ".join(vnnlib_and_inside(var_idx, gap) for gap in gaps)
            gap_assertion = f"(assert (or {gap_conditions}))"
            
            vnnlib_path = args.output_dir / f"iter{iteration}_rule{i+1}_gaps.vnnlib"
            with vnnlib_path.open('w', encoding='utf-8') as f:
                f.write(base_vnnlib_text)
                f.write(f"\n\n; --- Gap analysis for rule: {line.strip()} ---\n")
                f.write(gap_assertion + "\n")
            
            print(f"  Testing gaps for rule {i+1} (X_{var_idx}) -> {vnnlib_path.name}")
            result_path = args.output_dir / f"result_iter{iteration}_rule{i+1}.txt"
            run_nnenum(repo_root, args.onnx, vnnlib_path, args.timeout, result_path)
            
            cinput_path = result_path.with_suffix('.cinput')
            cex_input = parse_vector_from_file_content(cinput_path.read_text(encoding='utf-8')) if cinput_path.exists() else None

            if cex_input:
                print("\n--- FLAW IN EXPLANATION FOUND ---")
                counterexample_found_this_iteration = True
                
                input_tensor = np.array(cex_input, dtype=np.float32).reshape(1, -1)
                outputs = ort_session.run(None, {input_name: input_tensor})[0][0]
                
                print(f"  Found an UNSAFE point missed by the explanation:")
                print(f"  Input:  " + ", ".join(f"X_{i}={v:.6f}" for i, v in enumerate(cex_input)))
                print(f"  Output: " + ", ".join(f"Y_{i}={v:.6f}" for i, v in enumerate(outputs)))
                
                print("\n--- Refinement Step ---")
                print("Asking Gemini to refine the explanation...")
                try:
                    base_adv_lines = args.adv_inputs.read_text(encoding='utf-8').splitlines()
                    refined_text = call_gemini_to_refine_bounds(args.api_key, args.gemini_model, base_adv_lines, cex_input, explanation_text)
                    
                    next_explanation_path = args.output_dir / f"New Bounds {iteration + 1}.txt"
                    next_explanation_path.write_text(refined_text + '\n', encoding='utf-8')
                    print(f"Wrote refined explanation to: {next_explanation_path}")
                    current_explanation_path = next_explanation_path
                except Exception as e:
                    print(f"ERROR: Gemini refinement call failed: {e}", file=sys.stderr)
                    sys.exit("Stopping loop due to LLM error.")
                
                # Break the inner loop to start a new iteration with the refined explanation
                break
            
            time.sleep(1)

        if not counterexample_found_this_iteration:
            print("\n--- VERIFICATION SUCCESS ---")
            print("No counterexamples found in any gaps. The current explanation appears to be robust.")
            break

    print("\nDone.")

if __name__ == '__main__':
    main()