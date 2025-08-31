"""
Automation loop (Gap Analysis Strategy with External Inference):
 1) For each rule in an explanation file, calculate the "gaps" between the rule's bounds
    and the original, wider input bounds.
 2) Create a temporary vnnlib file containing the FULL original property and an
    additional assertion to search within these gaps.
 3) Run nnenum. If a counterexample is found, it signifies a flaw in the explanation.
 4) Call the separate 'run_inference.py' script to get the model's output for the counterexample.
 5) Immediately ask Gemini to refine the explanation with this new input/output pair.
 6) Repeat the process with the new, refined explanation.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import google.generativeai as genai


Interval = Tuple[float, float]
VarBounds = Dict[int, Interval]
BOUND_RE = re.compile(r"\(\s*assert\s*\(\s*(or|and)\s*\(.*?[<>]=?\s*X_(\d+).*?", re.DOTALL)
INDEP_RE = re.compile(r"^\s*X_(?P<var>\d+)\s+belongs\s+to\s+(?P<intervals>.+?)\s*$", re.IGNORECASE)
INTERVAL_RE = re.compile(r"\[\s*(?P<a>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*(?P<b>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")


def parse_base_vnnlib_bounds(vnnlib_text: str) -> VarBounds:
    """Parses a VNNLIB file to extract the simple input bounds for each X_i variable."""
    bounds: Dict[int, Dict[str, float]] = {}
    
    # Parsing single-line or multi-line asserts
    bound_pattern = re.compile(r"\(\s*([<>]=)\s+X_(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)")

    for match in bound_pattern.finditer(vnnlib_text):
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


def parse_first_vector_with_len(text: str, expected_len: Optional[int] = None) -> Optional[List[float]]:
    """Finds bracketed vectors in text and returns the first one matching expected_len if provided.
    Falls back to the first successfully parsed vector if expected_len is None or no match is found.
    """
    matches = re.findall(r"\[([^\]]+)\]", text)
    candidates: List[List[float]] = []
    for grp in matches:
        try:
            vec = [float(p.strip()) for p in grp.split(',') if p.strip()]
            candidates.append(vec)
        except Exception:
            continue
    if not candidates:
        return None
    if expected_len is None:
        return candidates[0]
    for v in candidates:
        if len(v) == expected_len:
            return v
    return None

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

def run_nnenum(repo_root: Path, onnx_path: Path, vnnlib_path: Path, timeout: int, outfile: Path) -> str:
    ps_cmd = (
        '$env:OPENBLAS_NUM_THREADS="1"; '
        '$env:OMP_NUM_THREADS="1"; '
        '$env:MKL_NUM_THREADS="1"; '
        '$env:NUMEXPR_NUM_THREADS="1"; '
        '$env:PYTHONPATH=(Resolve-Path ./src).Path; '
        f'python -m nnenum.nnenum "{onnx_path}" "{vnnlib_path}" {timeout} "{outfile}"'
    )
    res = subprocess.run(
        ['powershell', '-NoProfile', '-Command', ps_cmd],
        cwd=str(repo_root), capture_output=True, text=True, encoding='utf-8'
    )
    return (res.stdout or '') + '\n' + (res.stderr or '')

def ensure_gemini(api_key: Optional[str]):
    if genai is None:
        raise ImportError("google-generativeai is not installed. Run: pip install google-generativeai")
    if not api_key:
        raise ValueError("Gemini API key missing. Pass --api-key or set GEMINI_API_KEY env var.")
    genai.configure(api_key=api_key)

def call_gemini_to_refine_bounds(
    api_key: str, model_name: str, base_adv_lines: List[str],
    new_counterexample: List[float], prev_bounds_text: str,
    max_adv_lines: int = 10, max_retries: int = 5, initial_backoff: float = 10.0
) -> str:
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

    # Limit adversarial context to reduce token usage
    limited_adv = base_adv_lines[:max_adv_lines] if base_adv_lines else []
    
    parts = [header, "Previous explanation:\n" + prev_bounds_text.strip(), cex_section]
    if limited_adv:
        orig_adv_section = "Original UNSAFE inputs to explain (for context):\n" + "\n".join(limited_adv)
        parts.append(orig_adv_section)

    full_prompt = "\n\n".join(parts)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(full_prompt)
            return "".join(p.text for c in resp.candidates for p in c.content.parts).strip()
        except Exception as e:
            msg = str(e)
            last_err = e
            # Retry on quota/rate-limit errors
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                sleep = initial_backoff * (2 ** (attempt - 1))
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError(f"Gemini response failed after {max_retries} retries: {last_err}")

def call_gemini_to_refine_bounds_batch(
    api_key: str, model_name: str, base_adv_lines: List[str],
    new_counterexamples: List[List[float]], prev_bounds_text: str,
    outputs: Optional[List[List[float]]] = None,
    include_outputs: bool = False,
    max_adv_lines: int = 10, max_retries: int = 5, initial_backoff: float = 10.0
) -> str:
    """Batch refinement: include all counterexamples (optionally with outputs) in a single call."""
    ensure_gemini(api_key)
    model = genai.GenerativeModel(model_name)

    header = (
        "You are an expert at refining safety explanations for neural networks.\n"
        "Your previous explanation of an UNSAFE region was found to be flawed because it "
        "missed the following UNSAFE points (counterexamples).\n"
        "Refine the 'Previous explanation' to be more general so it INCLUDES these new unsafe points, "
        "while still covering the original adversarial inputs.\n"
        "Preserve the output format exactly. Do not include any commentary."
    )

    lines = []
    for idx, vec in enumerate(new_counterexamples):
        assignments = ", ".join(f"X_{i}={v}" for i, v in enumerate(vec))
        if include_outputs and outputs is not None and idx < len(outputs) and outputs[idx] is not None:
            lines.append(f"Input: {assignments}; Output: {outputs[idx]}")
        else:
            lines.append(f"Input: {assignments}")
    cex_section = "Newly found UNSAFE points to include:\n" + "\n".join(lines)

    # Limit adversarial context to reduce token usage
    limited_adv = base_adv_lines[:max_adv_lines] if base_adv_lines else []
    
    parts = [header, "Previous explanation:\n" + prev_bounds_text.strip(), cex_section]
    if limited_adv:
        orig_adv_section = "Original UNSAFE inputs to explain (for context):\n" + "\n".join(limited_adv)
        parts.append(orig_adv_section)

    full_prompt = "\n\n".join(parts)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(full_prompt)
            return "".join(p.text for c in resp.candidates for p in c.content.parts).strip()
        except Exception as e:
            msg = str(e)
            last_err = e
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                sleep = initial_backoff * (2 ** (attempt - 1))
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError(f"Gemini response failed after {max_retries} retries: {last_err}")

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
    parser.add_argument('--max-adv-lines', type=int, default=10)
    parser.add_argument('--max-retries', type=int, default=5)
    parser.add_argument('--initial-backoff', type=float, default=10.0)
    parser.add_argument('--batch-refine', action='store_true', help='Collect all counterexamples in an iteration and refine once at the end.')
    parser.add_argument('--io-log-path', type=Path, help='Optional path to write input-output pairs. Defaults to output-dir/unsafe_io_pairs_iter{N}.txt')
    parser.add_argument('--include-outputs-in-prompt', action='store_true', help='Include model outputs with inputs when prompting the LLM.')
    parser.add_argument('--skip-llm', action='store_true', help='Skip calling LLM; only collect and write input-output pairs.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    base_vnnlib_text = args.base_vnnlib.read_text(encoding='utf-8')
    original_input_bounds = parse_base_vnnlib_bounds(base_vnnlib_text)

    print("Parsed Original Input Bounds")
    for var, (vmin, vmax) in sorted(original_input_bounds.items()):
        print(f"  X_{var}: [{vmin}, {vmax}]")
    print("-" * 34)

    expected_input_len = len(original_input_bounds)
    
    current_explanation_path = args.explanation

    for iteration in range(args.iterations):
        print(f"\n Starting Iteration {iteration}: Analyzing {current_explanation_path.name}")
        if not current_explanation_path.exists():
            sys.exit(f"ERROR: Explanation file not found: {current_explanation_path}.")
        
        explanation_text = current_explanation_path.read_text(encoding='utf-8')
        explanation_lines = explanation_text.splitlines()
        
        counterexample_found_this_iteration = False
        collected_inputs: List[List[float]] = []
        collected_outputs: List[List[float]] = []
        
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
                f.write(f"\n\n; Gap analysis for rule: {line.strip()}\n")
                f.write(gap_assertion + "\n")
            
            print(f"  Testing gaps for rule {i+1} (X_{var_idx}) -> {vnnlib_path.name}")
            result_path = args.output_dir / f"result_iter{iteration}_rule{i+1}.txt"
            output_text = run_nnenum(repo_root, args.onnx, vnnlib_path, args.timeout, result_path)
            
            cinput_path = result_path.with_suffix('.cinput')
            coutput_path = result_path.with_suffix('.coutput')

            cex_input: Optional[List[float]] = None
            if cinput_path.exists():
                try:
                    cex_input = parse_first_vector_with_len(cinput_path.read_text(encoding='utf-8'), expected_input_len)
                except Exception as e:
                    print(f"  Warning: failed reading cinput file {cinput_path.name}: {e}")
            if cex_input is None:
                # Fallback: try to parse from nnenum stdout/stderr but ensure length matches
                cex_input = parse_first_vector_with_len(output_text, expected_input_len)

            if cex_input and len(cex_input) == expected_input_len:
                print("\n FLAW IN EXPLANATION FOUND")
                counterexample_found_this_iteration = True

                # Try to get outputs from .coutput; fallback to local inference
                output_vector: Optional[List[float]] = None
                if coutput_path.exists():
                    try:
                        output_vector = parse_vector_from_file_content(coutput_path.read_text(encoding='utf-8'))
                    except Exception as e:
                        print(f"  Warning: failed reading coutput file {coutput_path.name}: {e}")

                if output_vector is None:
                    # Call the separate inference script
                    inference_script_path = Path(__file__).parent / "model_inference.py"
                    cmd = [
                        sys.executable,
                        str(inference_script_path),
                        str(args.onnx),
                        str(cex_input)
                    ]
                    inference_result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    output_line = inference_result.stdout.strip() if inference_result.stdout else ""
                    output_vector = parse_vector_from_file_content(output_line)
                    if inference_result.returncode != 0:
                        print(f"  ERROR running inference script: {inference_result.stderr}", file=sys.stderr)

                print(f"  Found an UNSAFE point missed by the explanation:")
                print(f"  Input:  " + ", ".join(f"X_{i}={v:.6f}" for i, v in enumerate(cex_input)))
                if output_vector is not None:
                    print(f"  Output: [" + ", ".join(f"{v:.6f}" for v in output_vector) + "]")

                collected_inputs.append(cex_input)
                if output_vector is not None:
                    collected_outputs.append(output_vector)

                if not args.batch_refine:
                    if args.skip_llm:
                        print("\n LLM SKIPPED ")
                        print("Skipping LLM per --skip-llm; continuing scan/collection.")
                        continue
                    print("\n Refinement Step")
                    print("Asking Gemini to refine the explanation...")
                    try:
                        base_adv_lines = args.adv_inputs.read_text(encoding='utf-8').splitlines()
                        refined_text = call_gemini_to_refine_bounds(
                            args.api_key, args.gemini_model, base_adv_lines, 
                            cex_input, explanation_text,
                            max_adv_lines=args.max_adv_lines, max_retries=args.max_retries, initial_backoff=args.initial_backoff
                        )
                        
                        next_explanation_path = args.output_dir / f"New Bounds {iteration + 1}.txt"
                        next_explanation_path.write_text(refined_text + '\n', encoding='utf-8')
                        print(f"Wrote refined explanation to: {next_explanation_path}")
                        current_explanation_path = next_explanation_path
                    except Exception as e:
                        print(f"ERROR: Gemini refinement call failed: {e}", file=sys.stderr)
                        sys.exit("Stopping loop due to LLM error.")
                    
                    break  # Break inner loop to start new iteration
            elif cex_input is not None:
                # Length mismatch
                print(f"  Warning: parsed vector length {len(cex_input)} does not match expected {expected_input_len}; ignoring.")
            
            time.sleep(1)

        if (not args.batch_refine) and args.skip_llm and collected_inputs:
            # Write collected pairs even when not batching, if LLM is skipped
            io_path = args.io_log_path if args.io_log_path else (args.output_dir / f"unsafe_io_pairs_iter{iteration}.txt")
            with io_path.open('w', encoding='utf-8') as f:
                for idx, inp in enumerate(collected_inputs):
                    f.write(f"Input: {inp}\n")
                    if idx < len(collected_outputs):
                        f.write(f"Output: {collected_outputs[idx]}\n")
                    f.write("---\n")
            print(f"Wrote unsafe input-output pairs to: {io_path}")

        if args.batch_refine and collected_inputs:
            # Write all collected io pairs to a file
            io_path = args.io_log_path if args.io_log_path else (args.output_dir / f"unsafe_io_pairs_iter{iteration}.txt")
            with io_path.open('w', encoding='utf-8') as f:
                for idx, inp in enumerate(collected_inputs):
                    f.write(f"Input: {inp}\n")
                    if idx < len(collected_outputs):
                        f.write(f"Output: {collected_outputs[idx]}\n")
                    f.write("---\n")
            print(f"Wrote unsafe input-output pairs to: {io_path}")

            if args.skip_llm:
                print("\n LLM SKIPPED")
                print("Skipping LLM per --skip-llm; refined explanation will not be generated in this run.")
            else:
                print("\n Refinement Step ")
                print("Asking Gemini to refine the explanation with all collected unsafe points")
                try:
                    base_adv_lines = args.adv_inputs.read_text(encoding='utf-8').splitlines()
                    refined_text = call_gemini_to_refine_bounds_batch(
                        args.api_key, args.gemini_model, base_adv_lines,
                        collected_inputs, explanation_text,
                        outputs=collected_outputs if args.include_outputs_in_prompt else None,
                        include_outputs=args.include_outputs_in_prompt,
                        max_adv_lines=args.max_adv_lines, max_retries=args.max_retries, initial_backoff=args.initial_backoff
                    )
                    next_explanation_path = args.output_dir / f"New Bounds {iteration + 1}.txt"
                    next_explanation_path.write_text(refined_text + '\n', encoding='utf-8')
                    print(f"Wrote refined explanation to: {next_explanation_path}")
                    current_explanation_path = next_explanation_path
                except Exception as e:
                    print(f"ERROR: Gemini refinement call failed: {e}", file=sys.stderr)
                    sys.exit("Stopping loop due to LLM error.")

        if not counterexample_found_this_iteration:
            print("\nSUCCESS")
            print("No counterexamples found in any gaps. The current explanation is robust.")
            break

    print("\nDone.")

if __name__ == '__main__':
    main()

