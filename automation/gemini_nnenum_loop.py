#!/usr/bin/env python3
"""
Automation loop to:
 1) Read structured bounds from an explanation text file.
 2) Append the logical NEGATION of each bound as vnnlib assertions to a property file (by creating a temporary variant).
 3) Run nnenum against the modified property and parse counterexamples (adversarial inputs).
 4) If a counterexample is discovered, call Gemini to produce a new bounds file ("New Bounds <i>.txt"), then repeat.

Notes/Assumptions:
- The explanation file must contain lines in one of these formats:
    Independent bounds:
      X_1 belongs to [1,2]
      X_2 belongs to [2,3] OR [1,2]

    Dependent bounds (single instance per line):
      X_2 belongs to [2,5] when X_1 belongs to [1,3]
      X_4 belongs to [5,6] when X_1 belongs to [1,2] AND X_2 belongs to [2,3]

- Negation semantics implemented:
  * Independent union of intervals S = [a1,b1] OR [a2,b2] OR ...
    NOT(x in S) == (x < a1 OR x > b1) AND (x < a2 OR x > b2) AND ...
    In vnnlib (which uses >= and <=), we encode this as a conjunction of ORs:
      (assert (and
        (or (<= X_i a1) (>= X_i b1))
        (or (<= X_i a2) (>= X_i b2))
        ...))

  * Dependent: "X_k in S when A" (A is conjunction of simple bounds on variables)
    We conjoin antecedent A with NOT(consequent):
      (assert (and
        A
        (and (or (<= X_k a1) (>= X_k b1)) (or (<= X_k a2) (>= X_k b2)) ...)
      ))

- Each explanation line is treated as an independent verification instance. For each line, we create a modified copy
  of the base vnnlib with the negated assertion for that line, run nnenum, and check for a counterexample.

- This script calls nnenum via the repository path: "python src/nnenum/nnenum.py ..." to avoid PYTHONPATH issues.

- Gemini model: defaults to "gemini-1.5-pro"; override via --gemini-model. You must install google-generativeai:
    pip install google-generativeai

Usage example (PowerShell from repo root):
  python automation/gemini_nnenum_loop.py \
    --api-key YOUR_GEMINI_API_KEY \
    --onnx examples/acasxu/data/ACASXU_run2a_2_9_batch_2000.onnx \
    --base-vnnlib examples/acasxu/data/prop_8.vnnlib \
    --explanation "Original Bounds.txt" \
    --adv-inputs "examples/acasxu/data/adv_inputs.txt" \
    --iterations 3 --timeout 60

This will generate per-iteration outputs under automation_output/.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Optional Gemini import; error handled at call time
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

# ---------------------- Parsing utilities ----------------------
Interval = Tuple[float, float]

INDEP_RE = re.compile(r"^\s*X_(?P<var>\d+)\s+belongs\s+to\s+(?P<intervals>.+?)\s*$", re.IGNORECASE)
DEP_RE = re.compile(r"^\s*X_(?P<target>\d+)\s+belongs\s+to\s+(?P<cons>.+?)\s+when\s+(?P<ants>.+?)\s*$", re.IGNORECASE)
INTERVAL_RE = re.compile(r"\[(?P<a>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[\s,]+(?P<b>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]")
COND_RE = re.compile(r"\s*X_(?P<var>\d+)\s+belongs\s+to\s+(?P<intervals>.+?)\s*$", re.IGNORECASE)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAK-MJhlXpr1U_lBsnryLjsTwkuXwaAk-8")


def parse_intervals(text: str) -> List[Interval]:
    """Parse intervals from a string like "[a,b] OR [c,d] OR ..."""
    intervals: List[Interval] = []
    for m in INTERVAL_RE.finditer(text):
        a = float(m.group("a"))
        b = float(m.group("b"))
        if a > b:
            a, b = b, a
        intervals.append((a, b))
    return intervals


def parse_line(line: str) -> Optional[Dict]:
    """Return a parsed structure for a line or None if not matched.
    Structures:
      { 'type': 'indep', 'var': int, 'intervals': List[Interval], 'raw': str }
      { 'type': 'dep', 'target': int, 'cons_intervals': List[Interval],
        'antecedents': List[ {'var': int, 'intervals': List[Interval]} ], 'raw': str }
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith(';'):
        return None

    m = DEP_RE.match(line)
    if m:
        target = int(m.group('target'))
        cons_intervals = parse_intervals(m.group('cons'))
        ants_text = m.group('ants')
        # split antecedents on AND (case-insensitive)
        parts = re.split(r"\s+AND\s+", ants_text, flags=re.IGNORECASE)
        antecedents = []
        for p in parts:
            cm = COND_RE.match(p.strip())
            if not cm:
                raise ValueError(f"Cannot parse antecedent condition: '{p}' in line: '{line}'")
            v = int(cm.group('var'))
            ivals = parse_intervals(cm.group('intervals'))
            if not ivals:
                raise ValueError(f"No intervals parsed from antecedent: '{p}' in line: '{line}'")
            antecedents.append({'var': v, 'intervals': ivals})
        if not cons_intervals:
            raise ValueError(f"No consequent intervals parsed in line: '{line}'")
        return {
            'type': 'dep',
            'target': target,
            'cons_intervals': cons_intervals,
            'antecedents': antecedents,
            'raw': line,
        }

    m = INDEP_RE.match(line)
    if m:
        var = int(m.group('var'))
        ivals = parse_intervals(m.group('intervals'))
        if not ivals:
            raise ValueError(f"No intervals parsed in independent bound line: '{line}'")
        return {
            'type': 'indep',
            'var': var,
            'intervals': ivals,
            'raw': line,
        }

    return None


# ---------------------- vnnlib generation ----------------------

def vnnlib_or_outside(var_index: int, interval: Interval) -> str:
    a, b = interval
    # (or (<= X_i a) (>= X_i b))
    return f"(or (<= X_{var_index} {a}) (>= X_{var_index} {b}))"


def vnnlib_and_inside(var_index: int, interval: Interval) -> str:
    a, b = interval
    # (and (>= X_i a) (<= X_i b))
    return f"(and (>= X_{var_index} {a}) (<= X_{var_index} {b}))"


def build_negation_assert(parsed: Dict) -> str:
    """Build a vnnlib (assert ...) block expressing the negation of the parsed statement.
    See docstring notes for logic.
    """
    if parsed['type'] == 'indep':
        var = parsed['var']
        ors = [vnnlib_or_outside(var, ival) for ival in parsed['intervals']]
        # Conjunction of ORs is the negation of union membership
        body = "\n        ".join(f"{o}" for o in ors)
        return (
            "; Negation of: " + parsed['raw'] + "\n"
            "(assert (and\n"
            f"        {body}\n"
            "))"
        )

    elif parsed['type'] == 'dep':
        target = parsed['target']
        ant_and_parts: List[str] = []
        for ant in parsed['antecedents']:
            v = ant['var']
            # antecedent is disallowing OR union? No, antecedent stays positive membership: conjunct of unions => we encode as (and (and ...)) requiring variable to be inside at least one of the OR intervals.
            if len(ant['intervals']) == 1:
                ant_and_parts.append(vnnlib_and_inside(v, ant['intervals'][0]))
            else:
                # Membership in union: (or (and inside i1) (and inside i2) ...)
                ors_inside = " ".join(vnnlib_and_inside(v, iv) for iv in ant['intervals'])
                ant_and_parts.append(f"(or {ors_inside})")
        antecedent_expr = "\n        ".join(ant_and_parts) if ant_and_parts else ""

        cons_neg_ors = [vnnlib_or_outside(target, ival) for ival in parsed['cons_intervals']]
        cons_neg_body = "\n            ".join(f"{o}" for o in cons_neg_ors)

        return (
            "; Negation of: " + parsed['raw'] + "\n"
            "(assert (and\n"
            f"        {antecedent_expr}\n"
            "        (and\n"
            f"            {cons_neg_body}\n"
            "        )\n"
            "))"
        )

    raise ValueError("Unknown parsed type")


# ---------------------- nnenum execution ----------------------

def run_nnenum(repo_root: Path, onnx_path: Path, vnnlib_path: Path, timeout: int, outfile: Path) -> str:
    """Run nnenum via PowerShell with env vars and return combined stdout+stderr as text.
    Mirrors manual usage:
    $env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"; \
    $env:PYTHONPATH=(Resolve-Path ./src).Path; python -m nnenum.nnenum <onnx> <vnnlib> <timeout> <outfile>
    """
    ps_cmd = (
        '$env:OPENBLAS_NUM_THREADS="1"; '
        '$env:OMP_NUM_THREADS="1"; '
        '$env:MKL_NUM_THREADS="1"; '
        '$env:NUMEXPR_NUM_THREADS="1"; '
        '$env:PYTHONPATH=(Resolve-Path ./src).Path; '
        f'python -m nnenum.nnenum "{onnx_path}" "{vnnlib_path}" {timeout} "{outfile}"'
    )
    res = subprocess.run([
        'powershell', '-NoProfile', '-Command', ps_cmd
    ], cwd=str(repo_root), capture_output=True, text=True)
    return (res.stdout or '') + '\n' + (res.stderr or '')


def parse_cex_from_output(text: str) -> Optional[List[float]]:
    """Parse a concrete counterexample input vector from nnenum output.
    Looks for a line like: Input: [v0, v1, ...]
    Returns list of floats or None if not found/parsed.
    """
    # Accept various spacing
    m = re.search(r"Input:\s*\[([^\]]+)\]", text)
    if not m:
        return None
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(',')]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            return None
    return vals


def parse_adv_inputs_file(path: Path) -> List[str]:
    """Read adversarial inputs text file as raw lines to include in LLM prompt.
    Skips empty lines and lines starting with '#' or ';'. The format is free-form; lines are passed to the LLM as-is.
    """
    lines: List[str] = []
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith('#') or s.startswith(';'):
            continue
        lines.append(s)
    return lines

def comment_out_input_asserts_text(text: str) -> str:
    """Comment out any (assert ...) statements that reference input variables X_.
    Preserves formatting by prefixing each line in the matched statement with '; '.
    """
    out_lines: List[str] = []
    buf: List[str] = []
    in_assert = False
    depth = 0

    def paren_delta(s: str) -> int:
        return s.count('(') - s.count(')')

    for line in text.splitlines():
        # split off inline comments for counting
        code = line.split(';', 1)[0]
        if not in_assert:
            if '(assert' in code:
                in_assert = True
                depth = paren_delta(code)
                buf = [line]
                if depth <= 0:
                    # single-line assert; decide and flush
                    joined_code = "\n".join(l.split(';', 1)[0] for l in buf)
                    if 'X_' in joined_code:
                        for l in buf:
                            out_lines.append(l if l.lstrip().startswith(';') else '; ' + l)
                    else:
                        out_lines.extend(buf)
                    buf = []
                    in_assert = False
            else:
                out_lines.append(line)
        else:
            buf.append(line)
            depth += paren_delta(code)
            if depth <= 0:
                # end of assert
                joined_code = "\n".join(l.split(';', 1)[0] for l in buf)
                if 'X_' in joined_code:
                    for l in buf:
                        out_lines.append(l if l.lstrip().startswith(';') else '; ' + l)
                else:
                    out_lines.extend(buf)
                buf = []
                in_assert = False

    # flush any remainder conservatively
    if buf:
        joined_code = "\n".join(l.split(';', 1)[0] for l in buf)
        if 'X_' in joined_code:
            for l in buf:
                out_lines.append(l if l.lstrip().startswith(';') else '; ' + l)
        else:
            out_lines.extend(buf)

    return "\n".join(out_lines)

# ---------------------- Gemini call ----------------------

def ensure_gemini(api_key: Optional[str]):
    if genai is None:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")
    if not api_key:
        raise RuntimeError("Gemini API key missing. Pass --api-key or set GEMINI_API_KEY env var.")
    genai.configure(api_key=api_key)


def call_gemini_to_build_negated_assert(api_key: str, model_name: str, bounds_text: str) -> str:
    """Ask Gemini to convert the provided bounds text into a single vnnlib (assert ...) block
    that is the logical negation of all the bounds. The returned text must be:
    - Exactly one (assert ...) s-expression
    - Only uses <= and >= and variables X_i for inputs
    - No comments, no (declare-const ...) lines
    - For independent bounds: negate membership in a union: S=[a,b] OR [c,d] -> (and (or (<= X a) (>= X b)) (or (<= X c) (>= X d)))
    - For dependent bounds of the form "X_k belongs to S when (antecedent)": produce (and antecedent NOT(X_k in S))
    - Combine multiple rules with an (or ...): each rule contributes one conjunct (as defined above) to the OR
    """
    ensure_gemini(api_key)
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are given a set of bounds written in the following textual style. Convert them into a SINGLE vnnlib assertion "
        "that represents the NEGATION of all those bounds.\n\n"
        "Rules for conversion:\n"
        "1) Use only one s-expression that begins with (assert ...).\n"
        "2) Do not include variable declarations or comments.\n"
        "3) Only use operators <= and >=.\n"
        "4) For an independent bound like: X_2 belongs to [2,5] OR [7,9]\n"
        "   Negation is: (or (and (>= X_2 LOWER) (<= X_2 2)) (and (>= X_2 5) (<= X_2 UPPER))). Use sufficiently large constants for LOWER/UPPER (or the known domain bounds).\n"
        "5) For a dependent bound like: X_4 belongs to [5,6] when X_1 belongs to [1,2] AND X_2 belongs to [2,3]\n"
        "   Antecedent stays positive membership (use (and (>= X_i a) (<= X_i b)) or (or ...) for unions), and the consequent is negated as in rule 4.\n"
        "   The rule becomes: (and <antecedent_membership> (or (and (>= X_k LOWER) (<= X_k a)) (and (>= X_k b) (<= X_k UPPER)))).\n"
        "6) Combine all rules via a single OR across rules: (assert (or <rule1_expr> <rule2_expr> ...)).\n"
        "7) Output the assertion only. No extra text.\n\n"
        "Bounds text follows:\n\n" + bounds_text.strip()
    )

    resp = model.generate_content(prompt)
    text = getattr(resp, 'text', None)
    if not text:
        try:
            text = "\n".join([p.text for c in resp.candidates for p in c.content.parts])
        except Exception:
            raise RuntimeError("Gemini response had no text content for negated assertion")

    text = text.strip()
    if not text.startswith("(assert"):
        raise RuntimeError("Gemini negated assertion did not start with (assert")
    return text


def call_gemini_to_extract_bounds(
    api_key: str,
    model_name: str,
    base_adv_lines: List[str],
    new_counterexamples: List[List[float]],
    prev_bounds_text: Optional[str],
) -> str:
    """Call Gemini with the structured prompt, including:
    - Previous bounds (for refinement)
    - Original adversarial inputs (from file), passed as-is per line
    - New counterexamples found by nnenum, formatted as X_i=value
    Returns the generated, refined explanation text (expected to be line-based bounds per your format).
    """
    ensure_gemini(api_key)
    model = genai.GenerativeModel(model_name)

    header = (
        "Refine the previous bounds below using the provided adversarial inputs and newly found counterexamples. "
        "Each independent bound must be on its own line. Use exactly this format (no extra commentary):\n"
        "X_1 belongs to [1,2]\n"
        "X_2 belongs to [2,3] OR [1,2]\n\n"
        "For dependent bounds, put one instance per line in this format:\n"
        "X_2 belongs to [2,5] when X_1 belongs to [1,3]\n"
        "X_4 belongs to [5,6] when X_1 belongs to [1,2] AND X_2 belongs to [2,3]\n\n"
        "ONLY output lines in this exact format. Do not include any commentary."
    )

    prev = (prev_bounds_text or "").strip()

    orig_adv_section = ["Original adversarial inputs (from file):"]
    orig_adv_section.extend(base_adv_lines)

    cex_section = ["New counterexamples found so far (assignments per line):"]
    for vec in new_counterexamples:
        assignments = ", ".join(f"X_{i}={v}" for i, v in enumerate(vec))
        cex_section.append(assignments)

    parts = [header]
    if prev:
        parts.append("Previous bounds:\n" + prev)
    parts.append("\n".join(orig_adv_section))
    parts.append("\n".join(cex_section))

    full_prompt = "\n\n".join(parts)

    resp = model.generate_content(full_prompt)
    text = getattr(resp, 'text', None)
    if not text:
        try:
            text = "\n".join([p.text for c in resp.candidates for p in c.content.parts])
        except Exception:
            raise RuntimeError("Gemini response had no text content")
    return text.strip()


# ---------------------- Main loop ----------------------

def main():
    parser = argparse.ArgumentParser(description="Automate LLM-vs-nnenum loop with negated bounds injected into vnnlib.")
    parser.add_argument('--api-key', default=os.getenv('GEMINI_API_KEY'), help='Gemini API key (or set GEMINI_API_KEY env var).')
    parser.add_argument('--gemini-model', default='gemini-2.5-pro', help='Gemini model name (default: gemini-2.5-pro).')
    parser.add_argument('--onnx', required=True, help='Path to ONNX network file.')
    parser.add_argument('--base-vnnlib', required=True, help='Path to base vnnlib property file to copy and augment.')
    parser.add_argument('--explanation', required=True, help='Path to the initial explanation file (Original Explaination.txt).')
    parser.add_argument('--adv-inputs', required=True, help='Path to the adversarial inputs text file (lines are passed to LLM).')
    parser.add_argument('--timeout', type=int, default=60, help='nnenum timeout seconds (default: 60).')
    parser.add_argument('--iterations', type=int, default=3, help='Max loop iterations (default: 3).')
    parser.add_argument('--output-dir', default='automation_output', help='Directory to store generated files.')
    parser.add_argument('--stop-on-first-success', action='store_true', help='Stop per-iteration after the first line that yields a counterexample (default: try all lines).')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    onnx_path = Path(args.onnx).resolve()
    base_vnnlib = Path(args.base_vnnlib).resolve()
    explanation_path = Path(args.explanation).resolve()
    adv_inputs_path = Path(args.adv_inputs).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)
    if not base_vnnlib.exists():
        print(f"ERROR: base vnnlib not found: {base_vnnlib}", file=sys.stderr)
        sys.exit(1)
    if not adv_inputs_path.exists():
        print(f"ERROR: adversarial inputs file not found: {adv_inputs_path}", file=sys.stderr)
        sys.exit(1)

    adversarial_inputs_all: List[List[float]] = []
    base_adv_lines: List[str] = parse_adv_inputs_file(adv_inputs_path)

    # For iterations >= 1, explanation_path updates to New Bounds i.txt in output dir
    for iteration in range(args.iterations):
        if iteration == 0:
            cur_explanation = explanation_path
        else:
            cur_explanation = out_dir / f"New Bounds {iteration}.txt"

        if not cur_explanation.exists():
            if iteration == 0:
                print(f"ERROR: Initial explanation file not found: {cur_explanation}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Terminating: expected explanation for iteration {iteration} not found: {cur_explanation}")
                break

        # Read entire bounds text
        bounds_text = cur_explanation.read_text(encoding='utf-8')

        # Ask Gemini to produce a single vnnlib (assert ...) block representing the negation of all rules
        try:
            neg_block = call_gemini_to_build_negated_assert(args.api_key, args.gemini_model, bounds_text)
        except Exception as ex:
            print(f"Gemini negation build failed: {ex}", file=sys.stderr)
            print("Stopping loop due to LLM error.")
            break

        # Build temp vnnlib by copying base and appending our negated assertion block
        tmp_vnnlib = out_dir / f"negated_iter{iteration}.vnnlib"
        base_text = base_vnnlib.read_text(encoding='utf-8')
        processed = comment_out_input_asserts_text(base_text)
        tmp_vnnlib.write_text(processed + '\n', encoding='utf-8')
        with tmp_vnnlib.open('a', encoding='utf-8') as f:
            f.write('\n\n; ----------------- Auto-appended NEGATED assertion from explanation -----------------\n')
            f.write(neg_block.strip() + '\n')
            f.write('; ---------------------------------------------------------------------------\n')

        # Run nnenum once on this modified property
        print(f"[iter={iteration}] Running nnenum -> {tmp_vnnlib.name}")
        outfile_path = out_dir / f"result_iter{iteration}.txt"
        output_text = run_nnenum(repo_root, onnx_path, tmp_vnnlib, args.timeout, outfile_path)
        (out_dir / f"run_iter{iteration}.txt").write_text(output_text, encoding='utf-8')

        cex = parse_cex_from_output(output_text)
        if cex is None:
            print(f"No confirmed counterexample parsed from output in iteration {iteration}. Stopping loop.")
            break

        print(f"  -> Found counterexample input: {cex}")
        adversarial_inputs_all.append(cex)

        # Produce next explanation with Gemini
        try:
            prev_bounds_text = cur_explanation.read_text(encoding='utf-8')
            new_text = call_gemini_to_extract_bounds(
                args.api_key,
                args.gemini_model,
                base_adv_lines,
                adversarial_inputs_all,
                prev_bounds_text,
            )
        except Exception as ex:
            print(f"Gemini call failed: {ex}", file=sys.stderr)
            print("Stopping loop due to LLM error.")
            break

        next_file = out_dir / f"New Bounds {iteration+1}.txt"
        next_file.write_text(new_text + '\n', encoding='utf-8')
        print(f"Wrote: {next_file}")

    print("Done.")


if __name__ == '__main__':
    main()
