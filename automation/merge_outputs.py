"""
Merge unsafe_io_pairs_iter*.txt files and create a text file containing only the outputs.

By default, scans the 'automation' directory for files named
'unsafe_io_pairs_iter*.txt', ordered by iteration number, extracts all
"Output: [ ... ]" vectors and writes them line-by-line into 'outputs_only.txt'.

Usage examples:
  python automation/merge_outputs.py
  python automation/merge_outputs.py --dir automation --pattern "unsafe_io_pairs_iter*.txt" --out outputs_only.txt
  python automation/merge_outputs.py --unique
"""
import argparse
import re
from pathlib import Path
from typing import List

OUTPUT_LINE_RE = re.compile(r"^\s*Output:\s*(\[[^\]]*\])\s*$")
ITER_FILE_RE = re.compile(r"unsafe_io_pairs_iter(\d+)\.txt$")


def find_iter_number(path: Path) -> int:
    m = ITER_FILE_RE.search(path.name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def collect_outputs_from_file(path: Path) -> List[str]:
    outputs: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = OUTPUT_LINE_RE.match(line)
            if m:
                outputs.append(m.group(1).strip())
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Merge unsafe_io_pairs_iter*.txt files into a single outputs-only file.")
    parser.add_argument("--dir", default=None, help="Directory to scan for input files. Defaults to this script's directory.")
    parser.add_argument("--pattern", default="unsafe_io_pairs_iter*.txt", help="Glob pattern to match input files.")
    parser.add_argument("--out", default="outputs_only.txt", help="Output filename (created inside --dir unless absolute path).")
    parser.add_argument("--unique", action="store_true", help="Deduplicate identical outputs.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    if args.dir is None:
        base_dir = script_dir
    else:
        dir_path = Path(args.dir)
        base_dir = dir_path if dir_path.is_absolute() else (script_dir / dir_path).resolve()
    files = sorted(base_dir.glob(args.pattern), key=lambda p: (find_iter_number(p), p.name))

    if not files:
        print(f"No files found matching {args.pattern} in {base_dir}")
        return

    all_outputs: List[str] = []
    for f in files:
        outs = collect_outputs_from_file(f)
        if outs:
            all_outputs.extend(outs)
        else:
            pass

    if args.unique:
        seen = set()
        deduped: List[str] = []
        for o in all_outputs:
            if o not in seen:
                deduped.append(o)
                seen.add(o)
        all_outputs = deduped

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for o in all_outputs:
            f.write(o + "\n")

    print(f"Wrote {len(all_outputs)} outputs to: {out_path}")


if __name__ == "__main__":
    main()
