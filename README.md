# NNENUM – Non-Docker setup and usage

The `Dockerfile` shows one way to prepare the environment. If you prefer a native (non-Docker) setup, follow the instructions below.

This tool loads neural networks from ONNX files and properties from `vnnlib` files, then searches for counterexamples. The main entry point is `src/nnenum/nnenum.py`.

## Prerequisites
- OS: Windows, macOS, or Linux
- Python: 3.9 or 3.10 recommended (older/newer versions may not match the pinned wheels in requirements.txt)
- pip 21+ and a working C toolchain if your platform needs to build wheels (Linux)
- GLPK runtime library for `swiglpk` (often bundled on Windows/macOS wheels; Linux usually needs the system library)

## 1) Create and activate a virtual environment

---------------------------------------------------------------------------------

FRESH START - Remove existing venv and create new one:

1. Deactivate current venv (if active):
   deactivate

2. Remove the venv folder:
  rmdir -r venv

3. Create new virtual environment:

Make virtual environment using Python 3.10.11

py -3.10 -m venv venv
----------------------------------------

Temporarily allow script execution for this session
Run this command in your PowerShell before activating your venv:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

Then activate your venv:

venv\Scripts\activate

python -m pip install --upgrade pip

----------------------------------------


Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip/wheel (recommended):
```
pip install --upgrade pip wheel setuptools
```

## 2) Install Python dependencies

From the repository root (this directory):
```
pip install -r requirements.txt
```

If installation fails on Linux while building SciPy, ensure build tools are installed, for example on Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install -y python3-dev build-essential gfortran libatlas-base-dev
```
Then retry `pip install -r requirements.txt`.

## 3) Ensure GLPK is available for swiglpk
The package `swiglpk` requires a GLPK shared library. This is frequently bundled on Windows and macOS wheels, but on Linux it is typically provided by the OS.

- Ubuntu/Debian:
  - `sudo apt-get install -y libglpk40` (or `libglpk-dev` depending on your distro)
- Fedora/CentOS/RHEL:
  - `sudo dnf install -y glpk glpk-devel`
- macOS (Homebrew):
  - `brew install glpk`
- Windows:
  - In many cases `pip install swiglpk` just works because the wheel bundles GLPK. If you still see an import error for GLPK DLLs, either:
    - Use conda to install the runtime: `conda install -c conda-forge glpk`, or
    - Download a GLPK Windows build and add its `bin` directory (containing `glpk*.dll`) to your PATH, then restart your shell.

Quick verification (should print `ok` and exit without error):
```
python -c "import onnx, onnxruntime, swiglpk; print('ok')"
```

## 4) Run on the provided example

You can run the verifier directly via the module in `src/` (no need to modify PYTHONPATH if you call the script via its path):
```
python src/nnenum/nnenum.py examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib
```

Command-line usage:
```
python src/nnenum/nnenum.py <onnx_file> <vnnlib_file> [timeout=None] [outfile=None] [processes=<auto>] [settings=auto|control|image|exact]
```

Examples:
- 60 second timeout, write result to file:
```
python src/nnenum/nnenum.py examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib 60 result.txt
```
- Explicit parallelism and preset:
```
python src/nnenum/nnenum.py examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib 120 result.txt 8 control
```
normal run:

$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"; $env:PYTHONPATH=(Resolve-Path ./src).Path; python -m nnenum.nnenum examples/acasxu/data/ACASXU_run2a_2_9_batch_2000.onnx examples/acasxu/data/prop_8.vnnlib 60 result.txt

The tool prints or writes one of: `holds` (safe), `violated` (unsafe), or `timeout`.

## Notes on presets
- `auto` chooses settings based on input dimension count
- `control` favors smaller control-style networks
- `image` favors larger image benchmarks
- `exact` disables over-approximation (slower, exhaustive search)

## Troubleshooting
- ImportError for `swiglpk` or missing GLPK DLL/SO:
  - Ensure the GLPK runtime is installed and on your PATH/loader path (see step 3)
  - On Linux, verify `ldconfig -p | grep glpk` lists a GLPK library
  - On macOS, if Homebrew installed it, `otool -L $(python -c "import swiglpk, sys, inspect; import os; print(os.path.dirname(inspect.getfile(swiglpk)))")/swiglpk*.so` can help diagnose linkage
- SciPy or NumPy wheel not found for your Python version:
  - Use Python 3.9 or 3.10
  - Upgrade pip and wheel, then retry
- onnxruntime errors:
  - Reinstall with `pip install --force-reinstall onnxruntime==1.12.1`

## Optional: set PYTHONPATH instead of using script path
If you prefer calling via module syntax `python -m nnenum.nnenum`, set `PYTHONPATH` to include `src/`:

Windows (PowerShell):
```
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -m nnenum.nnenum examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib
```

macOS/Linux:
```
export PYTHONPATH=$(pwd)/src
python -m nnenum.nnenum examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib
```

## License
See [LICENSE](LICENSE).
