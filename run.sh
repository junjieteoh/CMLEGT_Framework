#!/bin/bash

# -----------------------------------------------
# EGT Framework
# This project REQUIRES Python 3.11
# It will not work with Python 3.13+
# Requires NumPy <2.0 for PyTorch compatibility
# -----------------------------------------------

# Activate the Python 3.11 environment where all dependencies are installed
conda activate fyp-py311

# Check Python version
PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_VERSION" != "3.11" ]; then
    echo "ERROR: Wrong Python version detected: $PY_VERSION"
    echo "This framework requires Python 3.11"
    echo "Please check that the fyp-py311 environment is properly set up"
    exit 1
fi

# Ensure correct NumPy version is installed
echo "Checking NumPy version..."
NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)
if [[ $? -ne 0 || $(echo "$NUMPY_VERSION" | cut -d. -f1) -ge 2 ]]; then
    echo "NumPy 2.x detected or not installed. Installing compatible NumPy version..."
    pip install 'numpy<2.0' -U
fi

# Print environment info
echo "Activated environment: fyp-py311"
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Execute the provided command or launch an interactive shell
if [ $# -gt 0 ]; then
    "$@"
else
    echo "Environment ready. Run your scripts here."
    exec $SHELL
fi 