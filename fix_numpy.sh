#!/bin/bash

# -----------------------------------------------
# NumPy Fix Script for EGT Framework
# -----------------------------------------------
# This script fixes the NumPy version compatibility issue
# with PyTorch in the EGT Framework
# -----------------------------------------------

echo "=== NumPy Fix Script for EGT Framework ==="
echo "This script will downgrade NumPy to a version compatible with PyTorch."

# Check current NumPy version
NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)
if [[ $? -eq 0 ]]; then
    echo "Current NumPy version: $NUMPY_VERSION"
    
    # Check if already compatible
    if [[ $(echo "$NUMPY_VERSION" | cut -d. -f1) -lt 2 ]]; then
        echo "NumPy version is already compatible (< 2.0). No action needed."
        python -c 'import torch; print(f"PyTorch version: {torch.__version__}")' 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "PyTorch is working correctly."
            exit 0
        fi
    fi
else
    echo "NumPy not found or not working correctly."
fi

# Uninstall current NumPy
echo "Removing current NumPy version..."
pip uninstall -y numpy

# Install compatible NumPy
echo "Installing NumPy 1.24.3 (last 1.x version)..."
pip install numpy==1.24.3

# Verify installation
echo "Verifying installation..."
NUMPY_VERSION=$(python -c 'import numpy; print(numpy.__version__)')
echo "NumPy version: $NUMPY_VERSION"

# Test PyTorch compatibility
echo "Testing PyTorch compatibility..."
python -c 'import torch; print(f"PyTorch version: {torch.__version__}")' 2>/dev/null

if [ $? -eq 0 ]; then
    echo "SUCCESS: PyTorch and NumPy are now compatible!"
    echo "You can now run the framework using: ./run.sh"
else
    echo "ERROR: There are still compatibility issues. Please contact the developers."
fi 