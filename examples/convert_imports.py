#!/usr/bin/env python3
"""
Convert relative imports to absolute imports in all Python files.
This script helps prepare the codebase for being a standalone repository.
"""

import os
import re
import sys

def convert_file(filepath):
    """Convert relative imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace relative imports like "from ..mechanisms.reward import X" 
    # with "from mechanisms.reward import X"
    pattern = r'from \.\.([\w\.]+) import'
    replacement = r'from \1 import'
    new_content = re.sub(pattern, replacement, content)
    
    # Replace relative imports like "from . import X" 
    # with appropriate absolute imports
    pattern = r'from \. import ([\w, ]+)'
    
    # Function to determine the correct absolute import path
    def get_absolute_import(match):
        module_name = os.path.basename(os.path.dirname(filepath))
        imported_items = match.group(1)
        return f'from {module_name} import {imported_items}'
    
    new_content = re.sub(pattern, get_absolute_import, new_content)
    
    # Write back if changes were made
    if new_content != content:
        print(f"Converting imports in {filepath}")
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

def process_directory(directory):
    """Process all Python files in a directory recursively."""
    changed_files = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != 'convert_imports.py':
                filepath = os.path.join(root, file)
                if convert_file(filepath):
                    changed_files += 1
    return changed_files

if __name__ == "__main__":
    # Get the root directory (parent of the examples directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    print(f"Converting imports in {root_dir}...")
    changed_files = process_directory(root_dir)
    print(f"Converted imports in {changed_files} files.") 