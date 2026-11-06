# Ensures `import mewcode` works when running pytest from the repo root
# without installing the package first. (With `pip install -e .` this is
# redundant but harmless.)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
