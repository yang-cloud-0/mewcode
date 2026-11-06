# 保证在仓库根目录直接跑 pytest 时 `import mewcode` 能成功，
# 无需先安装本包。（即使已 `pip install -e .`，保留它也无害。）
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
