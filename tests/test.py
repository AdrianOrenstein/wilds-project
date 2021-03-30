import pathlib
import sys

import pytest

if __name__ == "__main__":
    dir_path = pathlib.Path(__file__).resolve().parent
    sys.exit(pytest.main([str(dir_path), "-W", "ignore::DeprecationWarning"]))
