"""Enable `python -m vdedup ...`.

This is the most robust invocation: run from the project directory and the
package is importable without any install (and without relying on an editable
.pth, which some conda-based venvs do not process). The `vdedup` console script
is equivalent after `pip install .`.
"""
from .cli import cli

if __name__ == "__main__":
    cli()
