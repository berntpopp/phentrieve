# Allows running the CLI via 'python -m phentrieve'
import sys

from .cli import app

# For Python CLI compatibility, manually check for -h and redirect to --help
if "-h" in sys.argv:
    sys.argv.remove("-h")
    sys.argv.append("--help")

app(prog_name="phentrieve")
