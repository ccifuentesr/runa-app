"""Reference data from Cifuentes et al. (2020), Table 6."""

import csv
from pathlib import Path


DATA_FILE = Path(__file__).with_name("data") / "cifuentes20_table6.csv"


def load_table6():
    """Load the published average parameters for K5 V to L2 dwarfs."""
    with DATA_FILE.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))
