"""Entry point kept for the Qt app (``AppConstants::AIProcessor::trainScript()``).

The desktop app launches ``python TrainModel.py --yes [--det|--seg|--track]``.
The actual implementation now lives in the :mod:`cvtrain` package next to this
file; run ``python TrainModel.py --help`` for the full CLI.
"""

from __future__ import annotations

import sys

from cvtrain.cli import main

if __name__ == "__main__":
    sys.exit(main())
