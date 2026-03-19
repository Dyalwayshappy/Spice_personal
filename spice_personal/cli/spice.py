from __future__ import annotations

import sys

from spice.entry.cli import main as spice_core_main
from spice_personal.cli.main import main as spice_personal_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if args and args[0] == "personal":
        return int(spice_personal_main(args[1:]))
    return int(spice_core_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
