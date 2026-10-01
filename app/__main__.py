"""Project commands."""

from __future__ import annotations

import argparse

from app.doctor import run_doctor


def main() -> int:
    parser = argparse.ArgumentParser(description="Local character video tools")
    command = parser.add_subparsers(dest="command", required=True)
    doctor = command.add_parser("doctor", help="Check setup and feasibility prerequisites")
    doctor.add_argument("--json", action="store_true", help="Print structured results")
    args = parser.parse_args()
    if args.command == "doctor":
        return run_doctor(as_json=args.json)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
