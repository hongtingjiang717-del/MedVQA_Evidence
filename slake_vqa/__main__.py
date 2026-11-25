"""Run a SLAKE experiment from the repository root."""

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "data": "slake_vqa.dataset",
    "train": "slake_vqa.lora",
    "evaluate": "slake_vqa.benchmark",
    "segment": "slake_vqa.segmentation.unet",
    "audit": "slake_vqa.audit",
    "demo": "slake_vqa.web",
    "maira": "slake_vqa.maira",
    "verify": "slake_vqa.verify",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    del sys.argv[1]
    import_module(COMMANDS[args.command]).main()


if __name__ == "__main__":
    main()
