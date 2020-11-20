#! /usr/bin/env python3
from AaronTools.comp_output import CompOutput


def main(args):
    comp = CompOutput(args.filename)
    for i, key in enumerate(sorted(comp.frequency.by_frequency.keys())):
        if args.type == "neg" and key > 0:
            continue
        if args.type == "pos" and key < 0:
            continue
        if isinstance(args.type, int) and i > args.type:
            break
        val = comp.frequency.by_frequency[key]
        show = [val[x] for x in args.show if x != "vector"]
        line = "{:9.4f}\t" + "{}\t" * len(show)
        print(line.format(key, *show))
        if "vector" in args.show:
            print(val["vector"])
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prints frequencies from computational output file"
    )
    parser.add_argument(
        "filename", help="Completed QM output file with frequency info"
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        help="Types of frequencies to print (defaults to `all`. Allowed values: `all`, `neg[ative]`, `pos[itive]`, or `x` to print the first int(x) modes)",
        default="all",
    )
    parser.add_argument(
        "--show",
        "-s",
        type=str,
        nargs="*",
        help="Specify what additional information to show",
        choices=["intensity", "vector"],
        default=[],
    )
    args = parser.parse_args()
    try:
        args.type = int(args.type)
    except ValueError:
        args.type = args.type.lower()
        if args.type not in ["all", "neg", "negative", "pos", "positive"]:
            parser.print_help()
            exit(1)
        elif args.type in ["neg", "negative"]:
            args.type = "neg"
        elif args.type in ["pos", "positive"]:
            args.type = "pos"
    main(args)
