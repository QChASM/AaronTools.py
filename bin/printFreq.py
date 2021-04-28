#! /usr/bin/env python3
from AaronTools.fileIO import FileReader


def main(args):
    fr = FileReader(args.filename, just_geom=False)
    if args.show:
        s = "frequency\t"
        for x in args.show:
            if x == "vector":
                continue
            s += "%s\t" % x
        print(s)
    for i, data in enumerate(sorted(fr.other["frequency"].data, key=lambda x: x.frequency)):
        if args.type == "neg" and data.frequency > 0:
            continue
        if args.type == "pos" and data.frequency < 0:
            continue
        if isinstance(args.type, int) and i + 1 > args.type:
            break
        s = "%9.4f\t" % data.frequency
        for x in args.show:
            if x == "vector":
                continue
            val = getattr(data, x)
            if isinstance(val, float):
                s += "%9.4f\t" % val
            else:
                s += "%s\t" % str(val)
        
        print(s)
        
        if "vector" in args.show:
            print(data.vector)
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prints frequencies from computational output file",
        formatter_class=argparse.RawTextHelpFormatter,
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
        help="Specify what additional information to show\n"
        "Some info may not be available for certain file formats",
        choices=["intensity", "vector", "forcek", "symmetry"],
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
