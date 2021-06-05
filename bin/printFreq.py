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
    freq = fr.other["frequency"]
    if not any((args.fundamentals, args.overtones, args.combinations)):
        for i, data in enumerate(sorted(freq.data, key=lambda x: x.frequency)):
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
    if freq.anharm_data:
        for i, data in enumerate(sorted(freq.anharm_data, key=lambda x: x.frequency)):
            if args.fundamentals:
                s = "%9.4f\t" % data.frequency
                for x in args.show:
                    if not hasattr(data, x):
                        continue
                    val = getattr(data, x)
                    if isinstance(val, float):
                        s += "%9.4f\t" % val
                    else:
                        s += "%s\t" % str(val)
                
                print(s)
            
            if args.overtones:
                for k, overtone in enumerate(data.overtones):
                    s = "%i x %9.4f = %9.4f\t" % (
                        (k + 2), data.frequency, overtone.frequency
                    )
                    for x in args.show:
                        if not hasattr(overtone, x):
                            continue
                        val = getattr(overtone, x)
                        if isinstance(val, float):
                            s += "%9.4f\t" % val
                        else:
                            s += "%s\t" % str(val)
                    
                    print(s)
            
            if args.combinations:
                for key in data.combinations:
                    for combo in data.combinations[key]:
                        s = "%9.4f + %9.4f = %9.4f\t" % (
                            data.frequency,
                            freq.anharm_data[key].frequency,
                            combo.frequency,
                        )
                        for x in args.show:
                            if not hasattr(combo, x):
                                continue
                            val = getattr(combo, x)
                            if isinstance(val, float):
                                s += "%9.4f\t" % val
                            else:
                                s += "%s\t" % str(val)

                        print(s)


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
        choices=["intensity", "vector", "forcek",
        "symmetry", "delta_anh", "harmonic_frequency",
        "harmonic_intensity"],
        default=[],
    )
    parser.add_argument(
        "--fundamentals",
        "-f",
        action="store_true",
        default=False,
        dest="fundamentals",
        help="print anharmonic fundamental frequencies for files with anharmonic data",
    )
    parser.add_argument(
        "--overtone-bands",
        "-ob",
        action="store_true",
        default=False,
        dest="overtones",
        help="print overtone frequencies for files with anharmonic data",
    )
    parser.add_argument(
        "--combination-bands",
        "-cb",
        action="store_true",
        default=False,
        dest="combinations",
        help="print combination frequencies for files with anharmonic data",
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
