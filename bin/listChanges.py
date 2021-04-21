#!/usr/bin/env python3

from AaronTools.config import Config


def main(args):
    config = Config(args.config)
    default_keys = tuple("/" + k for k in config["DEFAULT"])
    print("Configuration:")
    for key, val in config.as_dict().items():
        if key.endswith(default_keys):
            continue
        if key.startswith(("Substitution", "Mapping")):
            print("  ", key, ": ", "\n\t".join(val.split("\n")), sep="")

    template = "  {:20s} {:20s} {:s}"
    print()
    print("Structures to run:")
    print(template.format("Name", "Type", "Change"))
    for key, val in config._changes.items():
        print(
            template.format(
                key if key else "original", str(val[1]), str(val[0])
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Lists the substitutions and mappings parsed from an AaronJr configuration file"
    )
    parser.add_argument("config", help="The AaronJr configuration file")
    args = parser.parse_args()
    main(args)
