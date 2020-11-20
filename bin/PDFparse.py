#!/usr/bin/env python3
import os
import re

from AaronTools.atoms import Atom
from AaronTools.geometry import Geometry
from pdfminer.high_level import extract_pages


def main(args):
    geom_patt = re.compile("([A-Z][a-z]*)((?:\s+-?\d+\.?\d*){3})")
    float_patt = re.compile("-?\d+\.?\d*")

    all_names = []
    atoms = []
    name = None
    for i, page in enumerate(extract_pages(args.infile)):
        print("parsing page {: 4d} please wait...".format(i + 1), end="\r")
        for element in page:
            last_line = None
            if hasattr(element, "get_text"):
                for line in element:
                    text = line.get_text()
                    match = geom_patt.search(text)
                    if not match and last_line and atoms:
                        name_match = geom_patt.search(name)
                        if name_match:
                            geom = Geometry(all_names[-1] + ".xyz")
                            geom.atoms.extend(atoms)
                        else:
                            geom = Geometry(atoms)
                            geom.name = name
                            geom.comment = name
                            if args.directory != "CURRENTDIR":
                                geom.name = os.path.join(
                                    args.directory, geom.name
                                )
                            orig_name = geom.name
                            i = 2
                            while geom.name in all_names:
                                geom.name = "{}_{:03d}".format(orig_name, i)
                                i += 1
                        if args.sort:
                            geom.refresh_connected()
                            geom.refresh_ranks()
                            geom.atoms = geom.reorder()[0]
                        geom.write()
                        all_names.append(geom.name)
                        atoms = []
                        name = None
                        # print()
                        # print(geom.name, len(geom.atoms))
                        # print(geom)
                    if match:
                        if not name:
                            name = last_line
                        element = match.group(1)
                        coords = float_patt.findall(match.group(2))
                        atoms.append(Atom(element, [float(c) for c in coords]))
                    last_line = text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="PDF file to scrape geometries from")
    parser.add_argument(
        "directory",
        nargs="?",
        default="CURRENTDIR",
        help="directory to save XYZ files in",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="perform canonical sorting before saving",
    )

    args = parser.parse_args()
    main(args)
