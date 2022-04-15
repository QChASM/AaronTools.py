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
    n_geoms = 0
    for i, page in enumerate(extract_pages(args.infile)):
        print("parsing page {: 4d} please wait...".format(i + 1), end="\r")
        for element in page:
            last_line = None
            if hasattr(element, "get_text"):
                for line in element:
                    text = line.get_text()
                    match = geom_patt.search(text)
                    if not match and last_line and atoms:
                        print("last line", last_line)
                        name_match = geom_patt.search(name)
                        if name_match:
                            n_geoms += 1
                            if args.index:
                                geom=Geometry(atoms, name="%04i" % n_geoms)
                            else:
                                geom=Geometry(atoms, name=all_names[-1])
                        else:
                            n_geoms += 1
                            geom = Geometry(atoms)
                            if args.index:
                                geom.name = "%04i" % n_geoms
                            else:
                                geom.name = name
                            geom.comment = name
                            orig_name = geom.name
                            i = 2
                            while geom.name in all_names:
                                geom.name = "{}_{:03d}".format(orig_name, i)
                                i += 1
                        if args.sort:
                            geom.refresh_connected()
                            geom.refresh_ranks()
                            geom.atoms = geom.reorder()[0]
                        if args.directory is not None:
                            geom.name = os.path.join(
                                args.directory, geom.name
                            )
                        all_names.append(geom.name)
                        geom.write()
                        atoms = []
                        name = None
                        print()
                        print(geom.name, len(geom.atoms))
                        print(geom.write(outfile=False))
                    if match:
                        if not name:
                            name = last_line
                        element = match.group(1)
                        coords = float_patt.findall(match.group(2))
                        atoms.append(Atom(element, [float(c) for c in coords]))
                    last_line = text.strip()
    
    print("parsed %i geometries" % n_geoms)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="PDF file to scrape geometries from")
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="directory to save XYZ files in",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="perform canonical sorting before saving",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="use the index of the geometry as the name",
    )

    args = parser.parse_args()
    main(args)
