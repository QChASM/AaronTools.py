import os.path

from AaronTools.geometry import Geometry
from AaronTools.finders import *
from cProfile import Profile

profile = Profile()

geoms = [os.path.join("test_files", fname) for fname in [
    "benzene.xyz",
    "6a2e5am1hex.xyz",
    "bpy.xyz",
    "dppe.xyz",
    "pentane.xyz",
    "pyridine.xyz",
    "benzene_1-NO2_4-Cl.xyz",
    "benzene_1-OH_4-Cl.xyz",
    "benzene_1-Ph_4-Cl.xyz",
    "benzene_dimer.xyz",
    "chiral_centers_1.xyz",
    "chiral_centers_2.xyz",
    "chiral_centers_3.xyz",
    "chiral_centers_4.xyz",
    "chiral_ring.xyz",
    "chiral_ring_mirror.xyz",
    "test_rmsd_sort1.xyz",
    "test_rmsd_sort2.xyz",
]]


tot_found = 0

for geom_name in geoms:
    print(geom_name)
    geom = Geometry(geom_name)
    profile.enable()
    try:
        res = geom.find(OfType("Ca"))
        tot_found += len(res)
        for atom in res:
            print(atom)
    except LookupError:
        pass
    profile.disable()

profile.print_stats()


print("found %i atoms" % tot_found)