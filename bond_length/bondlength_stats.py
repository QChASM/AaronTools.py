#!/usr/bin/env python3

import re
import json
import statistics as stat


ORDERKEY = {None: 1, '=': 2, '#':3, ':':1.5, '.':0.5, 'x':0}
# 1.5 -> aromatic, 0.5 -> hydrogen bond

class BondOrder:
    bond_data = []

    def __init__(self, name="", data=None):
        self.name = name
        self.data = data
        self.atoms = set([])
        self.order = None
        self.mean = None
        self.stdev = None
        self.min = None
        self.max = None
        self.N = None

        if name != "":
            self.atoms = set(re.findall("[A-Z][a-z]?", name))
            self.order = re.search("[A-Z][a-z]?(.)?[A-Z][a-z]?", name)
            self.order = ORDERKEY[self.order.group(1)]

        if data is not None:
            self.data = [float(i) for i in data]
            self.calc()

        BondOrder.bond_data += [self]

    def calc(self, data=None):
        """calculates statistics for a bond type"""
        if data is None:
            data = self.data

        self.N = len(data)
        if self.N == 1:
            self.mean = data[0]
        else:
            self.mean = stat.mean(data)
            self.stdev = stat.stdev(data)
            self.min = min(data)
            self.max = max(data)

    @classmethod
    def dump_json(self, fname):
        tmp = []
        for r in BondOrder.bond_data:
            t = {}
            for key in r.__dict__:
                if r.__dict__[key] is None:
                    continue
                if key in ["order", "mean", "stdev", "min", "max", "N"]:
                    t[key] = r.__dict__[key]
                if key in ["atoms"]:
                    t[key] = list(r.__dict__[key])
            tmp += [t]
        with open(fname, 'w') as f:
            json.dump(tmp, f)


if __name__ == "__main__":
    with open("bondlengths.json") as f:
        table = json.load(f)

    types = {}
    for row in table:
        name = row[0]
        value = row[3]
        try:
            types[name] += [value]
        except KeyError:
            types[name] = [value]

    for name in types:
        BondOrder(name, types[name])

    BondOrder.dump_json("bond_data.json")
