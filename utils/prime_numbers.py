#!/usr/bin/env python3

"""Generates prime numbers"""
import sys
from math import sqrt
from os import makedirs, path


class Primes:
    primes = [2, 3]
    clean = False
    cache = path.dirname(__file__) + "/cache/primes.dat"

    def __init__(self, clean=False, cache=None):
        Primes.clean = clean
        if cache is not None:
            Primes.cache = cache
        if Primes.clean or not path.exists(Primes.cache):
            dir, cache_file = path.split(Primes.cache)
            if not path.exists(dir):
                makedirs(dir)

            with open(Primes.cache, "w") as f:
                f.writelines([str(i) + "\n" for i in Primes.primes])
            f.close()

    @classmethod
    def next_prime(cls):
        # first return the ones we already found
        with open(Primes.cache) as f:
            for line in f:
                p = int(line.strip())
                cls.primes += [p]
                yield p
        f.close()

        # then start generating new ones
        q = cls.primes[-1] + 2
        while True:
            for p in cls.primes:
                if p > sqrt(q):
                    if q == cls.primes[-1]:
                        q += 2
                        break
                    cls.primes += [q]
                    cls.store_prime(q)
                    yield q
                if q % p == 0:
                    q += 2
                    break
            else:
                cls.primes += [q]
                cls.store_prime(q)
                yield q

    @classmethod
    def store_prime(cls, p):
        with open(cls.cache, "a") as f:
            if hasattr(p, "__iter__"):
                f.writelines([str(i) + "\n" for i in p])
            else:
                f.write(str(p) + "\n")
        f.close()

    @classmethod
    def list(cls, n):
        rv = []
        for i, p in enumerate(cls.next_prime()):
            if i >= n:
                break
            rv += [p]
        return rv


if __name__ == "__main__":
    Primes(clean=True)
    print(Primes.list(int(sys.argv[1])))
