#!/usr/bin/env python3

"""Generates prime numbers"""
import sys
from math import sqrt
from os import makedirs, path, access, W_OK

from AaronTools.const import AARONLIB


class Primes:
    """
    find and cache prime numbers
    """
    primes = [2, 3]
    clean = False
    cache = path.join(AARONLIB, "cache", "primes.dat")

    def __init__(self, clean=False, cache=None):
        Primes.clean = clean
        if cache is not None:
            Primes.cache = cache
        if Primes.clean or (not path.exists(Primes.cache) and access(Primes.cache, W_OK)):
            prime_dir, _ = path.split(Primes.cache)
            if not path.exists(prime_dir):
                makedirs(prime_dir)

            with open(Primes.cache, "w") as f:
                f.writelines([str(i) + "\n" for i in Primes.primes])
            f.close()

    @classmethod
    def next_prime(cls):
        """determine the next prime number"""
        # first return the ones we already found
        if path.exists(Primes.cache):
            with open(Primes.cache) as f:
                for line in f:
                    prime = int(line.strip())
                    cls.primes += [prime]
                    yield prime
            f.close()

        # then start generating new ones
        test_prime = cls.primes[-1] + 2
        while True:
            max_check = sqrt(test_prime)
            for prime in cls.primes[1:]:
                if test_prime % prime == 0:
                    test_prime += 2
                    break
                if prime > max_check:
                    cls.primes += [test_prime]
                    cls.store_prime(test_prime)
                    yield test_prime
                    test_prime += 2
                    break
            else:
                cls.primes += [test_prime]
                cls.store_prime(test_prime)
                yield test_prime
                test_prime += 2

    @classmethod
    def store_prime(cls, prime):
        """add the prime number to the cache"""
        if not path.exists(path.dirname(cls.cache)):
            return
        with open(cls.cache, "a") as f:
            if hasattr(prime, "__iter__"):
                f.writelines([str(i) + "\n" for i in prime])
            else:
                f.write(str(prime) + "\n")
        f.close()

    @classmethod
    def list(cls, n):
        """list the first n prime numbers"""
        rv = [prime for i, prime in enumerate(cls.primes) if i < n]
        if len(cls.primes) < n:
            if path.exists(cls.cache):
                cur_primes = len(cls.primes)
                k = 0
                with open(cls.cache) as f:
                    for line in f:
                        prime = int(line)
                        k += 1
                        if k + 2 > cur_primes - 2:
                            cls.primes += [prime]
                            rv += [prime]
                            if len(cls.primes) >= n:
                                break
            
            new_primes = []
        
            test_prime = cls.primes[-1] + 2
            while len(cls.primes) < n:
                # this is similar to next_prime, though it doesn't
                # cache each new prime number as it finds them
                # it saves them as a group
                # writing to a file is slow as heck
                max_check = sqrt(test_prime)
                for prime in cls.primes[1:]:
                    if test_prime % prime == 0:
                        test_prime += 2
                        break
                    if prime > max_check:
                        cls.primes += [test_prime]
                        new_primes += [test_prime]
                        test_prime += 2
                        break
                else:
                    cls.primes += [test_prime]
                    new_primes += [test_prime]
                    test_prime += 2

            rv += new_primes
            # print(new_primes)
            cls.store_prime(new_primes)
            
        return rv

    @classmethod
    def primes_below(cls, le):
        """list the primes that are less than or equal to le"""
        rv = [p for p in cls.primes if p <= le]
        if cls.primes[-1] <= le:
            for i, prime in enumerate(cls.next_prime()):
                if prime >= le:
                    break
                rv += [prime]
        return rv

if __name__ == "__main__":
    Primes(clean=True)
    print(Primes.list(int(sys.argv[1])))
