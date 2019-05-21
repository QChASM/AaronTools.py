#!/usr/bin/env python3
import re
import time
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup as bs


parent_fname = "bond_length.html"
domain = "https://cccbdb.nist.gov/"

class ScrapeBondLength:
    def __init__(self, parent_fname=parent_fname, overwrite=False):
        """
        grabs url entries from a master table
        """
        self.entries = {}
        self.table = []
        # parse html document
        with open(parent_fname) as f:
            soup = bs(f, features="html5lib")

        # get body
        body = soup.body

        # get child urls
        for item in body.find_all('a'):
            href = item['href']
            m = re.match('expbondlengths.*descript=(.*)(&all.*)?', href)
            if m is not None:
                name = m.group(1)
                self.entries[name] = domain + href
                self.download(name, overwrite)
        return

    def download(self, name, overwrite=False):
        href = self.entries[name]
        datafile = Path('html/' + name + '.html')
        if not overwrite and datafile.is_file():
            return
        try:
            with urllib.request.urlopen(href) as data:
                print("downloading", name)
                with open('html/' + name + '.html', 'w') as f:
                    f.write(data.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            print("while trying to download", name, "the following error occured:")
            print("  ", e, "sleeping for 2 seconds")
            time.sleep(2)
            self.download(name, overwrite)
        return

    def parse_entry(self, name):
        """grabs bond length from entry"""
        print("getting data for", name)
        # get table
        with open("html/" + name + ".html") as f:
            soup = bs(f, features="html5lib")
        body = soup.body

        # get data
        data = None
        for table in body.find_all('tbody'):
            # when only one entry
            for cell in table.find_all('td'):
                if cell.string == name:
                    data = table
                    break
            if data is not None:
                self.table += self.get_table(name, data)
                return
            # when multiple sub entries
            for cell in table.find_all('td'):
                for item in cell.find_all('a'):
                    href = item['href']
                    m = re.match('expbondlengths.*descript=(.*)&all=1', href)
                    if m is not None:
                        name = m.group(1)
                        self.entries[name] = domain + href
                        self.download(name, overwrite=True)
                        self.parse_entry(name)

    def get_table(self, name, data):
        table = []
        header = []
        # parse table
        for row in data.find_all('tr'):
            table_row = []
            for head in row.find_all('th'):
                header += [head.string]
            for cell in row.find_all('td'):
                table_row += [cell.string]
            if len(table_row) > 0 and re.match('r[A-Z][a-z]?[=#:.x]?[A-Z][a-z]?', table_row[0]):
                table += [table_row]
        return table


if __name__ == "__main__":
    import json

    scraper = ScrapeBondLength()
    entries = scraper.entries.keys()
    entries = list(entries)
    for e in entries:
        scraper.parse_entry(e)

    with open('bondlengths.json', 'w') as f:
        json.dump(scraper.table, f)
