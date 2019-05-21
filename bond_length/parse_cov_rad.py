#!/usr/bin/env python


class DataParser:
    """
    For parsing data files, such as *.csv files.

    CLASS ATTRIBUTES
    :sep: field separator for rows (default: ",")
    :str_delim: string delimiter (default: "'")
    :header: column labels
    :rows: data rows, as list(dict(header_element: value))
    :cols: data columns, keyed by index of corresponding header element
    """
    sep = ","
    str_delim = "'"
    header = []
    rows = []
    cols = {}

    @classmethod
    def parse_file(cls, f):
        def parse_lines(f):
            for line in f:
                line = line.strip()
                if line == '/*':
                    while line != '*/':
                        line = f.readline().strip()
                    continue
                if line == '':
                    continue

                if not cls.header:
                    cls.save_header(line)
                else:
                    cls(line)

        if isinstance(f, str):
            with open(f) as fobj:
                parse_lines(fobj)
        else:
            parse_lines(f)

    @classmethod
    def save_header(cls, header):
        """Splits header row and saves it"""
        cls.header = [h.strip() for h in header.split(cls.sep)]
        cls.header = [h.strip(cls.str_delim) for h in cls.header]

    @classmethod
    def config(cls, **kwargs):
        """
        Used to change seperator and string delimiter

        :**kwargs: acceptable keys are 'sep' and 'str_delim'
        """
        if 'sep' in kwargs:
            cls.sep = kwargs['sep']
        if 'str_delim' in kwargs:
            cls.str_delim = kwargs['str_delim']

    def __init__(self, row):
        """

        :row: a line from the data file
        """
        self.row = []
        row = [r.strip() for r in row.split(DataParser.sep)]
        row = [r.strip(DataParser.str_delim) for r in row]
        for i, r in enumerate(row):
            try:
                if float(r) == int(r):
                    r = int(r)
                else:
                    r = float(r)
            except ValueError:
                pass
            self.row += [r]
            if i in DataParser.cols:
                DataParser.cols[i] += [r]
            else:
                DataParser.cols[i] = [r]
        DataParser.rows += [self.row]


def make_json(data):
    """
    Construct bond_order.json file
    """
    def make_orders(b1, b2):
        rv = {}
        for i in range(4):
            try:
                bond_length = b1[i] + b2[i]
            except (IndexError, TypeError):
                continue
            if not isinstance(bond_length, (int, float)):
                continue
            # save and convert to angstroms
            rv[i] = bond_length / 100

        # two columns of single bond lengths
        if 0 in rv and 1 in rv:
            rv[1] = (rv[0] + rv[1])/2
            del rv[0]
        elif 0 in rv:
            rv[1] = rv[0]
            del rv[0]

        # make aromatic bond lengths
        if 1 in rv and 2 in rv:
            rv[1.5] = (rv[1] + rv[2]) / 2

        return rv

    rv = []
    for i, r1 in enumerate(data.rows[:-1]):
        e1 = r1[1]
        b1 = r1[2:]
        for j, r2 in enumerate(data.rows[i+1:]):
            e2 = r2[1]
            b2 = r2[2:]

            bond_data = {}
            bond_data['atoms'] = (e1, e2)
            for order, val in make_orders(b1, b2).items():
                bond_data[order] = val

            rv += [bond_data]

    return rv


if __name__ == '__main__':
    DataParser.parse_file('covalent_radii.txt')
    data = DataParser
    data.header = ['atom_no', 'element', '1', '1', '2', '3']

    data = make_json(data)
    with open('calculated_bond_lengths.json', 'w') as f:
        for d in data:
            f.write(str(d) + '\n')
