"""Holds constants"""
import os

HOME = os.environ['HOME'].rstrip('/') + '/'
AARONLIB = os.environ['AARONLIB'].rstrip('/') + '/'

QCHASM = os.path.dirname(os.path.abspath(__file__)).split('/')
QCHASM = '/'.join(QCHASM[:-1]) + '/'

CONNECTIVITY_THRESHOLD = 0.5

ELEMENTS = [
    'Bq', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
    'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
    'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
    'Rn', 'X'
]

TMETAL = {
    'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17,
    'Co': 1.16, 'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Y': 1.62, 'Zr': 1.45,
    'Nb': 1.34, 'Mo': 1.30, 'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.28,
    'Ag': 1.34, 'Cd': 1.48, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34,
    'W': 1.30, 'Re': 1.28, 'Os': 1.26, 'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34,
    'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47
}

RADII = {
    'H': 0.32, 'He': 0.93, 'Li': 1.23, 'Be': 0.90, 'B': 0.82, 'C': 0.77,
    'N': 0.75, 'O': 0.73, 'F': 0.72, 'Ne': 0.71, 'Na': 1.54, 'Mg': 1.36,
    'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.98,
    'K': 2.03, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18,
    'Mn': 1.17, 'Fe': 1.17, 'Co': 1.16, 'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25,
    'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.12,
    'Rb': 2.16, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.45, 'Nb': 1.34, 'Mo': 1.30,
    'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.28, 'Ag': 1.34, 'Cd': 1.48,
    'In': 1.44, 'Sn': 1.41, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
    'Cs': 2.35, 'Ba': 1.98, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34,
    'W': 1.30, 'Re': 1.28, 'Os': 1.26, 'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34,
    'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47, 'Bi': 1.46,
    'X': 0
}

CONNECTIVITY = {
    'H': 1, 'B': 4, 'C': 4, 'N': 4, 'O': 2, 'F': 1, 'Si': 6, 'Rh': 6, 'Fe': 6,
    'Ni': 6, 'Cu': 6, 'Ru': 6, 'Pd': 6, 'P': 4, 'S': 4, 'Cl': 1, 'I': 1,
    'Br': 1, 'X': 1000, 'Pt': 6,
    'Au': 6
}

ELECTRONEGATIVITY = {
    'H': 2.20, 'He': None, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16,
    'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.20, 'Tb': None, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None, 'Lu': 1.27, 'Hf': 1.3,
    'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0,
    'At': 2.2, 'Rn': None, 'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3,
    'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3,
    'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3, 'Md': 1.3,
    'No': 1.3
}

MASS = {
    'X': 0., 'H': 1.00782503207, 'He': 4.00260325415, 'Li': 7.016004548,
    'Be': 9.012182201, 'B': 11.009305406, 'C': 12.0, 'N': 14.00307400478,
    'O': 15.99491461956, 'F': 18.998403224, 'Ne': 19.99244017542,
    'Na': 22.98976928087, 'Mg': 23.985041699, 'Al': 26.981538627,
    'Si': 27.97692653246, 'P': 30.973761629, 'S': 31.972070999,
    'Cl': 34.968852682, 'Ar': 39.96238312251, 'K': 38.963706679,
    'Ca': 39.962590983, 'Sc': 44.955911909, 'Ti': 47.947946281,
    'V': 50.943959507, 'Cr': 51.940507472, 'Mn': 54.938045141,
    'Fe': 55.934937475, 'Co': 58.933195048, 'Ni': 57.935342907,
    'Cu': 62.929597474, 'Zn': 63.929142222, 'Ga': 68.925573587,
    'Ge': 73.921177767, 'As': 74.921596478, 'Se': 79.916521271,
    'Br': 78.918337087, 'Kr': 85.910610729, 'Rb': 84.911789737,
    'Sr': 87.905612124, 'Y': 88.905848295, 'Zr': 89.904704416,
    'Nb': 92.906378058, 'Mo': 97.905408169, 'Tc': 98.906254747,
    'Ru': 101.904349312, 'Rh': 102.905504292, 'Pd': 105.903485715,
    'Ag': 106.90509682, 'Cd': 113.90335854, 'In': 114.903878484,
    'Sn': 119.902194676, 'Sb': 120.903815686, 'Te': 129.906224399,
    'I': 126.904472681, 'Xe': 131.904153457, 'Cs': 132.905451932,
    'Ba': 137.905247237, 'La': 138.906353267, 'Lu': 174.940771819,
    'Hf': 179.946549953, 'Ta': 180.947995763, 'W': 183.950931188,
    'Re': 186.955753109, 'Os': 191.96148069, 'Ir': 192.96292643,
    'Pt': 194.964791134, 'Au': 196.966568662, 'Hg': 201.970643011,
    'Tl': 204.974427541, 'Pb': 207.976652071,
    'Bi': 208.980398734
}

EIJ = {
    'CC': 0.1500, 'CN': 0.1549, 'NC': 0.1549, 'CO': 0.1732, 'OC': 0.1732,
    'CP': 0.1732, 'PC': 0.1732, 'CS': 0.1732, 'SC': 0.1732, 'CH': 0.0548,
    'HC': 0.0548, 'CFe': 0.0387, 'FeC': 0.0387, 'CF': 0.1095, 'FC': 0.1095,
    'CCl': 0.2035, 'ClC': 0.2035, 'CBr': 0.2416, 'BrC': 0.2416, 'CI': 0.2877,
    'IC': 0.2877, 'CMg': 0.3623, 'MgC': 0.3623, 'CZn': 0.2872, 'ZnC': 0.2872,
    'CCa': 0.2872, 'CaC': 0.2872, 'NC': 0.1549, 'CN': 0.1549, 'NN': 0.1600,
    'NO': 0.1789, 'ON': 0.1789, 'NP': 0.1789, 'PN': 0.1789, 'NS': 0.1789,
    'SN': 0.1789, 'NH': 0.0566, 'HN': 0.0566, 'NFe': 0.0400, 'FeN': 0.0400,
    'NF': 0.1131, 'FN': 0.1131, 'NCl': 0.2101, 'ClN': 0.2101, 'NBr': 0.2495,
    'BrN': 0.2495, 'NI': 0.2972, 'IN': 0.2972, 'NMg': 0.3742, 'MgN': 0.3742,
    'NZn': 0.2966, 'ZnN': 0.2966, 'NCa': 0.2966, 'CaN': 0.2966, 'OC': 0.1732,
    'CO': 0.1732, 'ON': 0.1789, 'NO': 0.1789, 'OO': 0.2000, 'OP': 0.2000,
    'PO': 0.2000, 'OS': 0.2000, 'SO': 0.2000, 'OH': 0.0632, 'HO': 0.0632,
    'OFe': 0.0447, 'FeO': 0.0447, 'OF': 0.1265, 'FO': 0.1265, 'OCl': 0.2349,
    'ClO': 0.2349, 'OBr': 0.2789, 'BrO': 0.2789, 'OI': 0.3323, 'IO': 0.3323,
    'OMg': 0.4183, 'MgO': 0.4183, 'OZn': 0.3317, 'ZnO': 0.3317, 'OCa': 0.3317,
    'CaO': 0.3317, 'PC': 0.1732, 'CP': 0.1732, 'PN': 0.1789, 'NP': 0.1789,
    'PO': 0.2000, 'OP': 0.2000, 'PP': 0.2000, 'PS': 0.2000, 'SP': 0.2000,
    'PH': 0.0632, 'HP': 0.0632, 'PFe': 0.0447, 'FeP': 0.0447, 'PF': 0.1265,
    'FP': 0.1265, 'PCl': 0.2349, 'ClP': 0.2349, 'PBr': 0.2789, 'BrP': 0.2789,
    'PI': 0.3323, 'IP': 0.3323, 'PMg': 0.4183, 'MgP': 0.4183, 'PZn': 0.3317,
    'ZnP': 0.3317, 'PCa': 0.3317, 'CaP': 0.3317, 'SC': 0.1732, 'CS': 0.1732,
    'SN': 0.1789, 'NS': 0.1789, 'SO': 0.2000, 'OS': 0.2000, 'SP': 0.2000,
    'PS': 0.2000, 'SS': 0.2000, 'SH': 0.0632, 'HS': 0.0632, 'SFe': 0.0447,
    'FeS': 0.0447, 'SF': 0.1265, 'FS': 0.1265, 'SCl': 0.2349, 'ClS': 0.2349,
    'SBr': 0.2789, 'BrS': 0.2789, 'SI': 0.3323, 'IS': 0.3323, 'SMg': 0.4183,
    'MgS': 0.4183, 'SZn': 0.3317, 'ZnS': 0.3317, 'SCa': 0.3317, 'CaS': 0.3317,
    'HC': 0.0548, 'CH': 0.0548, 'HN': 0.0566, 'NH': 0.0566, 'HO': 0.0632,
    'OH': 0.0632, 'HP': 0.0632, 'PH': 0.0632, 'HS': 0.0632, 'SH': 0.0632,
    'HH': 0.0200, 'HFe': 0.0141, 'FeH': 0.0141, 'HF': 0.0400, 'FH': 0.0400,
    'HCl': 0.0743, 'ClH': 0.0743, 'HBr': 0.0882, 'BrH': 0.0882, 'HI': 0.1051,
    'IH': 0.1051, 'HMg': 0.1323, 'MgH': 0.1323, 'HZn': 0.1049, 'ZnH': 0.1049,
    'HCa': 0.1049, 'CaH': 0.1049, 'FeC': 0.0387, 'CFe': 0.0387, 'FeN': 0.0400,
    'NFe': 0.0400, 'FeO': 0.0447, 'OFe': 0.0447, 'FeP': 0.0447, 'PFe': 0.0447,
    'FeS': 0.0447, 'SFe': 0.0447, 'FeH': 0.0141, 'HFe': 0.0141, 'FeFe': 0.0100,
    'FeF': 0.0283, 'FFe': 0.0283, 'FeCl': 0.0525, 'ClFe': 0.0525,
    'FeBr': 0.0624, 'BrFe': 0.0624, 'FeI': 0.0743, 'IFe': 0.0743,
    'FeMg': 0.0935, 'MgFe': 0.0935, 'FeZn': 0.0742, 'ZnFe': 0.0742,
    'FeCa': 0.0742, 'CaFe': 0.0742, 'FC': 0.1095, 'CF': 0.1095, 'FN': 0.1131,
    'NF': 0.1131, 'FO': 0.1265, 'OF': 0.1265, 'FP': 0.1265, 'PF': 0.1265,
    'FS': 0.1265, 'SF': 0.1265, 'FH': 0.0400, 'HF': 0.0400, 'FFe': 0.0283,
    'FeF': 0.0283, 'FF': 0.0800, 'FCl': 0.1486, 'ClF': 0.1486, 'FBr': 0.1764,
    'BrF': 0.1764, 'FI': 0.2101, 'IF': 0.2101, 'FMg': 0.2646, 'MgF': 0.2646,
    'FZn': 0.2098, 'ZnF': 0.2098, 'FCa': 0.2098, 'CaF': 0.2098, 'ClC': 0.2035,
    'CCl': 0.2035, 'ClN': 0.2101, 'NCl': 0.2101, 'ClO': 0.2349, 'OCl': 0.2349,
    'ClP': 0.2349, 'PCl': 0.2349, 'ClS': 0.2349, 'SCl': 0.2349, 'ClH': 0.0743,
    'HCl': 0.0743, 'ClFe': 0.0525, 'FeCl': 0.0525, 'ClF': 0.1486,
    'FCl': 0.1486, 'ClCl': 0.2760, 'ClBr': 0.3277, 'BrCl': 0.3277,
    'ClI': 0.3903, 'ICl': 0.3903, 'ClMg': 0.4914, 'MgCl': 0.4914,
    'ClZn': 0.3896, 'ZnCl': 0.3896, 'ClCa': 0.3896, 'CaCl': 0.3896,
    'BrC': 0.2416, 'CBr': 0.2416, 'BrN': 0.2495, 'NBr': 0.2495, 'BrO': 0.2789,
    'OBr': 0.2789, 'BrP': 0.2789, 'PBr': 0.2789, 'BrS': 0.2789, 'SBr': 0.2789,
    'BrH': 0.0882, 'HBr': 0.0882, 'BrFe': 0.0624, 'FeBr': 0.0624,
    'BrF': 0.1764, 'FBr': 0.1764, 'BrCl': 0.3277, 'ClBr': 0.3277,
    'BrBr': 0.3890, 'BrI': 0.4634, 'IBr': 0.4634, 'BrMg': 0.5834,
    'MgBr': 0.5834, 'BrZn': 0.4625, 'ZnBr': 0.4625, 'BrCa': 0.4625,
    'CaBr': 0.4625, 'IC': 0.2877, 'CI': 0.2877, 'IN': 0.2972, 'NI': 0.2972,
    'IO': 0.3323, 'OI': 0.3323, 'IP': 0.3323, 'PI': 0.3323, 'IS': 0.3323,
    'SI': 0.3323, 'IH': 0.1051, 'HI': 0.1051, 'IFe': 0.0743, 'FeI': 0.0743,
    'IF': 0.2101, 'FI': 0.2101, 'ICl': 0.3903, 'ClI': 0.3903, 'IBr': 0.4634,
    'BrI': 0.4634, 'II': 0.5520, 'IMg': 0.6950, 'MgI': 0.6950, 'IZn': 0.5510,
    'ZnI': 0.5510, 'ICa': 0.5510, 'CaI': 0.5510, 'MgC': 0.3623, 'CMg': 0.3623,
    'MgN': 0.3742, 'NMg': 0.3742, 'MgO': 0.4183, 'OMg': 0.4183, 'MgP': 0.4183,
    'PMg': 0.4183, 'MgS': 0.4183, 'SMg': 0.4183, 'MgH': 0.1323, 'HMg': 0.1323,
    'MgFe': 0.0935, 'FeMg': 0.0935, 'MgF': 0.2646, 'FMg': 0.2646,
    'MgCl': 0.4914, 'ClMg': 0.4914, 'MgBr': 0.5834, 'BrMg': 0.5834,
    'MgI': 0.6950, 'IMg': 0.6950, 'MgMg': 0.8750, 'MgZn': 0.6937,
    'ZnMg': 0.6937, 'MgCa': 0.6937, 'CaMg': 0.6937, 'ZnC': 0.2872,
    'CZn': 0.2872, 'ZnN': 0.2966, 'NZn': 0.2966, 'ZnO': 0.3317, 'OZn': 0.3317,
    'ZnP': 0.3317, 'PZn': 0.3317, 'ZnS': 0.3317, 'SZn': 0.3317, 'ZnH': 0.1049,
    'HZn': 0.1049, 'ZnFe': 0.0742, 'FeZn': 0.0742, 'ZnF': 0.2098,
    'FZn': 0.2098, 'ZnCl': 0.3896, 'ClZn': 0.3896, 'ZnBr': 0.4625,
    'BrZn': 0.4625, 'ZnI': 0.5510, 'IZn': 0.5510, 'ZnMg': 0.6937,
    'MgZn': 0.6937, 'ZnZn': 0.5500, 'ZnCa': 0.5500, 'CaZn': 0.5500,
    'CaC': 0.2872, 'CCa': 0.2872, 'CaN': 0.2966, 'NCa': 0.2966, 'CaO': 0.3317,
    'OCa': 0.3317, 'CaP': 0.3317, 'PCa': 0.3317, 'CaS': 0.3317, 'SCa': 0.3317,
    'CaH': 0.1049, 'HCa': 0.1049, 'CaFe': 0.0742, 'FeCa': 0.0742,
    'CaF': 0.2098, 'FCa': 0.2098, 'CaCl': 0.3896, 'ClCa': 0.3896,
    'CaBr': 0.4625, 'BrCa': 0.4625, 'CaI': 0.5510, 'ICa': 0.5510,
    'CaMg': 0.6937, 'MgCa': 0.6937, 'CaZn': 0.5500, 'ZnCa': 0.5500,
    'CaCa': 0.5500, 'X': 0
}

RIJ = {
    'CC': 4.00, 'CN': 3.75, 'NC': 3.75, 'CO': 3.60, 'OC': 3.60, 'CP': 4.10,
    'PC': 4.10, 'CS': 4.00, 'SC': 4.00, 'CH': 3.00, 'HC': 3.00, 'CFe': 2.65,
    'FeC': 2.65, 'CF': 3.54, 'FC': 3.54, 'CCl': 4.04, 'ClC': 4.04, 'CBr': 4.17,
    'BrC': 4.17, 'CI': 4.36, 'IC': 4.36, 'CMg': 2.65, 'MgC': 2.65, 'CZn': 2.74,
    'ZnC': 2.74, 'CCa': 2.99, 'CaC': 2.99, 'NC': 3.75, 'CN': 3.75, 'NN': 3.50,
    'NO': 3.35, 'ON': 3.35, 'NP': 3.85, 'PN': 3.85, 'NS': 3.75, 'SN': 3.75,
    'NH': 2.75, 'HN': 2.75, 'NFe': 2.40, 'FeN': 2.40, 'NF': 3.29, 'FN': 3.29,
    'NCl': 3.79, 'ClN': 3.79, 'NBr': 3.92, 'BrN': 3.92, 'NI': 4.11, 'IN': 4.11,
    'NMg': 2.40, 'MgN': 2.40, 'NZn': 2.49, 'ZnN': 2.49, 'NCa': 2.74,
    'CaN': 2.74, 'OC': 3.60, 'CO': 3.60, 'ON': 3.35, 'NO': 3.35, 'OO': 3.20,
    'OP': 3.70, 'PO': 3.70, 'OS': 3.60, 'SO': 3.60, 'OH': 2.60, 'HO': 2.60,
    'OFe': 2.25, 'FeO': 2.25, 'OF': 3.15, 'FO': 3.15, 'OCl': 3.65, 'ClO': 3.65,
    'OBr': 3.77, 'BrO': 3.77, 'OI': 3.96, 'IO': 3.96, 'OMg': 2.25, 'MgO': 2.25,
    'OZn': 2.34, 'ZnO': 2.34, 'OCa': 2.59, 'CaO': 2.59, 'PC': 4.10, 'CP': 4.10,
    'PN': 3.85, 'NP': 3.85, 'PO': 3.70, 'OP': 3.70, 'PP': 4.20, 'PS': 4.10,
    'SP': 4.10, 'PH': 3.10, 'HP': 3.10, 'PFe': 2.75, 'FeP': 2.75, 'PF': 3.65,
    'FP': 3.65, 'PCl': 4.14, 'ClP': 4.14, 'PBr': 4.27, 'BrP': 4.27, 'PI': 4.46,
    'IP': 4.46, 'PMg': 2.75, 'MgP': 2.75, 'PZn': 2.84, 'ZnP': 2.84,
    'PCa': 3.09, 'CaP': 3.09, 'SC': 4.00, 'CS': 4.00, 'SN': 3.75, 'NS': 3.75,
    'SO': 3.60, 'OS': 3.60, 'SP': 4.10, 'PS': 4.10, 'SS': 4.00, 'SH': 3.00,
    'HS': 3.00, 'SFe': 2.65, 'FeS': 2.65, 'SF': 3.54, 'FS': 3.54, 'SCl': 4.04,
    'ClS': 4.04, 'SBr': 4.17, 'BrS': 4.17, 'SI': 4.36, 'IS': 4.36, 'SMg': 2.65,
    'MgS': 2.65, 'SZn': 2.74, 'ZnS': 2.74, 'SCa': 2.99, 'CaS': 2.99,
    'HC': 3.00, 'CH': 3.00, 'HN': 2.75, 'NH': 2.75, 'HO': 2.60, 'OH': 2.60,
    'HP': 3.10, 'PH': 3.10, 'HS': 3.00, 'SH': 3.00, 'HH': 2.00, 'HFe': 1.65,
    'FeH': 1.65, 'HF': 2.54, 'FH': 2.54, 'HCl': 3.04, 'ClH': 3.04, 'HBr': 3.17,
    'BrH': 3.17, 'HI': 3.36, 'IH': 3.36, 'HMg': 1.65, 'MgH': 1.65, 'HZn': 1.74,
    'ZnH': 1.74, 'HCa': 1.99, 'CaH': 1.99, 'FeC': 2.65, 'CFe': 2.65,
    'FeN': 2.40, 'NFe': 2.40, 'FeO': 2.25, 'OFe': 2.25, 'FeP': 2.75,
    'PFe': 2.75, 'FeS': 2.65, 'SFe': 2.65, 'FeH': 1.65, 'HFe': 1.65,
    'FeFe': 1.30, 'FeF': 2.19, 'FFe': 2.19, 'FeCl': 2.69, 'ClFe': 2.69,
    'FeBr': 2.81, 'BrFe': 2.81, 'FeI': 3.01, 'IFe': 3.01, 'FeMg': 1.30,
    'MgFe': 1.30, 'FeZn': 1.39, 'ZnFe': 1.39, 'FeCa': 1.64, 'CaFe': 1.64,
    'FC': 3.54, 'CF': 3.54, 'FN': 3.29, 'NF': 3.29, 'FO': 3.15, 'OF': 3.15,
    'FP': 3.65, 'PF': 3.65, 'FS': 3.54, 'SF': 3.54, 'FH': 2.54, 'HF': 2.54,
    'FFe': 2.19, 'FeF': 2.19, 'FF': 3.09, 'FCl': 3.59, 'ClF': 3.59,
    'FBr': 3.71, 'BrF': 3.71, 'FI': 3.90, 'IF': 3.90, 'FMg': 2.19, 'MgF': 2.19,
    'FZn': 2.29, 'ZnF': 2.29, 'FCa': 2.54, 'CaF': 2.54, 'ClC': 4.04,
    'CCl': 4.04, 'ClN': 3.79, 'NCl': 3.79, 'ClO': 3.65, 'OCl': 3.65,
    'ClP': 4.14, 'PCl': 4.14, 'ClS': 4.04, 'SCl': 4.04, 'ClH': 3.04,
    'HCl': 3.04, 'ClFe': 2.69, 'FeCl': 2.69, 'ClF': 3.59, 'FCl': 3.59,
    'ClCl': 4.09, 'ClBr': 4.21, 'BrCl': 4.21, 'ClI': 4.40, 'ICl': 4.40,
    'ClMg': 2.69, 'MgCl': 2.69, 'ClZn': 2.79, 'ZnCl': 2.79, 'ClCa': 3.04,
    'CaCl': 3.04, 'BrC': 4.17, 'CBr': 4.17, 'BrN': 3.92, 'NBr': 3.92,
    'BrO': 3.77, 'OBr': 3.77, 'BrP': 4.27, 'PBr': 4.27, 'BrS': 4.17,
    'SBr': 4.17, 'BrH': 3.17, 'HBr': 3.17, 'BrFe': 2.81, 'FeBr': 2.81,
    'BrF': 3.71, 'FBr': 3.71, 'BrCl': 4.21, 'ClBr': 4.21, 'BrBr': 4.33,
    'BrI': 4.53, 'IBr': 4.53, 'BrMg': 2.81, 'MgBr': 2.81, 'BrZn': 2.91,
    'ZnBr': 2.91, 'BrCa': 3.16, 'CaBr': 3.16, 'IC': 4.36, 'CI': 4.36,
    'IN': 4.11, 'NI': 4.11, 'IO': 3.96, 'OI': 3.96, 'IP': 4.46, 'PI': 4.46,
    'IS': 4.36, 'SI': 4.36, 'IH': 3.36, 'HI': 3.36, 'IFe': 3.01, 'FeI': 3.01,
    'IF': 3.90, 'FI': 3.90, 'ICl': 4.40, 'ClI': 4.40, 'IBr': 4.53, 'BrI': 4.53,
    'II': 4.72, 'IMg': 3.01, 'MgI': 3.01, 'IZn': 3.10, 'ZnI': 3.10,
    'ICa': 3.35, 'CaI': 3.35, 'MgC': 2.65, 'CMg': 2.65, 'MgN': 2.40,
    'NMg': 2.40, 'MgO': 2.25, 'OMg': 2.25, 'MgP': 2.75, 'PMg': 2.75,
    'MgS': 2.65, 'SMg': 2.65, 'MgH': 1.65, 'HMg': 1.65, 'MgFe': 1.30,
    'FeMg': 1.30, 'MgF': 2.19, 'FMg': 2.19, 'MgCl': 2.69, 'ClMg': 2.69,
    'MgBr': 2.81, 'BrMg': 2.81, 'MgI': 3.01, 'IMg': 3.01, 'MgMg': 1.30,
    'MgZn': 1.39, 'ZnMg': 1.39, 'MgCa': 1.64, 'CaMg': 1.64, 'ZnC': 2.74,
    'CZn': 2.74, 'ZnN': 2.49, 'NZn': 2.49, 'ZnO': 2.34, 'OZn': 2.34,
    'ZnP': 2.84, 'PZn': 2.84, 'ZnS': 2.74, 'SZn': 2.74, 'ZnH': 1.74,
    'HZn': 1.74, 'ZnFe': 1.39, 'FeZn': 1.39, 'ZnF': 2.29, 'FZn': 2.29,
    'ZnCl': 2.79, 'ClZn': 2.79, 'ZnBr': 2.91, 'BrZn': 2.91, 'ZnI': 3.10,
    'IZn': 3.10, 'ZnMg': 1.39, 'MgZn': 1.39, 'ZnZn': 1.48, 'ZnCa': 1.73,
    'CaZn': 1.73, 'CaC': 2.99, 'CCa': 2.99, 'CaN': 2.74, 'NCa': 2.74,
    'CaO': 2.59, 'OCa': 2.59, 'CaP': 3.09, 'PCa': 3.09, 'CaS': 2.99,
    'SCa': 2.99, 'CaH': 1.99, 'HCa': 1.99, 'CaFe': 1.64, 'FeCa': 1.64,
    'CaF': 2.54, 'FCa': 2.54, 'CaCl': 3.04, 'ClCa': 3.04, 'CaBr': 3.16,
    'BrCa': 3.16, 'CaI': 3.35, 'ICa': 3.35, 'CaMg': 1.64, 'MgCa': 1.64,
    'CaZn': 1.73, 'ZnCa': 1.73, 'CaCa': 1.98, 'X': 0
}


class PHYSICAL:
    # Physical constants
    BOLTZMANN = 0.001987204         # kcal/mol
    KB = 1.380662E-23               # J/K
    ROOM_TEMPERATURE = 298.15       # K
    PLANK = 6.62606957E-34
    SPEED_OF_LIGHT = 29979245800    # cm/s
    GAS_CONSTANT = 1.987204         # cal/mol
    STANDARD_PRESSURE = 101317      # Pa


class UNIT:
    # Unit conversion
    AMU_TO_KG = 1.66053886E-27
    HART_TO_KCAL = 627.5095
