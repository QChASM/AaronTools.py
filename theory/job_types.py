import os

from AaronTools.theory import ORCA_ROUTE, ORCA_BLOCKS, \
                              \
                              PSI4_JOB, PSI4_SETTINGS, PSI4_BEFORE_GEOM, PSI4_OPTKING, \
                              \
                              GAUSSIAN_ROUTE, GAUSSIAN_GEN_BASIS, GAUSSIAN_GEN_ECP, GAUSSIAN_CONSTRAINTS


class JobType:
    """parent class of all job types"""
    def __init__(self):
        pass

    def get_gaussian(self):
        """overwrite to return dict with GAUSSIAN_* keys"""
        pass
    
    def get_orca(self):
        """overwrite to return dict with ORCA_* keys"""
        pass

    def get_psi4(self):
        """overwrite to return dict with PSI4_* keys"""
        pass


class OptimizationJob(JobType):
    """optimization job"""
    def __init__(self, transition_state=False, constraints=None, geom=None):
        """use transition_state=True to do a TS optimization
        constraints - dict with 'atoms', 'bonds', 'angles' and 'torsions' as keys
                      constraints['atoms']: list(Atom) - atoms to constrain
                      constraints['bonds']: list(list(Atom, len=2)) - distances to constrain
                      constraints['angles']: list(list(Atom, len=3)) - 1-3 angles to constrain
                      constraints['torsions']: list(list(Atom, len=4)) - dihedral angles to constrain
        geom        - Geoemtry, required if using constraints"""
        super().__init__()

        self.ts = transition_state
        self.constraints = constraints
        self.structure = geom

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE, GAUSSIAN_CONSTRAINTS"""
        if self.ts:
            out = {GAUSSIAN_ROUTE:{'Opt':['ts']}}
        else:
            out = {GAUSSIAN_ROUTE:{'Opt':[]}}

        if self.constraints is not None and any(len(self.constraints[key]) > 0 for key in self.constraints.keys()):
            out[GAUSSIAN_ROUTE]['Opt'].append('ModRedundant')
            out[GAUSSIAN_CONSTRAINTS] = []

            if 'bonds' in self.constraints:
                for constraint in self.constraints['bonds']:
                    atom1, atom2 = constraint
                    ndx1 = self.structure.atoms.index(atom1) + 1
                    ndx2 = self.structure.atoms.index(atom2) + 1
                    out[GAUSSIAN_CONSTRAINTS].append("B %2i %2i F" % (ndx1, ndx2))

            if 'angles' in self.constraints:
                for constraint in self.constraints['angles']:
                    atom1, atom2, atom3 = constraint
                    ndx1 = self.structure.atoms.index(atom1) + 1
                    ndx2 = self.structure.atoms.index(atom2) + 1
                    ndx3 = self.structure.atoms.index(atom3) + 1
                    out[GAUSSIAN_CONSTRAINTS].append("A %2i %2i %2i F" % (ndx1, ndx2, ndx3))

            if 'torsions' in self.constraints:
                for constraint in self.constraints['torsions']:
                    atom1, atom2, atom3, atom4 = constraint
                    ndx1 = self.structure.atoms.index(atom1) + 1
                    ndx2 = self.structure.atoms.index(atom2) + 1
                    ndx3 = self.structure.atoms.index(atom3) + 1
                    ndx4 = self.structure.atoms.index(atom4) + 1
                    out[GAUSSIAN_CONSTRAINTS].append("D %2i %2i %2i %2i F" % (ndx1, ndx2, ndx3, ndx4))

        return out

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE, ORCA_BLOCKS"""
        if self.ts:
            out = {ORCA_ROUTE:['OptTS']}
        else:
            out = {ORCA_ROUTE:['Opt']}
        
        if self.constraints is not None and \
           any(len(self.constraints[key]) > 0 for key in self.constraints.keys()):
            
            out[ORCA_BLOCKS] = {'geom':[]}
            if 'atoms' in self.constraints:
                for constraint in self.constraints['atoms']:
                    atom1 = constraint
                    ndx1 = self.structure.atoms.index(atom1)
                    s = "    {C %2i C}" % (ndx1)
                    out[ORCA_BLOCKS]['geom'].append(s)

            if 'bonds' in self.constraints:
                for constraint in self.constraints['bonds']:
                    atom1, atom2 = constraint
                    ndx1 = self.structure.atoms.index(atom1)
                    ndx2 = self.structure.atoms.index(atom2)
                    s = "    {B %2i %2i C}" % (ndx1, ndx2)
                    out[ORCA_BLOCKS]['geom'].append(s)

            if 'angles' in self.constraints:
                for constraint in self.constraints['angles']:
                    atom1, atom2, atom3 = constraint
                    ndx1 = self.structure.atoms.index(atom1)
                    ndx2 = self.structure.atoms.index(atom2)
                    ndx3 = self.structure.atoms.index(atom3)
                    s = "    {A %2i %2i %2i C}" % (ndx1, ndx2, ndx3)
                    out[ORCA_BLOCKS]['geom'].append(s)
            
            if 'torsions' in self.constraints:
                for constraint in self.constraints['torsions']:
                    atom1, atom2, atom3, atom4 = constraint
                    ndx1 = self.structure.atoms.index(atom1)
                    ndx2 = self.structure.atoms.index(atom2)
                    ndx3 = self.structure.atoms.index(atom3)
                    ndx4 = self.structure.atoms.index(atom4)
                    s = "    {D %2i %2i %2i %2i C}" % (ndx1, ndx2, ndx3, ndx4)
                    out[ORCA_BLOCKS]['geom'].append(s)

            out[ORCA_BLOCKS]['geom'].append("end")
        
        return out

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB, PSI4_OPTKING, PSI4_BEFORE_GEOM"""
        if self.ts:
            out = {PSI4_JOB:{'optimize':[]}, PSI4_SETTINGS:{'opt_type':['ts']}}
        else:
            out =  {PSI4_JOB:{'optimize':[]}}

        #constraints
        if self.constraints is not None and any([len(self.constraints[key]) > 0 for key in self.constraints.keys()]):
            if 'atoms' in self.constraints:
                s = ""
                if len(self.constraints['atoms']) > 0 and self.structure is not None:
                    s += "freeze_list = \"\"\"\n"
                    for atom in self.constraints['atoms']:
                        s += "    %2i xyz\n" % (self.structure.atoms.index(atom) + 1)
    
                    s += "\"\"\"\n"
                    s += "    \n"

            out[PSI4_BEFORE_GEOM] = [s]

            out[PSI4_OPTKING] = {'frozen_cartesian': ['$freeze_list']}

            if 'bonds' in self.constraints:
                if len(self.constraints['bonds']) > 0 and self.structure is not None:
                    s = "(\"\n"
                    for bond in self.constraints['bonds']:
                        atom1, atom2 = bond
                        s += "        %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, \
                                                    self.structure.atoms.index(atom2) + 1)

                    s += "    \")\n"

                    out[PSI4_OPTKING]['frozen_distance'] = [s]

            if 'angles' in self.constraints:
                if len(self.constraints['angles']) > 0 and self.structure is not None:
                    s = "(\"\n"
                    for angle in self.constraints['angles']:
                        atom1, atom2, atom3 = angle
                        s += "        %2i %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, \
                                                        self.structure.atoms.index(atom2) + 1, \
                                                        self.structure.atoms.index(atom3) + 1)

                    s += "    \")\n"
                    
                    out[PSI4_OPTKING]['frozen_bend'] = [s]

            if 'torsions' in self.constraints:
                if len(self.constraints['torsions']) > 0 and self.structure is not None:
                    s += '(\"\n'
                    for torsion in self.constraints['torsions']:
                        atom1, atom2, atom3, atom4 = torsion
                        s += "        %2i %2i %2i %2i\n" % (self.structure.atoms.index(atom1) + 1, \
                                                            self.structure.atoms.index(atom2) + 1, \
                                                            self.structure.atoms.index(atom3) + 1, \
                                                            self.structure.atoms.index(atom4) + 1)

                    s += "    \")\n"
                    
                    out[PSI4_OPTKING]['frozen_dihedral'] = [s]
        
        return out


class FrequencyJob(JobType):
    """frequnecy job"""
    def __init__(self, temperature=298.15):
        """temperature in K for thermochem info that gets printed in output file"""
        super().__init__()
        self.temperature = temperature

    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        return {GAUSSIAN_ROUTE:{'Freq':['temperature=%.2f' % self.temperature]}}

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE:['Freq'], ORCA_BLOCKS:{'freq':['Temp    %.2f' % self.temperature]}}

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        return {PSI4_JOB:{'frequencies':[]}, PSI4_SETTINGS:{'T': ["%.2f" % self.temperature]}}


class SinglePointJob(JobType):
    """single point energy"""
    def get_gaussian(self):
        """returns a dict with keys: GAUSSIAN_ROUTE"""
        return {GAUSSIAN_ROUTE:{'SP':[]}}

    def get_orca(self):
        """returns a dict with keys: ORCA_ROUTE"""
        return {ORCA_ROUTE:['SP']}

    def get_psi4(self):
        """returns a dict with keys: PSI4_JOB"""
        return {PSI4_JOB:{'energy':[]}}
