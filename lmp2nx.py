######################################################
# lmp2nx.py
#
# August 2014
# M Khawaja
#
# - Reads in LAMMPS data file and converts it into
#     network of atoms and connections between them
######################################################

import networkx as nx
import numpy as np
import time
import sys
import os
sys.path.append('/work/mmk07/install/pizza/src/')
sys.path.append('/home/mmk07/installs/pizza/src/')
from data import *


def lmp2nx(filename, bond_coefficients=True):
    """ Read LAMMPS data file

        - Takes LAMMPS input file and converts it into a network
            of nodes of atoms and edges between them (bonds)
        - Assigns bond types to each edge
        - Reads coefficients for all interactions

            Input:
                - filename.
                    LAMMPS data file
                - get_angles = False.
                    Read bonded information on angles
                - get_dihedrals = False.
                    Read bonded information on dihedrals

            Output:
                - pizza.py data object
                - network object containing atom information
                - dictionary of interaction coefficients lists
                     ('bond', 'angle', 'dihedral')
                - dictionary of dictionaries of bond types corresponding
                to combinations of atom types

        Modified: 5/8/14 (MK)

    """

    # Read file into pizza.py data object
    try:
        d = data(filename)
        masses = d.get("Masses")
    except:
        # pizza.py:data does not like '#'...
        cmd="cut -d'#' -f1 %s >> temp-lmp2nx.txt"%filename
        os.system(cmd)
        d = data('temp-lmp2nx.txt')
        masses = d.get("Masses")
        os.system('rm -f temp-lmp2nx.txt')

    # Read atom information: Masses, Pairs, Atoms (compulsory)
    pair_coeffs = d.get("Pair Coeffs")
    atoms = d.get("Atoms")

    # Assign atom information
    g = nx.Graph()
    for atom in atoms:
        num = int(atom[0])
        g.add_node(num)
        g_i = g.node[num]
        g_i['mol']          =    int(atom[1])
        g_i['atom_type']    =    int(atom[2])
        g_i['charge']       =    float(atom[3])
        g_i['position']     =    np.array(atom[4:7], dtype=float)
        if len(atom[7:10]) == 3:
            g_i['image']        =    np.array(atom[7:10], dtype=int)
        else:
            g_i['image']        =    np.array([0,0,0], dtype=int)
        g_i['eps']          =     float(pair_coeffs[g_i['atom_type']-1][1])
        g_i['sigma']        =     float(pair_coeffs[g_i['atom_type']-1][2])
        g_i['mass']         =     float(masses[g_i['atom_type']-1][1])

    # Read and assign unit cell
    ux = d.headers['xlo xhi']
    uy = d.headers['ylo yhi']
    uz = d.headers['zlo zhi']
    g.graph['unit_cell'] = [ux, uy, uz]

    # Read and assign bonds (edges)
    bonds = d.get("Bonds")
    for bond in bonds:
        btype = int(bond[1])
        i = int(bond[2])
        j = int(bond[3])
        g.add_edge(i, j, bond_type=btype)

    if bond_coefficients == True:
        lammps_coefficients={}
        lammps_coefficients['bond']     = d.get("Bond Coeffs")
        lammps_coefficients['angle']    = d.get("Angle Coeffs")
        lammps_coefficients['dihedral'] = d.get("Dihedral Coeffs")

        connections={}
        connections['bond']             = bonds
        connections['angle']            = d.get("Angles")
        connections['dihedral']         = d.get("Dihedrals")

        type_key = _get_coefficients_as_types(g, lammps_coefficients, connections)

        return d, g, lammps_coefficients, type_key
    else:
        return d, g



def compute_bonded_information(g, type_key):
    """ Computed bonded information

        - Takes networkx graph of atoms (nodes) and bonds (edges)
            and computes lists of the bonds, angles and dihedrals

            Input:
                - networkx graph generated from lmp2nx()
                - dictionary of which atoms correspond to which bonded
                    interaction types

            Output:
                - dictionary of lists of bonds, angles and dihedrals
                    in the LAMMPS format (e.g. bond number, bond_type,
                    atom_number_i, atom_number_j)
                    - undefined types listed as type=0
                - if missing angles found: list of missing angles
                - if missing dihedrals found: list of missing dihedrals

        Modified: 5/8/14 (MK)
    """
    # Determine all paths of length <=3 from every atom
    all_paths = nx.all_pairs_shortest_path(g, cutoff=3)
    connections={}
    res=[connections]

    # Calculate bonds
    bonds = g.edges(data=True)
    # Convert into LAMMPS format (bond number, type, atom_i, atom_j)
    connections['bond'] = [[i+1, bonds[i][2]['bond_type'], bonds[i][0], bonds[i][1]]\
                                     for i in range(len(bonds))]

    # Calculate angles:
    #     - list of angles by atom numbers in the form [i, j, k]
    angles=[]
    for firstAtom, paths in all_paths.items():
        for lastAtom, path in paths.items():
            if len(path) == 3:
                angles.append(path)

    angles = _removeDuplicatesArray(angles)
    # Convert into LAMMPS format
    cons = []
    nmissing_angles = []
    for i in range(len(angles)):
        n=i+1
        atom_types=[g.node[a]['atom_type'] for a in angles[i]]
        # Look up angle type given atom types
        key='-'.join(str(at) for at in atom_types)
        try:
            btype=type_key['angle'][key]
        except KeyError:
            # If coefficients not defined, set type to 0
            btype=0
            nmissing_angles.append(key)
        cons.append([n, btype]+angles[i])
    connections['angle'] = cons
    if len(nmissing_angles)>0:
        nmissing_angles = _removeDuplicates(nmissing_angles)
        res.append(nmissing_angles)
        print str(len(nmissing_angles)) +' angle types are missing coefficients'
        print nmissing_angles

    # Calculate dihedrals:
    #     - list of dihedrals by atom numbers in the form [i, j, k, l]
    dihedrals =[]
    i=1
    for firstAtom, paths in all_paths.items():
        for lastAtom, path in paths.items():
            i+=1
            if len(path) == 4:
                dihedrals.append(path)
    dihedrals = _removeDuplicatesArray(dihedrals)
    # Convert into LAMMPS format
    cons = []
    nmissing_dihedrals = []
    for i in range(len(dihedrals)):
        n=i+1
        atom_types=[g.node[a]['atom_type'] for a in dihedrals[i]]
        # Look up angle type given atom types
        key='-'.join(str(at) for at in atom_types)
        try:
            btype=type_key['dihedral'][key]
        except KeyError:
            # If coefficients not defined, set type to 0
            btype=0
            nmissing_dihedrals.append(key)
        cons.append([n, btype]+dihedrals[i])
    if len(nmissing_dihedrals)>0:
        nmissing_dihedrals = _removeDuplicates(nmissing_dihedrals)
        res.append(nmissing_dihedrals)
        print str(len(nmissing_dihedrals)) +' dihedral types are missing coefficients'
    connections['dihedral'] = cons

    return res




def write_lammps_output(g, lammps_coefficients, connections, newfile='new.txt'):
    """ Writes new lammps file based on specified atoms, coefficients and connections
            Input:
                - graph generated by lmp2nx()
                - coefficients in LAMMPS format (d.get("Bond Coeff") etc.)
                - connections in LAMMPS format ([bond number, type, atom_i/j/k/l] etc.)
                - new output file (default: new.txt)
            Output:
                - writes new output file
                - returns new file as LAMMPS data object
    """

    # New data object
    d = data()
    d.title = "Generated with lmp2nx on %s. "%(time.strftime("%H:%M:%S %d/%m/%Y"))
    try:
        d.title += g.graph['comment'] + '\n'
    except:
        d.title += '\n'

    # Ignore any missing bonds/angles/dihedrals
    connections = connections[0]

    # Populate pairs(atom_type), masses(atom_type) dictionaries
    pairs={}
    masses={}
    for n, atom in g.nodes_iter(data=True):
        at = int(atom['atom_type'])
        pairs[at] = [atom['eps'], atom['sigma']]
        masses[at] = atom['mass']

    # Set headers
    d.headers['atoms']                =     g.number_of_nodes()
    d.headers['atom types']            =     max(masses.keys())
    d.headers['angle types']         =    len(lammps_coefficients['angle'])
    d.headers['bond types']         =    len(lammps_coefficients['bond'])
    d.headers['dihedral types']     =    len(lammps_coefficients['dihedral'])
    d.headers['angles']                =     len(connections['angle'])
    d.headers['bonds']                =     len(connections['bond'])
    d.headers['dihedrals']            =     len(connections['dihedral'])
    d.headers['xlo xhi']            =     g.graph['unit_cell'][0]
    d.headers['ylo yhi']            =     g.graph['unit_cell'][1]
    d.headers['zlo zhi']            =     g.graph['unit_cell'][2]

    # Make sure all atom types missing masses are also missing parameters i.e.
    #   the atom type doesn't exist in the system.
    check_m = []
    check_p = []

    # Convert everything into lines format
    # Atoms
    atom_ls=[]
    for n, atom in g.nodes_iter(data=True):
        l = "%-10d %8d %8d %10.6f %15.9f %15.9f %15.9f %8d %8d %8d\n"
        prms = (n, atom['mol'], atom['atom_type'], atom['charge'], atom['position'][0],
                atom['position'][1], atom['position'][2], atom['image'][0], atom['image'][1],
                atom['image'][2])
        atom_ls.append(l%prms)

    # Masses
    mass_ls=list(np.zeros(max(masses.keys())))
    mkeys = masses.keys()
    for m in range(1, 1+len(mass_ls)):
      if m in mkeys:
        l = str(m).ljust(10)+str(masses[m])+'\n'
      else:
        l = str(m).ljust(10)+'0.001\n'
        check_m.append(m)
        print 'Mass for key %s not found: setting to 0.001' %m
        #raise StandardError('ERROR: mass for key %s not found' % m)
      mass_ls[m-1] = l

    # Pair coeffs
    pair_ls=list(np.zeros(max(masses.keys())))
    pkeys = pairs.keys()
    pkeys.sort()
    for p in range(1, 1+len(pair_ls)):
      if p in pkeys:
        l = "%-10d %15.9f %15.9f\n" % (p, pairs[p][0], pairs[p][1])
      else:
        l = str(p).ljust(10)+'0.001\t1.0\n'
        check_p.append(p)
        print 'Pair coeffs for key %s not found: setting to 0.001, 1.0' % p
      pair_ls[p-1] = l

    assert (np.sort(np.unique(check_m)) == np.sort(np.unique(check_p))).all()

    # Bonds/angles/dihedrals coeffs
    coeff_ls={}
    for bonded, values in lammps_coefficients.iteritems():
        coeff_ls[bonded] = []
        for v in values:
            l = "%-10d" % v[0]
            for coeff in v[1:]:
                l += "%15.9f " % coeff
            l += '\n'
            coeff_ls[bonded].append(l)

    # Bonds/angles/dihedrals connections
    conn_ls={}
    for bonded, values in connections.iteritems():
        conn_ls[bonded] = []
        for v in values:
            l=''
            for el in v:
                l += '%10d' % el
            l+='\n'
            conn_ls[bonded].append(l)


    # Set sections
    d.sections['Masses']            =    mass_ls
    d.sections['Pair Coeffs']         =    pair_ls
    d.sections['Bond Coeffs']         =    coeff_ls['bond']
    d.sections['Angle Coeffs']         =    coeff_ls['angle']
    d.sections['Dihedral Coeffs']     =    coeff_ls['dihedral']
    d.sections['Atoms']                 =    atom_ls
    d.sections['Bonds']                =     conn_ls['bond']
    d.sections['Angles']            =     conn_ls['angle']
    d.sections['Dihedrals']            =     conn_ls['dihedral']

    d.write(newfile)

    return d




def _get_coefficients_as_types(g, lammps_coefficients, connections):
    """ By default, coefficients are expressed as:
            "coefficient number ... parameters (k, r0 etc.)"
        This determines them in terms of atom types, i.e. this allows you to
            ask what is the bond type between atom types i and j (and k, l for
            angles and dihedrals)

            Input:
                - networkx graph generated from lmp2nx()
                - dictionary of coefficients in the LAMMPS format
                - dictionary of connections in the LAMMPS format
            Output:
                - dictionary of dictionary of coefficients with
                keys 'type_i-type_j-...' = bonded_type (includes reverse keys)

        N.B. requires at least one of each bond/angled/dihedral defined in
                coeffs to exist
    """
    type_coeffs={}
    for bonded, values in lammps_coefficients.iteritems():
        type_coeffs[bonded] = {}
        i = 0
        for v in values:
            btype = int(v[0])
            # Iterate over all connections looking for bond/angle/dihedral
            #     of type btype
            #    NB Can't just stop at first match, as bond types etc. can be
            #        multiply defined i.e. bond type 1 could be between atom
            #        types 1 and 2, and 5 and 6.
            found = 0
            for bond in connections[bonded]:
                if int(bond[1]) == btype:
                    found = int(bond[0])
                    atoms = np.array(bond[2:], dtype=int)
                    atom_types = [str(g.node[a]['atom_type']) for a in atoms]
                    # Dictionary key in the form atom_i-atom_j-...
                    key = '-'.join(atom_types)
                    rkey = '-'.join(atom_types[::-1])
                    type_coeffs[bonded][key] = btype
                    type_coeffs[bonded][rkey] = btype


    return type_coeffs



def _removeDuplicates(array):
    """ Removes duplicates from list
        Input:
            - list
        Output:
            - list with duplicates removed
    """
    seen = {}
    result = []
    for item in array:
        if item in seen.keys(): continue
        seen[item] = 1
        result.append(item)

    return result



def _removeDuplicatesArray(lists):
    """ Removes duplicates from list of arrays, checking itself and the
    mirrored list

        Input:
            - list of lists
        Output:
            - list of lists with duplicates and reversed duplicates removed

    """

    def idfun(x):
        return [str(x),str(x[::-1])]
    seen = {}
    result = []
    for item in lists:
        markers = idfun(item)
        if markers[0] in seen or markers[1] in seen: continue
        seen[markers[0]] = 1
        result.append(item)

    return result



def main(data_file=None, new_file=None):
    if data_file is None and new_file is None:
        data_file = sys.argv[1]
        new_file = sys.argv[2]

    d, graph, lammps_coefficients, type_key = lmp2nx(data_file)
    connections = compute_bonded_information(graph, type_key)
    new_d = write_lammps_output(graph, lammps_coefficients, connections, new_file)

    return new_d



if __name__ == "__init__":
    main()
