import lmp2nx as ln
import nx2opls as no
import numpy as np
import math, sys

def pbc_distance(x0, x1, box):
    delta = np.abs(x1 - x0)
    delta = np.where(delta > 0.5 * box, box - delta, delta)
    return (delta ** 2).sum(axis=-1)


def _random_sphere(r):
    theta = 2.*math.pi*np.random.random()
    phi   = math.acos(2.*np.random.random()-1)
    x     = r*math.cos(theta)*math.sin(phi)
    y     = r*math.sin(theta)*math.sin(phi)
    z     = r*math.cos(phi)
    return np.array([x,y,z])


def _opls_to_atom_type(opls_type, network):
    """ Return atom type corresponding to opls_nonbonded_type if found in network,
    otherwise return first available atom type """
    types = {}
    for n, atom in network.nodes_iter(data=True):
        ot = atom['opls_nonbonded_type']
        at = atom['atom_type']
        if ot in types.keys():
            assert types[ot] == at
        else:
            types[ot] = at
    if opls_type in types.keys():
        return types[opls_type]
    else:
        return max(types.values())+1


def _change_hydrogen_types(atom_ids, network):
    for atom in atom_ids:
        hydrogens = [n for m, n in network.edges_iter(atom) if network.node[n]['opls_bonded_type']=='HC']
        assert len(hydrogens) == 1
        network.node[hydrogens[0]].update(hc_of_ct_params)
    return


def _change_carbon_types(atom_ids, network, n_hydrogens=1):
    for atom in atom_ids:
        hydrogens = [n for m, n in network.edges_iter(atom) if network.node[n]['opls_bonded_type']=='HC']
        assert len(hydrogens) == n_hydrogens
        if n_hydrogens == 1:    # C of CH
            network.node[atom].update(ct_of_ch_params)
        elif n_hydrogens == 2:  # C of CH2
            network.node[atom].update(ct_of_ch2_params)
        else:
            raise StandardError
    return


def _change_hydrogen_bond_types(atom_ids, network, n_hydrogens=2):
    # Change CM-HC to CT-HC
    for atom in atom_ids:
        hydrogens = [n for m,n in network.edges_iter(atom) if network.node[n]['opls_bonded_type']=='HC']
        assert len(hydrogens) == n_hydrogens
        for h in hydrogens:
            network.edge[atom][h]['bond_type'] = ct_hc_bond_type
            network.edge[h][atom]['bond_type'] = ct_hc_bond_type
    return


def _change_carbon_bond_types(atom_ids, network):
    # Change CM-CT to CT-CT
    for atom in atom_ids:
        carbons = [n for m,n in network.edges_iter(atom) if network.node[n]['opls_bonded_type']=='CT']
        for c in carbons:
            network.edge[atom][c]['bond_type'] = ct_ct_bond_type
            network.edge[c][atom]['bond_type'] = ct_ct_bond_type
    return


def _add_hydrogen(atom_ids, network, r_hc=1.09):
    # Add a hydrogen atom randomly placed on a sphere of radius r_hc, bonded to atom
    for atom in atom_ids:
        c_pos = network.node[atom]['position']
        h_pos = c_pos + _random_sphere(r_hc)
        nmax  = len(network.nodes())
        network.add_node(nmax+1)
        network.add_edge(atom, nmax+1, bond_type=ct_hc_bond_type)
        network.node[nmax+1].update(hc_of_ct_params)
        network.node[nmax+1]['position'] = h_pos
        network.node[nmax+1]['mol'] = network.node[atom]['mol']
    return


def hydrogenate(i, j, network):
    """
    Change the hydrogen types of i and j, and add a hydrogen to each.
    Change atom types i and j (C of CH2).
    Change the bond type between i and j (CM-CM -> CT-CT).
    Change the bond type between i, j and their hydrogens (CM-HC -> CT-HC)
    (i and j are neighbouring CM atoms)
    """
    _change_hydrogen_types([i,j], network)
    _add_hydrogen([i,j], network)
    _change_carbon_types([i,j], network, n_hydrogens=2)
    network.edge[i][j]['bond_type'] = ct_ct_bond_type
    network.edge[j][i]['bond_type'] = ct_ct_bond_type
    _change_hydrogen_bond_types([i,j], network, n_hydrogens=2)
    _change_carbon_bond_types([i,j], network)

    return


def cross_link(i, j, x, y, network):
    """ Form a cross-link between i and x and change their hydrogen types.
    Change the hydrogen types of i and x.
    Change the hydrogen types of j and y, and add a hydrogen to each.
    Change atom types i and x (C of CH), j and y (C of CH2)
    Change the bond type between i and j, x and y (CM-CM -> CT-CT).
    Change the bond type between i, j, x, y and their hydrogens (CM-HC -> CT-HC)
    (i and j, x and y are neighbouring CM atoms) """
    _change_hydrogen_types([i,j,x,y], network)
    _add_hydrogen([j,y], network)
    _change_carbon_types([i,x], network, n_hydrogens=1)
    _change_carbon_types([j,y], network, n_hydrogens=2)
    network.edge[i][j]['bond_type'] = ct_ct_bond_type
    network.edge[j][i]['bond_type'] = ct_ct_bond_type
    network.edge[x][y]['bond_type'] = ct_ct_bond_type
    network.edge[y][x]['bond_type'] = ct_ct_bond_type
    _change_hydrogen_bond_types([i,x], network, n_hydrogens=1)
    _change_hydrogen_bond_types([j,y], network, n_hydrogens=2)
    _change_carbon_bond_types([i,j,x,y], network)
    network.add_edge(i, x, bond_type=ct_ct_bond_type)

    return



def count_atom_type(network, atype):
    nstr = len(atype)
    ki   = 'opls_bonded_type'
    nt   = len([n for n,atom in network.nodes_iter(data=True) if atom[ki][:nstr]==atype])
    return nt


def _isCarbon(n, network):
    atom = network.node[n]
    return atom['opls_bonded_type'][0] == 'C'


def _isCT(n, network):
    atom = network.node[n]
    return atom['opls_bonded_type'] == 'CT'


def _isCM(n, network):
    atom = network.node[n]
    return atom['opls_bonded_type'] == 'CM'


def _isCTorCM(n, network):
    return _isCM(n,network) or _isCT(n,network)


def _isCZ(n, network):
    atom = network.node[n]
    return atom['opls_bonded_type'] == 'CZ'


def _isHC(n, network):
    atom = network.node[n]
    return atom['opls_bonded_type'] == 'HC'


def isTripleCarbon(n, network):
    ans = False
    if _isCarbon(n, network) is True:
        subcount = 0
        for i,j in network.edges_iter(n):
            subcount += int(_isCT(j, network))

        if subcount == 3:
            assert len(network.edges(n)) == 4
            ans = True
    return ans


def count_triple_carbons(network):
    count = 0
    for n, atom in network.nodes_iter(data=True):
        count += int(isTripleCarbon(n, network))

    return count


def get_backbone(network, nmols=1):
    # Find chain ends
    chain_ends = []
    for n, atom in network.nodes_iter(data=True):
        if atom['opls_bonded_type'] != 'CT':
            continue
        ccount = 0
        for i, j in network.edges_iter(n):
            ccount += int(_isCT(j, network)) + int(_isCM(j, network))
        if ccount == 1:
            chain_ends.append(n)
    assert len(chain_ends) == 2*nmols

    for mol_i in range(nmols):
        start, end = np.split(np.array(chain_ends), nmols)[mol_i]

        # Find backbone
        ni   = start
        backbone = [start]
        while ni != end:
            assert _isCTorCM(ni, network)
            triple_i = isTripleCarbon(ni, network)
            network.node[ni]['cl'] = triple_i
            nexti = None
            for i,j in network.edges_iter(ni):
                # Don't connect cross-linked sites, skipping the rest of the chain
                if triple_i is False:
                    if _isCTorCM(j, network) is True and j not in backbone:
                        nexti = j
                else:
                    if _isCTorCM(j, network) is True and j not in backbone and isTripleCarbon(j, network) is False:
                        nexti = j
            assert nexti is not None
            assert network.node[nexti]['mol'] == mol_i+1
            backbone.append(nexti)
            ni = nexti
        nct = sum([int(_isCTorCM(n, network)) for n,atom in network.nodes_iter(data=True) if atom['mol']==mol_i+1])
        assert nct == len(backbone)
        network.node[ni]['cl'] = isTripleCarbon(ni, network)
        
        backbones.append(np.copy(backbone))

    return backbones


def compute_cl_bond_distances(network, backbone):
    cl_sites = [i for i in backbone if network.node[i]['cl']==True]
    cl_pos   = [backbone.index(i) for i in cl_sites]
    dist     = np.diff(cl_pos)
    return dist


# Initialise
if len(sys.argv) != 1:
    if len(sys.argv) == 5:
        dataFile   = sys.argv[1]
        outFile    = sys.argv[2]
        clFraction = float(sys.argv[3])/100.
        r2max      = float(sys.argv[4])**2
        clStep     = None
        nclSteps   = None
    elif len(sys.argv) == 7:
        dataFile   = sys.argv[1]
        outFile    = sys.argv[2]
        clFraction = float(sys.argv[3])/100.
        r2max      = float(sys.argv[4])**2
        clStep     = int(sys.argv[5])
        nclSteps   = int(sys.argv[6])
    else:
        raise StandardError('ERROR: incorrect number of input parameters (infile, outfile, clFraction, rmax)')

    fflocation='./gmx/'

    lammps_data, network, lammps_coefficients, bond_types = ln.lmp2nx(dataFile)
    opls_types = no.define_opls_types(network, bond_types, lammps_coefficients, fflocation)
    bond_type_nums = no.equivalent_bonds(opls_types, bond_types)
    unit_cell  = np.array(network.graph['unit_cell'])
    box_length = unit_cell[:,1] - unit_cell[:,0]

    # Set up and define parameters
    bond_types = no.numbers_to_types(opls_types, bond_type_nums)['bond']
    ct_hc_bond_type = bond_types['CT-HC']
    ct_ct_bond_type = bond_types['CT-CT']
    hc_of_ct_params = {'atom_type': _opls_to_atom_type('opls_140', network),
     'charge': 0.06,
     'eps': 0.03,
     'image': [0,0,0],
     'mass': 1.008,
     'opls_bonded_type': 'HC',
     'opls_nonbonded_type': 'opls_140',
     'sigma': 2.5}
    ct_of_ch_params  = {'atom_type': _opls_to_atom_type('opls_137', network),
     'charge': -0.06,
     'eps': 0.066,
     'image': [0,0,0],
     'mass': 12.011,
     'mol': 1,
     'opls_bonded_type': 'CT',
     'opls_nonbonded_type': 'opls_137',
     'sigma': 3.5}
    ct_of_ch2_params = {'atom_type': _opls_to_atom_type('opls_136', network),
     'charge': -0.12,
     'eps': 0.066,
     'image': [0,0,0],
     'mass': 12.011,
     'opls_bonded_type': 'CT',
     'opls_nonbonded_type': 'opls_136',
     'sigma': 3.5}


    # list all double bonded carbons (CM) and a dictionary of their partner CMs
    cm_list    = [n for n, atom in network.nodes(data=True) if atom['opls_bonded_type']=='CM']
    np.random.shuffle(cm_list)  # want to search through the list at random
    cm_pos     = [network.node[i]['position'] for i in cm_list]
    cm_bonded_to = {}
    for cm in cm_list:
        cm_bonded_to[cm] = [j for i,j in network.edges_iter(cm)
            if network.node[j]['opls_bonded_type']=='CM'][0]

    n_cm       = len(cm_list)
    assert n_cm % 2 == 0
    n_cl       = 0

    # if cross-linking in steps, need counts for initial doubly bonded carbons
    if clStep is not None:
        n_c_all   = count_atom_type(network, 'C')
        n_n_all   = count_atom_type(network, 'NZ')
        n_acn     = n_n_all
        n_c_but   = n_c_all - 3*n_n_all 
        n_but     = n_c_but/4
        n_cm_init = n_c_but/2
        n_cm_stop = int(float(n_cm_init)/float(nclSteps))
        assert n_c_but % 4 == 0
        assert n_but + n_acn == 1000
        assert len(cm_list) <= n_cm_init
        assert n_cm_stop > 1 

    modified = []
    n_hyd = 0
    for i in cm_list:
        j = cm_bonded_to[i]
        assert i!=j

        if i in modified or j in modified:
            assert i in modified and j in modified
            continue

        # decide whether to hydrogenate or cross-link
        if np.random.rand() > clFraction:
            print 'Hydrogenating...'
            hydrogenate(i, j, network)
            n_hyd+=2
        else:
            print 'Attempting to cross-link...'
            assert network.node[i]['opls_bonded_type']=="CM" and network.node[j]['opls_bonded_type']=="CM"
            # do a PBC distance search
            cm_remaining=[n for n,atom in network.nodes_iter(data=True) if n!=i and n!=j
                and atom['opls_bonded_type']=='CM']
            cm_remaining_pos=[network.node[n]['position'] for n in cm_remaining]
            if len(cm_remaining) == 0:
                print '...run out of cross-linking sites.'
                hydrogenate(i, j, network) 
                n_hyd += 2
                break
            dist2 = pbc_distance(network.node[i]['position'], cm_remaining_pos, box_length)
            r2min = min(dist2)
            if r2min > r2max:
                print '...nothing close enough. Hydrogenating.'
                hydrogenate(i, j, network) # too far to cross-link
                n_hyd += 2
            else:
                x = cm_remaining[np.argmin(dist2)]
                y = cm_bonded_to[x]
                assert x!=y and x!=i and y!=j
                cross_link(i, j, x, y, network)
                print '...successful.'
                n_cl += 2
                modified += [x,y]
        print ''
        modified += [i,j]
        if clStep is not None:
            if clStep != nclSteps:
               if len(modified) > n_cm_stop:
                   break
            else:
                # Don't terminate early if last cl step 
                continue


    if clStep is not None:
        n_cm_act   = count_triple_carbons(network)
        cl_success = 100.*float(n_cm_act)/(float(n_cm_init)/2.)
        target     = clFraction*100.*float(clStep)/float(nclSteps)
    else:
        cl_success = 100.*float(n_cl)/(n_cm/2.)
        target     = clFraction*100.

    print 'n_cl %s n_hyd %s' % (n_cl, n_hyd)
    print 'Target = %3.1f%%. \t Cross-linked = %4.2f%%.' % (target, cl_success)
    network.graph['comment']="CL fraction = %4.2f %%" % cl_success
