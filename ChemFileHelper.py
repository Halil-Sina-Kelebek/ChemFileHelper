import numpy as np
import scipy.sparse as sp
import random
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import typing_extensions
import teaserpp_python                                              
import periodictable
import networkx as nx
from typing import Tuple, Dict, List, Any, overload, Optional
from typing_extensions import Literal
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from simpleicp import PointCloud, SimpleICP
from networkx.algorithms import isomorphism
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from numba import njit, prange
from scipy.spatial.distance import cdist
from ase import Atoms
from ase.visualize import view
from  IPython.display import HTML
from collections import defaultdict
matplotlib.use('TkAgg')

from collections import Counter
from itertools import combinations, permutations
import nglview as nv

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

an = {"H"  : 1,
      "C"  : 6,
      "N"  : 7,
      "O"  : 8,
      "F"  : 9,
      "S"  : 16,
      "Se" : 34} 

cpk_colors_rgb = {
    1: np.array([255, 255, 255]),  # white
    2: np.array([0, 255, 255]),    # cyan
    3: np.array([148, 0, 211]),    # violet
    4: np.array([0, 100, 0]),      # dark green
    5: np.array([245, 245, 220]),  # beige
    6: np.array([0, 0, 0]),        # black
    7: np.array([0, 0, 255]),      # blue
    8: np.array([255, 0, 0]),      # red
    9: np.array([0, 255, 0]),      # green
    10: np.array([0, 255, 255]),   # cyan
    11: np.array([148, 0, 211]),   # violet
    12: np.array([0, 100, 0]),     # dark green
    13: np.array([245, 245, 220]), # beige
    14: np.array([245, 245, 220]), # beige
    15: np.array([255, 165, 0]),   # orange
    16: np.array([255, 255, 0]),   # yellow
    17: np.array([0, 255, 0]),     # green
    18: np.array([0, 255, 255]),   # cyan
    19: np.array([148, 0, 211]),   # violet
    20: np.array([0, 100, 0]),     # dark green
    21: np.array([245, 245, 220]), # beige
    22: np.array([128, 128, 128]), # grey
    23: np.array([245, 245, 220]), # beige
    24: np.array([245, 245, 220]), # beige
    25: np.array([245, 245, 220]), # beige
    26: np.array([255, 140, 0]),   # dark orange
    27: np.array([245, 245, 220]), # beige
    28: np.array([245, 245, 220]), # beige
    29: np.array([245, 245, 220]), # beige
    30: np.array([245, 245, 220]), # beige
    31: np.array([245, 245, 220]), # beige
    32: np.array([245, 245, 220]), # beige
    33: np.array([245, 245, 220]), # beige
    34: np.array([245, 245, 220]), # beige
    35: np.array([139, 0, 0]),     # dark red
    36: np.array([0, 255, 255]),   # cyan
    37: np.array([148, 0, 211]),   # violet
    38: np.array([0, 100, 0]),     # dark green
    39: np.array([245, 245, 220]), # beige
    40: np.array([245, 245, 220]), # beige
    41: np.array([245, 245, 220]), # beige
    42: np.array([245, 245, 220]), # beige
    43: np.array([245, 245, 220]), # beige
    44: np.array([245, 245, 220]), # beige
    45: np.array([245, 245, 220]), # beige
    46: np.array([245, 245, 220]), # beige
    47: np.array([245, 245, 220]), # beige
    48: np.array([245, 245, 220]), # beige
    49: np.array([245, 245, 220]), # beige
    50: np.array([245, 245, 220]), # beige
    51: np.array([245, 245, 220]), # beige
    52: np.array([245, 245, 220]), # beige
    53: np.array([148, 0, 211]),   # dark violet
    54: np.array([0, 255, 255]),   # cyan
    55: np.array([148, 0, 211]),   # violet
    56: np.array([0, 100, 0]),     # dark green
    57: np.array([245, 245, 220]), # beige
    58: np.array([245, 245, 220]), # beige
    59: np.array([245, 245, 220]), # beige
    60: np.array([245, 245, 220]), # beige
    61: np.array([245, 245, 220]), # beige
    62: np.array([245, 245, 220]), # beige
    63: np.array([245, 245, 220]), # beige
    64: np.array([245, 245, 220]), # beige
    65: np.array([245, 245, 220]), # beige
    66: np.array([245, 245, 220]), # beige
    67: np.array([245, 245, 220]), # beige
    68: np.array([245, 245, 220]), # beige
    69: np.array([245, 245, 220]), # beige
    70: np.array([245, 245, 220]), # beige
    71: np.array([245, 245, 220]), # beige
    72: np.array([245, 245, 220]), # beige
    73: np.array([245, 245, 220]), # beige
    74: np.array([245, 245, 220]), # beige
    75: np.array([245, 245, 220]), # beige
    76: np.array([245, 245, 220]), # beige
    77: np.array([245, 245, 220]), # beige
    78: np.array([245, 245, 220]), # beige
    79: np.array([245, 245, 220]), # beige
    80: np.array([245, 245, 220]), # beige
    81: np.array([245, 245, 220]), # beige
    82: np.array([245, 245, 220]), # beige
    83: np.array([245, 245, 220]), # beige
    84: np.array([245, 245, 220]), # beige
    85: np.array([245, 245, 220]), # beige
    86: np.array([0, 255, 255]),   # cyan
    87: np.array([148, 0, 211]),   # violet
    88: np.array([0, 100, 0]),     # dark green
    89: np.array([255, 192, 203]), # pink
    90: np.array([255, 192, 203]), # pink
    91: np.array([255, 192, 203]), # pink
    92: np.array([255, 192, 203]), # pink
    93: np.array([255, 192, 203]), # pink
    94: np.array([255, 192, 203]), # pink
    95: np.array([255, 192, 203]), # pink
    96: np.array([255, 192, 203]), # pink
    97: np.array([255, 192, 203]), # pink
    98: np.array([255, 192, 203]), # pink
    99: np.array([255, 192, 203]), # pink
    100: np.array([255, 192, 203]),# pink
    101: np.array([255, 192, 203]),# pink
    102: np.array([255, 192, 203]),# pink
    103: np.array([255, 192, 203]),# pink
    104: np.array([255, 192, 203]),# pink
    105: np.array([255, 192, 203]),# pink
    106: np.array([255, 192, 203]),# pink
    107: np.array([255, 192, 203]),# pink
    108: np.array([255, 192, 203]),# pink
    109: np.array([255, 192, 203]),# pink
    110: np.array([255, 192, 203]),# pink
    111: np.array([255, 192, 203]),# pink
    112: np.array([255, 192, 203]),# pink
    113: np.array([255, 192, 203]),# pink
    114: np.array([255, 192, 203]),# pink
    115: np.array([255, 192, 203]),# pink
    116: np.array([255, 192, 203]),# pink
    117: np.array([255, 192, 203]),# pink
    118: np.array([255, 192, 203]),# pink
    }

# TODO: FURTHER COMPRESSION, instead of storing a template centered around 0,0,0, representative fragment should be at location of 1 of the copies, then transformation of rets of copies will be relative to that copy, this way can save further space.

############################################################################################################
### README: Code conventions                                                                             ###
############################################################################################################

'''
Helper functions for reading files containing molecule and certain functions to process xyz data. Meant for 
use with bottom_up.py but can be used for any project working in chemical compound space. currently quite 
basic, containing certain functions to get the distance matrix, coulomb matrix, bond graph adjacency matrix, 
and xyz coordinates from distance matrix. also a point cloud alignment algorithm with compatability with 
classed point cloud data. A meaningful set of classes for atomic point clouds can be the atomic charge / 
number of atoms.  

CM - refers to coulomb matrix
DM - refers to euclidean distance matrix
AM - refers to bond based graph adjacency matrix 
'''

############################################################################################################
### Block prints                                                                               (ChatGPT) ###
############################################################################################################

class BlockPrints:
    __doc__='''
            # Example usage
            def some_function():
                print("This will be blocked")

            with BlockPrints():
                some_function()
                print("This will also be blocked")

            print("This will be printed")
            '''

    def __init__(self, block_prints=True, block_stderr=False):
        self.block_prints = block_prints
        self.block_stderr = block_stderr

    def __enter__(self):
        if self.block_prints:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if self.block_stderr:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        if self.block_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.block_stderr:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def read_xyz_JCTC(file_path, amon=True, removeHs=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    energies = {}
    elems_list = []
    Zs = []
    Xs = []
    output_len = len(lines)
    # print(file_path)
    # print(file_path)
    # print(os.path.basename(file_path))
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # print(basename)
    f = basename.split("_")
    # print(file_path, f)
    if amon:
        target = int(f[1])

        order = int(f[2][1:])
    if not amon:
        target = -1
        order = int(f[1])
    i = 0
    while i < output_len:
        if int(lines[i]):
            mol_size = int(lines[i])
            energies_line = lines[i+1]
            if len(energies_line.split('=')) == 2:
                level, str_energy = lines[i+1].split('=')
                energy = float(str_energy)
                energies[level] = energy

            else:
                split_energies = energies_line.split(" ")
                for energy_level in split_energies:
                    level, str_energy = energy_level.split("=")
                    energy = float(str_energy)
                    energies[level] = energy
            charges = np.zeros(mol_size)
            elements = np.empty(mol_size, dtype=object)
            geoms = np.zeros((mol_size, 3))
            j = 0
            while j < mol_size:
                atom = lines[i+2+j].split()
                charges[j] = symbol_to_an(atom[0])
                elements[j] = atom[0]
                geoms[j, :] = np.array(atom[1:], dtype=float)
                j+=1
            # energies.append(energy)
            # elems_list.append(elements)
            # Zs.append(charges)
            # Xs.append(geoms)
            # print(charges.shape, geoms.shape, end=", ")
            if removeHs:
                geoms = geoms[charges!=1]
                charges = charges[charges!=1]
            # print(charges.shape, geoms.shape)
        return  Molecule_JCTC(xyz=geoms, zs=charges, energies=energies, filename=basename, target=target, order=order)

def read_sdf_JCTC(file_path, amon=True, removeHs=False):
    """currently returns energies={} for the Molecule_JCTC object created"""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # print(basename)
    f = basename.split("_")
    # print(file_path, f)
    if amon:
        target = int(f[1])

        order = int(f[2][1:])
    if not amon:
        target = -1
        order = -1#int(f[1])
    geoms, charges, _ = read_sdf_file(filename=file_path)
    return  Molecule_JCTC(xyz=geoms, zs=charges, energies={}, filename=basename, target=target, order=order) # NOTE: the energy will be fixed by reading the energy file and reading the corresponding line for now, later read the variations from the sdf file

def read_pdb_JCTC(file_path, amon=True, removeHs=False):
    """currently returns energies={} for the Molecule_JCTC object created"""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    # print(basename)
    f = basename.split("_")
    # print(file_path, f)
    if amon:
        target = int(f[1])

        order = int(f[2][1:])
    if not amon:
        print()
        target = -1
        order = -1
    geoms, charges, _ = read_pdb_file(filename=file_path)
    return  Molecule_JCTC(xyz=geoms, zs=charges, energies={}, filename=basename, target=target, order=order) # NOTE: the energy will be fixed by reading the energy file and reading the corresponding line for now, later read the variations from the sdf file


        # else: i += 1


    # # Convert lists to numpy arrays with object dtype
    # coords_array = np.array(Xs, dtype=object)
    # charges_array = np.array(Zs, dtype=object)
    # elements_array = np.array(elems_list, dtype=object)
    # energies_array = np.array(energies, dtype=object)

    # # Save as NPZ file
    # np.savez(f'qm9_targets_{file_idx:02}.npz', coords=coords_array, charges=charges_array, elements=elements_array, energies=energies_array)


############################################################################################################
### Molecule datatype for amon learning curves (for JCTC amon delta learning paper)                      ###
############################################################################################################

@dataclass
class Molecule_JCTC:
    xyz      : np.ndarray
    zs       : np.ndarray
    energies : Dict[str, float] # dictionary of energy level of theory as key and value at that level
    filename : str
    target   : int
    order    : int



############################################################################################################
### Mol obj from xyz array                                                                               ###
############################################################################################################
def MolFromNumpyXYZBlock(xyz: np.ndarray, zs: np.ndarray, AM: Optional[np.ndarray]=None):
    # # Example data
    # xyz_data = np.array([
    #     [0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.5],
    # ])
    # atomic_numbers = [6, 6]  # Carbon atoms
    # adjacency_matrix = np.array([
    #     [0, 1],
    #     [1, 0]
    # ])
    if AM is not None:
        bonds = {1: Chem.rdchem.BondType.SINGLE,
                 2: Chem.rdchem.BondType.DOUBLE,
                 3: Chem.rdchem.BondType.TRIPLE,
                 4: Chem.rdchem.BondType.QUADRUPLE,
                 5: Chem.rdchem.BondType.QUINTABLE,
                 6: Chem.rdchem.BondType.HEXTUPLE}

    # Step 1: Create an empty molecule
    mol = Chem.RWMol()

    # Step 2: Add atoms
    for z in zs:
        atom = Chem.Atom(z)
        mol.AddAtom(atom)

    # Step 3: Add bonds using the adjacency matrix
    if AM is not None:
        num_atoms = len(zs)
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                if AM[i, j] != 0:
                    mol.AddBond(i, j, bonds[AM[i, j]])  # You might need to adjust the bond type accordingly

    # Step 4: Set atomic coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(xyz[i]))
    mol.AddConformer(conf)

    # Convert to a final Mol object and sanitize
    mol = mol.GetMol()
    AllChem.SanitizeMol(mol)
    
    return mol

############################################################################################################
### Reading files                                                                                        ###
############################################################################################################

def read_pdb_file(filename: str, removeHs=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:

    # Force correct file type. # TODO: Should really use os.dir.path or Path objects, no guarantee there will be '.' in [-4] index if file type at end isnt of length 3
    if filename[-4:] != ".pdb" and filename[-4] != ".":
        filename += ".pdb"
    elif filename[-4:] != ".pdb": raise Exception(f"Incorrect file type. - {filename[-4:]}")

    # Open file using rdkit.
    mol = Chem.rdmolfiles.MolFromPDBFile(filename, removeHs=removeHs)
    # mol = Chem.AddHs(mol, addCoords=True)
    # Initialize lists.
    z = []
    atom_types = []

    if mol is None:
        print("Failed to load the molecule from the PDB file.")
    else:
        # get xyz coordinates
        xyz = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            # get symbols and charges
            atom_types.append(atom.GetSymbol())
            z.append(an[atom.GetSymbol()])
            
    atom_types = np.array(atom_types)
    z = np.array(z)

    return xyz, z, atom_types

def read_smiles(smiles: str, addHs=False):
    mol = Chem.MolFromSmiles(smiles)
    z = []
    atom_types = []

    if mol is None:
        print("Failed to load the molecule from SMILES.")
    else:
        # get xyz coordinates
        print(smiles)
        # if mol.GetNumAtoms() == 1:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)

        xyz = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            # get symbols and charges
            atom_types.append(atom.GetSymbol())
            z.append(an[atom.GetSymbol()])
            
    atom_types = np.array(atom_types)
    z = np.array(z)

    return xyz, z, atom_types



############################################################################################################

# def read_xyz_file(filename: str, removeHs=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:

#     # Force correct file type.
#     if filename[-4:] != ".xyz" and filename[-4] != ".":
#         filename += ".xyz"
#     elif filename[-4:] != ".xyz": raise Exception(f"Incorrect file type. - {filename[-4:]}")
#     with open(filename) as f:
#         lines = f.readlines()
#         length = int(lines[0])
#         print(lines[0:1], lines[0:0], lines[2:length+2])
#         xyzdat_lst = lines[0:1] + lines[2:length+2]
#         xyzdat = "".join(xyzdat_lst)
#     # Open file using rdkit.
#     mol = Chem.MolFromXYZBlock(xyzdat)
#     print(xyzdat)
#     # mol = Chem.AddHs(mol, addCoords=True)
#     # Initialize lists.
#     z = []
#     atom_types = []

#     if mol is None:
#         print("Failed to load the molecule from the XYZ file.")
#         raise Exception
#     else:
#         # get xyz coordinates
#         xyz = mol.GetConformer().GetPositions()
#         for atom in mol.GetAtoms():
#             # get symbols and charges
#             atom_types.append(atom.GetSymbol())
#             z.append(an[atom.GetSymbol()])
            
#     atom_types = np.array(atom_types)
#     z = np.array(z)
#     # print(z)
#     # print(atom_types)
#     return xyz, z, atom_types

# def read_xyz_file(filename: str, removeHs=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     with open(filename, 'r') as f:
#         lines = f.readlines()
        
#         # The first line indicates the number of atoms in the molecule
#         num_atoms = int(lines[0].strip())
        
#         # The atoms and their XYZ coordinates are listed after the first two lines
#         atom_types = []
#         zs = []
#         xyz = []
#         for i in range(2, 2 + num_atoms):
#             tokens = lines[i].split()
#             atom_type = tokens[0]
#             print(atom_type)
#             if atom_type == "H" and removeHs:
#                 continue
#             x, y, z = map(float, tokens[1:4])
#             atom_types.append(atom_type)
#             zs.append(symbol_to_an(atom_type))
#             xyz.append((x, y, z))
#         xyz = np.array(xyz)
#         zs = np.array(zs)
#         atom_types = np.array(atom_types)
#         print(f"XYZ, ZS, AT:\n{xyz, zs, atom_types}")
#     return xyz, zs, atom_types

def mol_from_xyz_file_smiles(filename: str, removeHs=False):
    with open(filename, 'r') as f:
        smiles = f.readlines()[-2]

        print(f"LINES:\n{smiles}")
        mol = Chem.MolFromSmiles(smiles)#, removeHs=removeHs)
        if not removeHs:
            Chem.addHs(mol)
        return mol

############################################################################################################

def read_sdf_file(filename: str, removeHs=False) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    # Force correct file type.
    if filename[-4:] != ".sdf" and filename[-4] != ".":
        filename += ".sdf"
    elif filename[-4:] != ".sdf": raise Exception(f"Incorrect file type. - {filename[-4:]}")

    # Open file using rdkit.
    supplier = Chem.SDMolSupplier(filename, removeHs=removeHs)
    
    # Initialize lists.
    atom_types = []
    z = []

    for mol in supplier:
        # get xyz coordinates
        # mol = Chem.AddHs(mol, addCoords=True)
        xyz = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            # get symbols and charges
            atom_types.append(atom.GetSymbol())
            z.append(an[atom.GetSymbol()])

    atom_types = np.array(atom_types)
    z = np.array(z)
    # print(z)
    # print(atom_types)
    # print(xyz)
    return xyz, z, atom_types

############################################################################################################

def read_file(filename: str, removeHs = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    assert filename[-4:] == ".sdf" or filename[-4:] == ".pdb" or filename[-4:] == ".xyz", f"{filename} is of incorrect file type, sdf or pdb accepted."
    print(filename[-4:])
    return read_sdf_file(filename, removeHs=removeHs) if filename[-4:] == ".sdf" else read_pdb_file(filename, removeHs=removeHs) if filename[-4:] == ".pdb" else read_xyz_file(filename, removeHs=removeHs)

def file_to_rdkit_mol(filename: str, removeHs = True) -> Chem.rdchem.Mol:
    assert filename[-4:] == ".sdf" or filename[-4:] == ".pdb", f"{filename} is of incorrect file type, sdf or pdb accepted."
    print(filename[-4:])
    if filename[-4:] == ".sdf":
        supplier = Chem.SDMolSupplier(filename, removeHs=removeHs)
        for m in supplier:
            mol = m
            break
    if filename[-4:] == ".pdb":
        mol = Chem.rdmolfiles.MolFromPDBFile(filename, removeHs=removeHs)
    return mol


############################################################################################################
### Atomic Number to symbol                                                                              ###
############################################################################################################
# (ChatGPT)

# Function to convert atomic number to atomic symbol
def an_to_symbol(atomic_number):
    try:
        element = periodictable.elements[atomic_number]
        # Convert the atomic symbols array to a chararray
        atom_types = element.symbol
        return atom_types
        # atomic_symbols_chararray = np.chararray(atom_types.shape, itemsize=2)
        # atomic_symbols_chararray[:] = atom_types

        # return atomic_symbols_chararray
    except KeyError:
        return "Unknown"

# Vectorize the function to work with NumPy arrays
v_an_to_symbol = np.vectorize(an_to_symbol)

############################################################################################################
### Symbol to Atomic Number                                                                              ###
############################################################################################################

# (ChatGPT)

# Function to convert atomic symbool to atomic number
def symbol_to_an(symbol : str):
    try:
        element = periodictable.elements.symbol(symbol)
        atomic_number = element.number
        return atomic_number
    except periodictable.core.ElementNotFoundError: # NOTE: This exception doesnt exist. not a huge deal but should be fixed.
        return -1

# if __name__ == "__main__":
#     print(symbol_to_an("He"))
#     raise Exception

############################################################################################################
### Get Euclidean distance matrix + bond graph adjacency matrix                                          ###
############################################################################################################

# def get_dists_from_pos(xyz: np.ndarray) -> np.ndarray: 
#     rel_xyz = xyz[:,np.newaxis,:] - xyz
#     dists = np.linalg.norm(rel_xyz, ord=2, axis=2)
#     return dists

def get_dists_from_pos(xyz: np.ndarray) -> np.ndarray:
    dists = cdist(xyz, xyz, metric='euclidean')
    return dists

def get_bond_adjacency_matrix(filename: str, bond_order_weighted=False, removeHs=False):
    # (Partial ChatGPT)
    mol = Chem.SDMolSupplier(filename, removeHs=removeHs)[0] if filename[-4:] == ".sdf" else Chem.rdmolfiles.MolFromPDBFile(filename, removeHs=removeHs) if filename[-4:] == ".pdb" else  mol_from_xyz_file_smiles(filename, removeHs=removeHs) 
    # mol = Chem.AddHs(mol, addCoords=True)
    if mol is None:
        raise ValueError(f"Failed to read a valid molecule from the {filename[-4:]} file.")
    Chem.GetSSSR(mol)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if bond_order_weighted:
            weight = bond.GetBondTypeAsDouble()
        else: weight = 1
        adjacency_matrix[atom1, atom2] = weight
        adjacency_matrix[atom2, atom1] = weight
    print(adjacency_matrix.shape)
    return adjacency_matrix

def get_composition_weighted_AM(AM: np.ndarray, z: np.ndarray):
    z_ixz_j = np.outer(z, z)
    return AM * z_ixz_j

############################################################################################################
### FOR jctc_learning_curve.ipynb, expensive way (without prunings yet) of finding all possible sub DMs in larger fragment, should be sufficent for QM9 size mols but too expensive (likely) for large molecules ###
############################################################################################################

def find_charge_combinations(q1, q2):
    #NOTE: WASTEFUL TO GENERATE ALL COMBS SO FIX LATER (USE SMT LIKE THE BACKTRACK VERSION BELOW COMMENTED OUT) IF THIS IS NOW IN ChemFileHelper CHECK OUT THE testing_combinations.py FILE IN MY AQML DIRECTORY

    # NOTE 2: LATER CHECK AS AN IMEDIATE QUICK CONDITION THAT q1 CONTAINS ATLEAST ENOUGH ELEMS TO FILL OUT q2 ATLEAST 1 WAY

    # NOTE 3: EVENTUALLY MAKE THIS NUMBA COMPATIBLE
    
    # Count occurrences of each element in q2 (amon charges)
    count_q2 = Counter(q2)
    # print(count_q2)
    # Generate all possible combinations of indices from q1 of length len(q2)
    all_indices = list(range(len(q1)))
    combs = list(combinations(all_indices, len(q2)))
    # print(combs)
    valid_combinations = []
    
    # Check each combination if it matches the count of elements in q2
    for comb in combs:
        selected_elements = q1[list(comb)]
        count_selected = Counter(selected_elements)
        
        if count_selected == count_q2:
            valid_combinations.append(list(comb))
    
    return np.array(valid_combinations)

############################################################################################################

def sort_and_permute_charges(combination, q1):
    # Sort indices by the values in b in descending order
    sorted_indices = sorted(combination, key=lambda x: q1[x], reverse=True)
    
    # Group indices by their corresponding value in b
    value_to_indices = defaultdict(list)
    for index in sorted_indices:
        value_to_indices[q1[index]].append(index)
    
    # Generate all permutations within each group of identical values
    grouped_permutations = []
    for indices in value_to_indices.values():
        if len(indices) > 1:
            perms = list(permutations(indices))
            grouped_permutations.append(perms)
        else:
            grouped_permutations.append([indices])
    
    # Combine the permutations
    all_combinations = []
    
    def combine_permutations(groups, current):
        if not groups:
            all_combinations.append(current)
            return
        for perm in groups[0]:
            combine_permutations(groups[1:], current + list(perm))
    
    combine_permutations(grouped_permutations, [])
    # print(grouped_permutations)
    
    return all_combinations

############################################################################################################
### Get coulomb matrix from dist matrix: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.108.058301 ###
############################################################################################################

def get_reciprocal(arr: np.ndarray, zeros: Optional[Literal["zero", "inf", "one"]] = "zero"):
    '''
    Get the elementwise reciprocal of all elements in arr.
    If arr contains zero values, the zero is kept at those indices.
    '''
    arr_c = copy.deepcopy(arr)
    arr_c[arr_c != 0] = 1/arr_c[arr_c != 0] 

    if zeros == "inf":
        arr_c[arr == 0] = np.inf
    elif zeros == "one":
        arr_c[arr == 0] = 1

    return arr_c

def reorder_coulomb_matrix(coulomb_mat: np.ndarray):
    # sort coulomb matrix
    cm_summed = np.sum(coulomb_mat, axis=1)
    sorted_inds = np.argsort(-cm_summed)
    coulomb_mat = coulomb_mat[np.ix_(sorted_inds,sorted_inds)]
    return coulomb_mat, sorted_inds

############################################################################################################

def get_coulomb_matrix(dist_mat: np.ndarray, z: np.ndarray, supress_prints=True) -> Tuple[np.ndarray, np.ndarray]:
    with BlockPrints(block_prints=supress_prints):
        recip_dist = get_reciprocal(dist_mat)
        
        # get all products of z_i * z_j as a matrix
        z_ixz_j = np.outer(z, z) 

        # Calculate values for diagonal entries
        diag_elems = np.diag(0.5*np.power(z, 2.4))

        # get coulomb matrix
        coulomb_mat = np.multiply(z_ixz_j, recip_dist) + diag_elems

        # sort coulomb matrix
        coulomb_mat, sorted_inds = reorder_coulomb_matrix(coulomb_mat)
        # print(coulomb_mat)
    return coulomb_mat, sorted_inds

############################################################################################################

def get_padded_local_coulomb_matrix(dist_mat: np.ndarray, z: np.ndarray, pad: int, supress_prints=True) -> Tuple[np.ndarray, np.ndarray]:
    with BlockPrints(block_prints=supress_prints):
        print()
        mol_size = len(z)
        print(mol_size)
        padded_cm = np.zeros((pad, pad))
        padded_z = np.zeros(pad)
        recip_dist = get_reciprocal(dist_mat)
        
        # get all products of z_i * z_j as a matrix
        z_ixz_j = np.outer(z, z) 

        # Calculate values for diagonal entries
        diag_elems = np.diag(0.5*np.power(z, 2.4))

        # get coulomb matrix
        coulomb_mat = np.multiply(z_ixz_j, recip_dist) + diag_elems

        # sort coulomb matrix
        coulomb_mat = -np.sort(-coulomb_mat, axis=1)
        padded_cm[:mol_size, :mol_size] = coulomb_mat
        padded_z[:mol_size] = z

        # print(coulomb_mat)
    return padded_cm, padded_z
import time
from numba import config
@njit
def unique_with_counts(arr): 
    '''np.unique(..., return_counts=True) not yet supported by numba so manual implementation'''
    # Sort the array
    sorted_arr = np.sort(arr)
    
    length = sorted_arr.shape[0]
    # Initialize the unique values and their counts
    unique_values = np.zeros(length)#, dtype=sorted_arr.dtype)
    counts = np.zeros(length)#, dtype=np.int64)
    
    # Initialize the first value
    current_value = sorted_arr[0]
    current_count = 1
    unique_idx = 0
    
    # Iterate over the sorted array
    for i in range(1, length):
        if sorted_arr[i] == current_value:
            current_count += 1
        else:
            unique_values[unique_idx] = current_value
            counts[unique_idx] = current_count
            unique_idx += 1
            current_value = sorted_arr[i]
            current_count = 1
    
    # Append the last unique value and count
    unique_values[unique_idx] = current_value
    counts[unique_idx] = current_count
    unique_idx += 1
    
    return unique_values[:unique_idx], counts[:unique_idx]
@njit
def compute_charge_padding(pad_array, arr):
    max_classes = pad_array[0]
    max_occurrences = pad_array[1]
    # Get unique values and their counts from the initial array
    unique_values, counts = unique_with_counts(arr)
    
    # Initialize the result array
    fewer_occurrences = np.zeros(len(max_classes), dtype=np.int64)
    
    # Iterate over the maximum occurrences and their corresponding classes
    for i in range(len(max_classes)):
        max_class = max_classes[i]
        max_count = max_occurrences[i]
        
        # Find the current count in the unique_values and counts arrays
        current_count = 0
        for j in range(len(unique_values)):
            if unique_values[j] == max_class:
                current_count = counts[j]
                break
        
        # Calculate the fewer occurrences
        fewer_occurrences[i] = max_count - current_count
        # Generate the new array based on fewer_occurrences
        z_padding = []
        for i in range(len(max_classes)):
            z_padding.extend([max_classes[i]] * fewer_occurrences[i])
        z_padding = np.array(z_padding)
        CM_padding = np.zeros_like(z_padding) 

    return z_padding, CM_padding


def get_weight_matrix(DM: np.ndarray, sigma: float, kernel: Optional[Literal["laplacian", "gaussian"]] = "laplacian", DM_presorted: Optional[bool]=False, DM_sort_inds: Optional[np.ndarray]=None, z: Optional[np.ndarray]=None, lcm_hpad: Optional[np.ndarray]=None, lcm_vpad: Optional[int]=None ):
    """Get the distance based weight matrix for coulomb matrix representations, if DM is not presorted and sort inds are not passed in, sorting will be done assuming binned_local_coulomb_matrix representation as is defined in ChemHelperFile"""
    if not DM_presorted:
        if type(DM_sort_inds) != np.ndarray:
            assert type(z) == np.ndarray, "if distance matrix is not presorted and sort indeces are not passed in, charges needed to figure out the sort order through binned_local_CM representation, its suggested to pass in presorted DM or pass in the DM_sort_inds if not"
            _,_, DM_sort_inds = get_binned_local_coulomb_matrix(DM, z, pad_array = lcm_hpad, vert_pad = lcm_vpad)
        
        sorted_DM = np.zeros_like(DM)
        for i in range(DM.shape[0]): 
            sorted_DM[i,:]  = DM[i, DM_sort_inds[i]] 
        DM = sorted_DM

    if kernel == "laplacian":
        weights = np.exp(-sorted_DM/sigma)    # laplacian weights
    elif kernel == "gaussian":
        weights = np.exp(-sorted_DM**2/sigma) # gaussian  weights
    else: raise ValueError(f"unrecognized kernel type: {kernel}, currently only 'laplacian' and 'gaussian' kernels accepted.")
    return weights, sorted_DM


def get_binned_local_coulomb_matrix(dist_mat: np.ndarray, z: np.ndarray, pad_array: Optional[np.ndarray]=None, vert_pad: Optional[int]=None, supress_prints=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structure of pad array -> (2,n) where n is the number of distinct element types present. first row is the atomic numbers, second row is the number of atoms of that type present at max occurence"""

    with BlockPrints(block_prints=supress_prints):
        print()
        mol_size = len(z)
        print(mol_size)

        recip_dist = get_reciprocal(dist_mat)
        
        # get all products of z_i * z_j as a matrix
        z_ixz_j = np.outer(z, z) 

        # Calculate values for diagonal entries
        diag_elems = np.diag(0.5*np.power(z, 2.4))

        # get coulomb matrix
        coulomb_mat = np.multiply(z_ixz_j, recip_dist) + diag_elems
        # print(coulomb_mat)
        # print(dist_mat)

        row_sort_inds = []
        
        print(z)
        charges = z
        all_coulomb_interactions = coulomb_mat
        if type(pad_array) == np.ndarray:
            z_pad, CM_pad = compute_charge_padding(pad_array, z)
            print(charges, z_pad)
            print(charges.shape, z_pad.shape)
            charges = np.concatenate((charges, z_pad))
            print(charges)
            print(charges.shape)
            # raise
            CM_pad = np.tile(CM_pad, (all_coulomb_interactions.shape[0], 1))
            all_coulomb_interactions = np.concatenate((all_coulomb_interactions, CM_pad),axis=1)

        sorted_cm = np.empty_like(all_coulomb_interactions)

        for i in range(coulomb_mat.shape[0]):
            # Prepare keys for lexsort
            coulomb_interactions = all_coulomb_interactions[i, :]
            # if type(pad_array) == np.ndarray:
            #     z_pad, CM_pad = compute_charge_padding(pad_array, z)
            #     coulomb_interactions = np.concatenate((coulomb_interactions, CM_pad))

            keys = (-coulomb_interactions, -charges)
            
            # Get sorted indices
            sorted_indices = np.lexsort(keys)
            print(z)
            # raise
            # Apply sorted indices to sort the row
            sorted_cm[i, :] = all_coulomb_interactions[i, sorted_indices]
            # sorted_zs = charges[sorted_indices]
            # print(sorted_zs)
            row_sort_inds.append(sorted_indices)
        row_sort_inds = np.array(row_sort_inds)
        # sort coulomb matrix

        # coulomb_mat = -np.sort(-coulomb_mat, axis=1)
        # padded_cm[:mol_size, :mol_size] = coulomb_mat
        # padded_z[:mol_size] = z

        # print(coulomb_mat)
        padded_zs = z                   
        if type(vert_pad) == int:
            vert_pad -= sorted_cm.shape[0]
            assert vert_pad >= 0, "inputted padding is too small"
            if vert_pad != 0:
                vert_CM_pad = np.zeros((vert_pad, sorted_cm.shape[1]))
                vert_zs_pad = np.zeros((vert_pad))
                sorted_cm = np.concatenate([sorted_cm, vert_CM_pad], axis=0)
                padded_zs = np.concatenate([z, vert_zs_pad])
    return sorted_cm, padded_zs, row_sort_inds

############################################################################################################
### Pad charges                                                                                          ###
############################################################################################################

def pad_charges(qs: np.ndarray,pad:Optional[int]=None):
    if pad is None:
        raise ValueError("Nvm lol need to pass pad, dont wanna program this case cause wont use it")
    padded_qs_list = []
    for i, q in enumerate(qs):
        num_zeros = pad - len(q)
        zero_elems = np.zeros(num_zeros)
        padded_q = np.concatenate([q, zero_elems])
        padded_qs_list.append(padded_q)
    padded_qs = np.array(padded_qs_list)
    return padded_qs
############################################################################################################
### Get Structure for Coulomb Matrix                                                                     ###
############################################################################################################
def seq_AM_to_xyz(AMs:np.ndarray, zs:Optional[np.ndarray]=None, AM_rep: Optional[Literal["DM", "CM"]]="CM"):
    '''
    parameters:
        AMs - np.ndarray: (m,n,n) Adjacency matrix for molecule, can take in either Distance Matrix (DM) or Coulomb Matrix (CM) as Adjaceny matrix but needs to be consistent between Adjacency matrix samples.
        z  - np.ndarray: (m,n) vector of charges for molecule. Order of charges must appear in same order as AM ordering.  
        AM_rep - Optional[Literal["DM", "CM"]]: literal to select between using DM and CM, by default takes in CM.

    return:
        xyzs - np.ndarray: (m,n,3) coordinates for each AM, Note if the Adjacency Matrix contains impossible to represent interactions in 3D (ie not geometrically feasible)
        the generated structure may be fairly unpredictible.
    '''
    if AM_rep == "DM":
        pass
    num_structures, num_atoms = AMs.shape[0], AMs.shape[1]
    xyzs = np.zeros((num_structures, num_atoms, 3))

    for i in prange(num_structures):
        xyzs[i] = AM_to_xyz(AMs[i], zs[i], AM_rep=AM_rep)

    return xyzs
def seq_AM_to_xyz(AMs:np.ndarray, zs:Optional[np.ndarray]=None, AM_rep: Optional[Literal["DM", "CM"]]="CM"):
    pass
def seq_AM_to_xyz(AMs:np.ndarray, zs:Optional[np.ndarray]=None, AM_rep: Optional[Literal["DM", "CM"]]="CM"):
    '''
    parameters:
        AMs - np.ndarray: (m,n,n) Adjacency matrix for molecule, can take in either Distance Matrix (DM) or Coulomb Matrix (CM) as Adjaceny matrix but needs to be consistent between Adjacency matrix samples.
        z  - np.ndarray: (m,n) vector of charges for molecule. Order of charges must appear in same order as AM ordering.  
        AM_rep - Optional[Literal["DM", "CM"]]: literal to select between using DM and CM, by default takes in CM.

    return:
        xyzs - np.ndarray: (m,n,3) coordinates for each AM, Note if the Adjacency Matrix contains impossible to represent interactions in 3D (ie not geometrically feasible)
        the generated structure may be fairly unpredictible.
    '''
    num_structures, num_atoms = AMs.shape[0], AMs.shape[1]
    xyzs = np.zeros((num_structures, num_atoms, 3))

    for i in prange(num_structures):
        xyzs[i] = AM_to_xyz(AMs[i], zs[i], AM_rep=AM_rep)

    return xyzs
def AM_to_xyz(AM:np.ndarray, z:Optional[np.ndarray]=None, AM_rep: Optional[Literal["DM", "CM"]]="CM", approximate: Optional[bool]=False):
    '''
    parameters:
        AM - np.ndarray: (n,n) Adjacency matrix for molecule, can take in either Distance Matrix (DM) or Coulomb Matrix (CM) as Adjaceny matrix
        z  - np.ndarray: (n,) vector of charges for molecule. Order of charges must appear in same order as CM ordering.
        AM_rep Optional[Literal["DM", "CM"]]: literal to select between using DM and CM, by default takes in CM.

    return:
        xyz - np.ndarray: (n,3) coordinates for AM, Note if the Adjacency Matrix contains impossible to represent interactions in 3D (ie not geometrically feasible)
        the generated structure may be fairly unpredictible.
    '''
    if AM_rep == "CM":
        assert z is not None, "charges, z must be passed in if using CM"
        z_ixz_j = np.outer(z, z)
        inv_DM = AM / z_ixz_j
        np.fill_diagonal(inv_DM, val=0)
        DM = get_reciprocal(inv_DM)
    elif AM_rep == "DM":
        DM = AM
    else:
        raise ValueError(f"AM_rep {AM_rep} not supported. AM_rep can take either Literal \'DM\' or \'CM\' ")

    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    DM_sqr = DM * DM

    DM_sqr_row = DM_sqr[0,:][np.newaxis,:]
    DM_sqr_col = DM_sqr[:,0][:,np.newaxis]

    M = (DM_sqr_row + DM_sqr_col - DM_sqr) / 2
    (eig_vals, eig_vecs) = np.linalg.eig(M)

    xyz = eig_vecs * np.sqrt(eig_vals)
    mask = np.isclose(eig_vals,np.zeros_like(eig_vals))==False

    xyz = xyz.T[mask].T 
    if xyz.shape[1] == 2:
        z_coords = np.zeros([xyz.shape[0],1])
        xyz = np.concatenate((xyz, z_coords), axis=1)
    elif xyz.shape[1] == 1:
        yz_coords = np.zeros([xyz.shape[0],2])
        xyz = np.concatenate((xyz, yz_coords), axis=1)
    elif xyz.shape[1] == 0: 
        raise Exception("for some reason xyz has shape[1]==0, might be that all points are on the same location or AM is 1x1 or possibly i didnt realize how the shape of xyz is changed in the case that all points are co-linear let me know if you see this message.")
    return xyz

def _DM_to_xyz(DM:np.ndarray):
    '''
    parameters:
        DM - np.ndarray: (n,n) Adjacency matrix for molecule, can take in either Distance Matrix (DM) or Coulomb Matrix (CM) as Adjaceny matrix
        z  - np.ndarray: (n,) vector of charges for molecule. Order of charges must appear in same order as CM ordering.
        AM_rep Optional[Literal["DM", "CM"]]: literal to select between using DM and CM, by default takes in CM.

    return:
        xyz - np.ndarray: (n,3) coordinates for AM, Note if the Adjacency Matrix contains impossible to represent interactions in 3D (ie not geometrically feasible)
        the generated structure may be fairly unpredictible.
    '''

    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    DM_sqr = DM * DM

    DM_sqr_row = DM_sqr[0,:][np.newaxis,:]
    DM_sqr_col = DM_sqr[:,0][:,np.newaxis]

    M = (DM_sqr_row + DM_sqr_col - DM_sqr) / 2
    (eig_vals, eig_vecs) = np.linalg.eig(M)

    xyz = eig_vecs * np.sqrt(eig_vals)
    mask = np.isclose(eig_vals,np.zeros_like(eig_vals))==False

    xyz = xyz.T[mask].T 
    if xyz.shape[1] == 2:
        z_coords = np.zeros([xyz.shape[0],1])
        xyz = np.concatenate((xyz, z_coords), axis=1)
    elif xyz.shape[1] == 1:
        yz_coords = np.zeros([xyz.shape[0],2])
        xyz = np.concatenate((xyz, yz_coords), axis=1)
    elif xyz.shape[1] == 0: 
        raise Exception("for some reason xyz has shape[1]==0, might be that all points are on the same location or AM is 1x1 or possibly i didnt realize how the shape of xyz is changed in the case that all points are co-linear let me know if you see this message.")
    return xyz

def AM_to_xyz(AM:np.ndarray, z:Optional[np.ndarray]=None, AM_rep: Optional[Literal["DM", "CM"]]="CM", approximate: Optional[bool]=False):
    '''
    parameters:
        AM - np.ndarray: (n,n) Adjacency matrix for molecule, can take in either Distance Matrix (DM) or Coulomb Matrix (CM) as Adjaceny matrix
        z  - np.ndarray: (n,) vector of charges for molecule. Order of charges must appear in same order as CM ordering.
        AM_rep Optional[Literal["DM", "CM"]]: literal to select between using DM and CM, by default takes in CM.

    return:
        xyz - np.ndarray: (n,3) coordinates for AM, Note if the Adjacency Matrix contains impossible to represent interactions in 3D (ie not geometrically feasible)
        the generated structure may be fairly unpredictible.
    '''
    if AM_rep == "CM":
        assert z is not None, "charges, z must be passed in if using CM"
        z_ixz_j = np.outer(z, z)
        inv_DM = AM / z_ixz_j
        np.fill_diagonal(inv_DM, val=0)
        DM = get_reciprocal(inv_DM)
    elif AM_rep == "DM":
        DM = AM
    else:
        raise ValueError(f"AM_rep {AM_rep} not supported. AM_rep can take either Literal \'DM\' or \'CM\' ")

    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    DM_sqr = DM * DM

    DM_sqr_row = DM_sqr[0,:][np.newaxis,:]
    DM_sqr_col = DM_sqr[:,0][:,np.newaxis]

    M = (DM_sqr_row + DM_sqr_col - DM_sqr) / 2
    (eig_vals, eig_vecs) = np.linalg.eig(M)

    xyz = eig_vecs * np.sqrt(eig_vals)
    mask = np.isclose(eig_vals,np.zeros_like(eig_vals))==False

    xyz = xyz.T[mask].T 
    if xyz.shape[1] == 2:
        z_coords = np.zeros([xyz.shape[0],1])
        xyz = np.concatenate((xyz, z_coords), axis=1)
    elif xyz.shape[1] == 1:
        yz_coords = np.zeros([xyz.shape[0],2])
        xyz = np.concatenate((xyz, yz_coords), axis=1)
    elif xyz.shape[1] == 0: 
        raise Exception("for some reason xyz has shape[1]==0, might be that all points are on the same location or AM is 1x1 or possibly i didnt realize how the shape of xyz is changed in the case that all points are co-linear let me know if you see this message.")
    return xyz


# def ddtheta_to_xyz(d1, d2, theta):


def get_coulomb_matrix(dist_mat: np.ndarray, z: np.ndarray, supress_prints=True) -> Tuple[np.ndarray, np.ndarray]:
    with BlockPrints(block_prints=supress_prints):
        recip_dist = get_reciprocal(dist_mat)
        
        # get all products of z_i * z_j as a matrix
        z_ixz_j = np.outer(z, z) 

        # Calculate values for diagonal entries
        diag_elems = np.diag(0.5*np.power(z, 2.4))

        # get coulomb matrix
        coulomb_mat = np.multiply(z_ixz_j, recip_dist) + diag_elems

        # sort coulomb matrix
        coulomb_mat, sorted_inds = reorder_coulomb_matrix(coulomb_mat)
        # print(coulomb_mat)
    return coulomb_mat, sorted_inds


############################################################################################################
### Get cluster centers for Distance Matrices:   https://www.science.org/doi/pdf/10.1126/science.1242072 ###
############################################################################################################

def calc_loc_density(DM: np.ndarray, cutoff: float, supress_prints=True) -> np.ndarray:
    with BlockPrints(block_prints=supress_prints):
        cdists = DM - cutoff
        print(cdists)
        cdists[cdists >= 0] = 0
        cdists[cdists < 0] = 1
        local_density = np.sum(cdists, axis=0)
        print(local_density)
        print(local_density.shape)
    return local_density

############################################################################################################

def calc_dist2higher_dens(DM: np.ndarray, local_density: np.ndarray, supress_prints=True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Calculate the minimum distance between an atom and any other 
    atom with higher density for every atom. Value for atom with highest 
    density is given as the distance from the max density atom to the
    atom furthest away from it, as in the maximum distance from the max 
    density atom.
    '''
    with BlockPrints(block_prints=supress_prints):
        d = copy.deepcopy(DM)

        max_i = np.argmax(local_density)
        rel_loc_dens = local_density[:, np.newaxis] - local_density
        low_dens_mask = rel_loc_dens <= 0

        # exclude max density point from mask
        low_dens_mask[max_i, :] = False

        # masking out all points where the local density is less than density of target point 
        d[low_dens_mask] = np.inf 
        print(d)
        delta = np.min(d, axis=0)
        arg_delta = np.argmin(d, axis=0)
        print(delta) 
        print(max_i, delta[max_i])
        delta_at_max_i = np.max(d[max_i])
        delta[max_i] = delta_at_max_i
        arg_delta[max_i] = np.argmax(d[max_i])
        print(max_i, delta[max_i])
    return delta, arg_delta

############################################################################################################

def find_cluster_centers(delta: np.ndarray, dens: np.ndarray, atoms_per_cluster = 70, supress_prints=True) -> np.ndarray:
    '''
    return numpy array of atom indices corresponding to atoms with gamma scores 
    (delta*density) over 1 standard deviation away from the mean
    '''
    with BlockPrints(block_prints=supress_prints):
        cluster_dist = delta*dens
        num_atoms = len(cluster_dist)
        num_centers = num_atoms / atoms_per_cluster

        cluster_center_idx = np.argpartition(-cluster_dist, atoms_per_cluster)[:atoms_per_cluster]
        print(cluster_center_idx)
        # delta_std = np.std(cluster_dist)
        # delta_mean = np.mean(cluster_dist)

        # lim = delta_mean + 1 * delta_std

        # cluster_center_idx = np.argwhere(cluster_dist > lim)
    return cluster_center_idx

############################################################################################################

def pick_center_atom(gamma: np.ndarray, random_choice=False) -> Tuple[int, np.ndarray]:
    '''
    return index of atom with largest delta*density score (gamma) 
    and update gamma array to mask out already picked values
    '''
    if random_choice: atom_idx = random.randrange(len(gamma))
    else: atom_idx = np.argmax(gamma)

    # mask out already selected atom indices # TODO: Unecessary mask since will already be masked later.
                                             #       Remove when functionality confirmed.
    gamma[atom_idx] = -np.inf

    return atom_idx, gamma

# seems like it will be useful for similarity based groupings
def generate_clusters(DM: np.ndarray, cutoff : Optional[int]=5):
    loc_dens = calc_loc_density(DM, cutoff=cutoff, supress_prints=False)
    delta, arg_delta = calc_dist2higher_dens(DM, loc_dens)
    centers = find_cluster_centers(delta, loc_dens)

    clusters_set = np.ones_like(delta)*np.inf
    clusters_set[centers] = centers
    print(clusters_set)
    q = centers.tolist()

    while q:
        elem = q.pop(0)

        new = np.argwhere(arg_delta == elem)
        if elem in new: new.remove(elem)
        new = new[clusters_set[new] == np.inf]
        clusters_set[new] = clusters_set[elem]
        q = q + new.tolist()
        print(q)
    print(clusters_set)
    print(np.unique(clusters_set, return_index=True))
    print(np.unique(clusters_set, return_inverse=True))
    sets = np.unique(clusters_set, return_inverse=True)[1]
    return clusters_set, sets
   
############################################################################################################
### Error Matrix norm                                                                                    ###
############################################################################################################

def error_matrix_norm(mat1: np.ndarray, mat2: np.ndarray, order=2) -> float:
    return np.linalg.norm(mat1 - mat2, ord=order)

############################################################################################################
### subgraph isomorphism occurence finder                                                                ###
############################################################################################################

# (partially ChatGPT)

def find_occurences_G(G: nx.classes.graph.Graph, Q: nx.classes.graph.Graph) -> List[np.ndarray]:
    # Create an IsomorphismMatcher object
    matcher = isomorphism.GraphMatcher(G, Q)

    # Initialize a list to store the occurrences
    occurrences = []
    print()
    # Iterate through the isomorphisms to find occurrences
    for subgraph_nodes in matcher.subgraph_isomorphisms_iter():
        # occurrence = G.subgraph(subgraph_nodes)
        # print(subgraph_nodes.keys)
        occurrences.append(np.fromiter(subgraph_nodes.keys(), dtype=int))

    return occurrences

############################################################################################################
### match 3d substructs                                                                                  ###
############################################################################################################

def find_occurences_3d():
    Chem.rdchem.GetSubstructMatches()

############################################################################################################
### Extract 3D Coordinates From Distance Matrix                                                          ###
############################################################################################################

def DM_to_3d_coordinates(DM):
    # (ChatGPT)
    # Perform multidimensional scaling (MDS)
    if DM.shape == (1,1):
        coordinates = np.array([[0,0,0]])
    else:
        mds = MDS(n_components=3, dissimilarity='precomputed')
        coordinates = mds.fit_transform(DM)
    
    return coordinates

############################################################################################################
### Plot Molecule:                                                                                       ###
############################################################################################################

def plot_vector(start, end):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Extract the x and y coordinates from the start and end points
    x_coords = [start[0], end[0]]
    y_coords = [start[1], end[1]]

    # Plot the vectors
    ax.quiver(*start, *(end - start), angles='xy', scale_units='xy', scale=1, color='b')

    # Set the x and y axis limits
    ax.set_xlim([min(0, start[0], end[0]), max(0, start[0], end[0])])
    ax.set_ylim([min(0, start[1], end[1]), max(0, start[1], end[1])])

    # Add labels to the start and end points
    ax.annotate('start', start, textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=8, color='r')
    ax.annotate('end', end, textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=8, color='r')

    # Set the aspect ratio to equal
    ax.set_aspect('equal')

    # Show the plot
    plt.show()

def plot_molecules_from_xyz(arrays, c=None, a=None):
    # print(c, len(c), 6*[c[0]])
    # Create a new figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if a == None:
        a = [0.7 for i in range(len(arrays))]
    # Plot each array
    for i, array in enumerate(arrays):
        # Extract X, Y, Z coordinates from the array
        x, y, z = array[:, 0], array[:, 1], array[:, 2]

        if c is None:
            # Generate a unique color for each array
            c = plt.cm.viridis(i / len(arrays))
        if len(c) == x.shape:
            # Plot the array
            ax.scatter(x, y, z, c=c, alpha=a[i],s=500, label=f'Array {i+1}')
        elif len(c) == len(arrays):
            # Plot the array
            print(x.shape[0]*[c[i]])
            ax.scatter(x, y, z, c=x.shape[0]*[c[i]])#, alpha=a[i],s=500, label=f'Array {i+1}')
        else:
            ax.scatter(x, y, z, c=c)#, alpha=a[i],s=500, label=f'Array {i+1}')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Plotting Multiple XYZ Arrays')
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_molecules_from_xyz_subplots(arrays, c=None):
    num_arrays = len(arrays)
    num_cols = int(np.ceil(np.sqrt(num_arrays)))
    num_rows = int(np.ceil(num_arrays / num_cols))

    # Create a new figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, array in enumerate(arrays):
        ax = axes[i]
        # print(array)
        # Extract X, Y, Z coordinates from the array
        x, y, z = array[:, 0], array[:, 1], array[:, 2]

        # Plot each array on a separate subplot
        if c is None:
            # Generate a unique color for each array
            c_i = plt.cm.viridis(i / num_arrays)
        else:
            c_i = c[i]


        # Plot the array
        ax.scatter(x, y, z, c=c_i, s=500, label=f'Array {i+1}')

        # Set labels and title for each subplot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Array {i+1}')

        # Set aspect ratio for each subplot
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

        # Add a legend to each subplot
        ax.legend()

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()

def plot_molecule_from_xyz(xyz, colour_mask=None, alpha_mask=0.6, indeces = None):
    if indeces == None: 
        indeces = np.arange(len(xyz[:,0]))
    # (ChatGPT)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('3D Scatter Plots')
    
    overall_min = float('inf')
    overall_max = float('-inf')

    x = xyz[:,0][indeces]
    y = xyz[:,1][indeces]
    z = xyz[:,2][indeces]
        
    # Create a 3D subplot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Plot the scatter plot
    ax1.scatter(x, y, z, c=colour_mask, alpha=alpha_mask, s=500) # fragments[i].z
        
    # Set subplot title and labels
    ax1.set_title('Molecule in 3D space')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Calculate the overall minimum and maximum values
    overall_min = min(np.min(x), np.min(y), np.min(z))
    overall_max = max(np.max(x), np.max(y), np.max(z))
   
    ax1.set_xlim(overall_min, overall_max)
    ax1.set_ylim(overall_min, overall_max)
    ax1.set_zlim(overall_min, overall_max)
    plt.tight_layout()
    
    # Show the figure
    plt.show()

def extend_polyacetylene(segment_coords, segment_symbols, smile_string, n):
    base_str = copy.copy(smile_string)
    for i in range(n):
        smile_string += "=" + base_str
    atomic_symbols = {'C': 6, 'H': 1}
    # Calculate the number of atoms in the segment
    num_atoms_in_segment = len(segment_coords)
    
    # Create an empty array to hold the extended coordinates
    extended_coords = np.zeros((num_atoms_in_segment * n, 3))
    
    # Create a Z vector to hold the atomic numbers
    z_vector = np.zeros(num_atoms_in_segment * n, dtype=int)
    
    # Repeat the segment 'n' times, adjusting the X-coordinate for each repetition
    for i in range(n):
        start_index = i * num_atoms_in_segment
        end_index = (i + 1) * num_atoms_in_segment
        
        extended_coords[start_index:end_index, :] = segment_coords + np.array([i * 7.407, 0.0, 0.0])
        z_vector[start_index:end_index] = [atomic_symbols[symbol] for symbol in segment_symbols]
    z_vector = np.array(z_vector)
    return extended_coords, z_vector, smile_string

# def write_sdf(filename, xyz, atomic_numbers, AM):
#     filename = "./" + filename
#     if filename[-4:] != ".sdf": filename += ".sdf"
#     with open(filename, 'w') as f:
#         # Write the header
#         f.write(f"length {len(atomic_numbers)} PAC\n\n\n")
        
#         # Write the atom count and bond count
#         num_atoms = len(atomic_numbers)
#         AM_mask = np.copy(AM)
#         AM_mask[AM_mask != 0] = 1
#         num_bonds = int(np.sum(AM_mask) / 2)  # Divide by 2 since bonds are symmetric
#         f.write(f"{num_atoms:>3}{num_bonds:>3}\n")
#         # Write the atom count and bond count (no bonds in this example)
#         # f.write(f"{len(atomic_numbers):>3}  0\n")
#         # f.write(f"> <SMILES>\n{smi}")
        
#         # Write the atoms
#         for atom_num, position in zip(atomic_numbers, xyz):
#             symbol = an_to_symbol(atom_num)
#             f.write(f"{position[0]:>10.4f}{position[1]:>10.4f}{position[2]:>10.4f} {symbol:<2} 0\n")
#         #
#         #  Write the bonds with bond orders
#         for i in range(num_atoms):
#             for j in range(i + 1, num_atoms):
#                 bond_order = AM[i, j]
#                 if bond_order > 0:
#                     f.write(f"{i + 1:>3}{j + 1:>3}{bond_order}\n")  # Bond between atoms i and j with bond order
                    
#         # Write footer
#         f.write("M  END\n")
#         f.write("$$$$\n")
#         print(f"{filename} written...")

import numpy as np
from rdkit import Chem

# def smiles_to_bond_matrix(smiles):
#     print("TEST")
#     # Parse the SMILES string to create an RDKit molecule object
#     mol = Chem.MolFromSmiles(smiles)
    
#     if mol is None:
#         raise ValueError("Invalid SMILES string.")
#     print("TEST2")
#     # Get the number of atoms in the molecule
#     num_atoms = mol.GetNumAtoms()
#     print("TEST3")

#     # Initialize a bond adjacency matrix filled with zeros
#     bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
#     print("TEST4")

#     # Iterate over the bonds in the molecule and update the bond matrix
#     for i, bond in enumerate(mol.GetBonds()):
#         print(i)
#         # if i % 10 == 0: print(i)
#         begin_atom_idx = bond.GetBeginAtomIdx()
#         end_atom_idx = bond.GetEndAtomIdx()
#         bond_type = bond.GetBondTypeAsDouble()
        
#         # Assign bond type as a weight in the matrix
#         bond_matrix[begin_atom_idx, end_atom_idx] = bond_type
#         bond_matrix[end_atom_idx, begin_atom_idx] = bond_type

#     return bond_matrix

def smiles_to_bond_matrix(smiles: str):
    print("TEST")
    # Parse the SMILES string to create an RDKit molecule object
    # mol = Chem.MolFromSmiles(smiles)
    
    # if mol is None:
    #     raise ValueError("Invalid SMILES string.")
    print("TEST2")
    # Get the number of atoms in the molecule
    print("TEST3")

    # Initialize a bond adjacency matrix filled with zeros
    num_atoms = smiles.count("C")
    bond_matrix = sp.csr_matrix((num_atoms, num_atoms), dtype=float)

    print(bond_matrix)
    # bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    print("TEST4")
    strlen = len(smiles)

    # Iterate over the bonds in the molecule and update the bond matrix
    for i, char in enumerate(smiles):
        if char.lower != "c":
            continue
        if i != strlen:
            bond_matrix[i, i+1] = 1 if smiles[i+1].lower() == "c" else 2
        if i != 0:
            bond_matrix[i, i-1] = 1 if smiles[i-1].lower() == "c" else 2

        # print(i)

    return bond_matrix

def generate_sdf(xyz_coordinates, bond_matrix, atomic_numbers, output_filename):
    num_atoms = len(xyz_coordinates)

    # Create an RDKit molecule object
    mol = Chem.RWMol()

    # Add atoms with atomic numbers and coordinates
    for atom_num, coords in zip(atomic_numbers, xyz_coordinates):
        print(atom_num, coords)
        atom = Chem.Atom(int(atom_num))
        atom.SetDoubleProp("x", coords[0])
        atom.SetDoubleProp("y", coords[1])
        atom.SetDoubleProp("z", coords[2])
        mol.AddAtom(atom)

    # Add bonds from the bond matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bond_value = bond_matrix[i, j]
            if bond_value == 1:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
            elif bond_value == 2:
                mol.AddBond(i, j, Chem.BondType.DOUBLE)
            else:
                mol.AddBond(i, j, Chem.BondType.SINGLE)


    # Convert the RDKit molecule to a standard molecule
    mol = mol.GetMol()

    # Generate the SDF file
    w = Chem.SDWriter(output_filename)
    w.write(mol)
    w.close()

# # Example usage:
# xyz_coordinates = np.array([[0.0, 0.0, 0.0],
#                             [1.0, 0.0, 0.0],
#                             [0.0, 1.0, 0.0]])

# bond_matrix = np.array([[0, 1, 1],
#                         [1, 0, 0],
#                         [1, 0, 0]])

# atomic_numbers = np.array([6, 6, 6])  # Atomic numbers for carbon atoms

# output_filename = 'output.sdf'
# generate_sdf(xyz_coordinates, bond_matrix, atomic_numbers, output_filename)

def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # You can adjust the seed for different conformations
    mol = Chem.RemoveHs(mol)
    
    # Create the polymer by repeating the monomer
    n = 10  # Change this value to determine the polymer length
    polymer = Chem.MolFromSmiles("".join([smiles] * n))
    
    # Embed the polymer
    polymer = Chem.AddHs(polymer)
    AllChem.EmbedMolecule(polymer, randomSeed=42)
    polymer = Chem.RemoveHs(polymer)
    
    # Get the coordinates and atomic numbers
    coords = polymer.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in polymer.GetAtoms()]
    
    return coords, atomic_numbers


# if __name__ == "__main__":
#     # # Acetylene

#     # import os
#     print("Current working directory:", os.getcwd())
#     # xyz = np.array([[0.000,  0.000, 0.000],  # C
#     #                 [0.930,  0.930, 0.000],  # H
#     #                 [1.340,  0.000, 0.000],  # C
#     #                 [0.410, -0.930, 0.000],  # H
#     #                 [2.680,  0.000, 0.000],  # C
#     #                 [3.610,  0.930, 0.000]   # H
#     #                 ])
#     # xyz = np.array([
#     #     [ 3.158, -0.175, -0.001],
#     #     [ 1.789,  0.414,   -0.0],
#     #     [  0.66, -0.292,  0.001],
#     #     [ -0.66,  0.294,  0.001],
#     #     [-1.788, -0.414,   -0.0],
#     #     [-3.158,  0.172, -0.001],
#     #     [ 3.116,  -1.261,   0.0],
#     #     [ 3.713,  0.157,  0.877],
#     #     [ 3.711,  0.155, -0.881],
#     #     [ 1.745,  1.496, -0.001],
#     #     [ 0.703, -1.374,  0.002],
#     #     [-0.704,  1.376,  0.002],
#     #     [-1.743, -1.496, -0.002],
#     #     [-3.119,  1.259, -0.008],
#     #     [-3.714, -0.165, -0.876],
#     #     [-3.709, -0.154,  0.881]])

#     xyz = np.array([
#         [ 3.158, -0.175, 0],
#         [ 1.789,  0.414, 0],
#         [  0.66, -0.292, 0],
#         [ -0.66,  0.294, 0],
#         [-1.788, -0.414, 0],
#         [-3.158,  0.172, 0]])
#         # NOTE: Hydrogens
#         # [ 3.116, -1.261, 0],
#         # [ 1.745,  1.496, 0],
#         # [ 0.703, -1.374, 0],
#         # [-0.704,  1.376, 0],
#         # [-1.743, -1.496, 0],
#         # [-3.119,  1.259, 0]])
#         # NOTE: END HYDROGENS
#         # [ 3.713,  0.157,  0.877],
#         # [ 3.711,  0.155, -0.881],
#         # NOTE: START HYDROGENS
#         # [-3.714, -0.165, -0.876],
#         # [-3.709, -0.154,  0.881]])
#     dxyz = np.diff(xyz, axis=0)
#     print(dxyz)
#     s_sym = ['C','C','C','C','C','C']#,"H","H","H","H","H","H"]
#     # s_sym = ['C','C','C','C','C','C',"H","H","H","H","H","H"]

#     # # Input the desired number of carbon atoms (n)
#     # n = 138  # You can change this value to your desired length

#     # # Extend the polyacetylene segment
#     smi = "CC=CC=CC"
#     print("ext Entered")
#     ext_xyz, ext_z, ext_smi = extend_polyacetylene(xyz, s_sym, smi, 16100)
#     print("AM gen step")
#     AM = smiles_to_bond_matrix(ext_smi)
#     dext_xyz = np.diff(ext_xyz, axis=0)
#     # print(dext_xyz)
#     # print(len(ext_xyz))
#     print("gen sdf step")
#     generate_sdf(ext_xyz, AM, ext_z, f"PAC_len_{len(ext_xyz)}.sdf")
#     print("read sdf step")
#     read_sdf_file(f"PAC_len_{len(ext_xyz)}.sdf")
#     # xyz, z = smiles_to_xyz("C=CC=C")
#     # xyz, z, ans = read_sdf_file(filename="./PACTEST.sdf")

#     # xyz, z, ans = read_smiles("C=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC")
#     # print(xyz)
#     # plot_molecule_from_xyz(ext_xyz, colour_mask=ext_z)
# # # Print the extended segment
# # for i, atom in enumerate(ext_xyz):
# #     print(f"{i+1}: C {atom[0]:.3f} {atom[1]:.3f} {atom[2]:.3f}")


def plot_molecule_from_file(filename, colour_mask=None, alpha_mask=0.6):
    xyz, z, _ = read_file(filename)
    if colour_mask==None: colour_mask=z
    plot_molecule_from_xyz(xyz, colour_mask=colour_mask, alpha_mask=alpha_mask)

def plot_matrices(matrices, legend):
    n = len(matrices)
    fig, axes = plt.subplots(1, n+1, figsize=(n * 4 + 2, 4))  # Create subplots
    print(type(fig), type(axes))
    for i in range(n):
        ax = axes[i] if n > 1 else axes  # Handle single subplot case
        print(type(ax))
        im = ax.imshow(matrices[i], cmap='viridis')  # Plot matrix
        
        # Write numerical values in the cells
        for j in range(matrices[i].shape[0]):
            for k in range(matrices[i].shape[1]):
                text = ax.text(k, j, f'{matrices[i][j, k]:.2f}', ha='center', va='center')

        ax.set_title(legend[i%n])  # Set subplot title

    # Create a separate axis for the colorbar
    cax = axes[-1]
    fig.colorbar(im, cax=cax)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()  # Display the plot

# # Example usage:
# # Create some sample matrices
# matrix1 = np.random.rand(5, 5)
# matrix2 = np.random.rand(6, 6)
# matrix3 = np.random.rand(7, 7)

# # Plot the matrices
# matrices = [matrix1, matrix2, matrix3]
# plot_matrices(matrices)
############################################################################################################
### Visualize Molecule with ASE                                                                          ###
############################################################################################################
def atoms_to_html(atoms):
    'Return the html representation the atoms object as string'

    from tempfile import NamedTemporaryFile
    
    with NamedTemporaryFile('r+', suffix='.html') as ntf:
        atoms.write(ntf.name, format='html')
        ntf.seek(0)
        html = ntf.read()
        # print(html)
    return html
def visualize_mol(coords: np.ndarray, qs: np.ndarray, windowed: Optional[bool]=True, style: Optional[Literal["ball", "stick", "ball_and_stick", "cartoon", "spacefill"]]="ball_and_stick"):
    '''Visualize molecule given coords and atomic numbers, in windowed mode style cant be chosen, for inline viewing viewer object is returned and a choice of style is permitted'''
    styles = ["ball", "stick", "ball_and_stick", "cartoon", "spacefill"]
    mol = Atoms(numbers=qs, positions=coords)
    if windowed:
        view(mol)
    else:
        # Convert ASE Atoms object to nglview structure
        viewer = nv.show_ase(mol)

        # Set visualization style: 'ball+stick', 'cartoon', 'stick', etc.
        if   style == styles[0]: pass
        elif style == styles[1]: viewer.add_stick()
        elif style == styles[2]: viewer.add_ball_and_stick()
        elif style == styles[3]: viewer.add_cartoon()
        elif style == styles[4]: viewer.add_spacefill()

        # Display the visualization

        return viewer

############################################################################################################
### Match point clouds: (debugging and fragment comparison)                                    (ChatGPT) ###
############################################################################################################

import numpy as np
from scipy.spatial import KDTree

def cw_icp_algorithm(source_points, target_points, source_classes, target_classes, max_iterations=10000):
    transformed_source = np.copy(source_points)
    transformation = np.identity(4)

    for iteration in range(max_iterations):
        # Find the nearest neighbors between the target and transformed source points
        tree = KDTree(target_points)
        distances, indices = tree.query(transformed_source)

        # Compute the transformation matrix using the selected corresponding points
        correspondences = transformed_source[indices]
        weights = (source_classes == target_classes[indices]).astype(float)
        if np.sum(weights) != 0:  # Check if the sum of weights is not zero
            weights /= np.sum(weights)  # Normalize weights
        mean_source = np.average(correspondences, axis=0, weights=weights)
        mean_target = np.mean(target_points, axis=0)
        centered_source = correspondences - mean_source
        centered_target = target_points - mean_target
        covariance_matrix = np.dot((centered_source * weights[:, None]).T, centered_target)
        u, _, vh = np.linalg.svd(covariance_matrix)
        rotation = np.dot(u, vh)
        translation = mean_target - np.dot(rotation, mean_source)

        # Apply the transformation to the source points
        transformed_source = np.dot(transformed_source, rotation.T) + translation

        # Update the transformation matrix correctly
        transformation = np.dot(transformation, np.vstack([np.column_stack([rotation, translation]), [0, 0, 0, 1]]))

    return transformed_source, transformation


def icp_BAD(source_points, target_points):
    pc_fix = PointCloud(source_points, columns=["x", "y", "z"])
    pc_mov = PointCloud(target_points, columns=["x", "y", "z"])

    # Create simpleICP object, add point clouds, and run algorithm
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
    print(distance_residuals)
    return X_mov_transformed, H

import numpy as np
from scipy.spatial import KDTree

def icp(source_points, target_points, max_iterations=200, tolerance=1e1):
    """
    Iterative Closest Point (ICP) algorithm implementation.
    
    Arguments:
    - source: numpy array of shape (N, 3) representing the source point cloud.
    - target: numpy array of shape (M, 3) representing the target point cloud.
    - max_iterations: maximum number of iterations to perform.
    - tolerance: convergence criteria threshold.
    
    Returns:
    - R: 3x3 rotation matrix.
    - t: 3x1 translation vector.
    - transformed_source: numpy array of shape (N, 3) representing the transformed source point cloud.
    """
    
    assert source_points.shape[1] == target_points.shape[1] == 3, "Point clouds must have 3 columns (x, y, z)"
    
    # Initialize transformation
    R = np.eye(3)
    t = np.zeros((3, 1))
    transformed_source = source_points.copy()
    
    for iteration in range(max_iterations):
        print(iteration,end="\r")
        # Find nearest neighbors between the target and transformed source point clouds
        tree = KDTree(target_points)
        distances, indices = tree.query(transformed_source)
        
        # Select the closest points
        closest_points_target = target_points[indices]
        closest_points_source = transformed_source
        
        # Compute the centroid of the closest points
        centroid_target = np.mean(closest_points_target, axis=0)
        centroid_source = np.mean(closest_points_source, axis=0)
        
        # Compute the covariance matrix
        H = np.dot((closest_points_source - centroid_source).T, closest_points_target - centroid_target)
        
        # Perform singular value decomposition
        U, _, Vt = np.linalg.svd(H)
        Rn = np.dot(Vt.T, U.T)
        
        # Update the rotation and translation
        t = centroid_target.T - np.dot(Rn, centroid_source.T)
        R = np.dot(Rn, R)
        
        # Transform the source point cloud
        transformed_source = np.dot(R, transformed_source.T).T + t.T
        
        # Check for convergence
        # if np.sum(np.abs(Rn - np.eye(3))) < tolerance:
        #     break
    
    return transformed_source, (R, t) 

# def teaserpp_rotation(source_points, destination_points):    
#     # NOTE:  ORIGINAL PARAMETERS
#     # Populating the parameters
#     solver_params = teaserpp_python.RobustRegistrationSolver.Params()
#     print(dir(solver_params))
#     # raise Exception
#     solver_params.cbar2 = 1
#     solver_params.noise_bound = 0.01
#     solver_params.estimate_scaling = True
#     solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
#     solver_params.rotation_gnc_factor = 1.4
#     solver_params.rotation_max_iterations = 100
#     solver_params.rotation_cost_threshold = 1e-12
#     # solver_params./

#     solver = teaserpp_python.RobustRegistrationSolver(solver_params)
#     # print(dir(solver))
#     # raise Exception

#     # TODO: might need to switch rep_xyz and fragment_xyz order
#     solver.solve(source_points.T, destination_points.T)
    
#     solution = solver.getSolution()
#     R = solution.rotation
#     t = solution.translation

#     print(f"TRANSLATION: {t}")
#     aligned_source = (R @ source_points.T + t[:, np.newaxis]).T
#     error = np.sum(np.linalg.norm(aligned_source - destination_points, axis=1))
    
#     return aligned_source, (R, t)

def teaserpp_rotation(source_points, destination_points, allow_reflection=False):    
    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.01
    # solver_params.estimate_scaling = True
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    # solver_params.

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Align using the original source points
    solver.solve(source_points.T, destination_points.T)
    solution = solver.getSolution()
    R = solution.rotation
    t = solution.translation

    aligned_source = (R @ source_points.T + t[:, np.newaxis]).T
    error = np.sum(np.linalg.norm(aligned_source - destination_points, axis=1))
    if not allow_reflection:
        return aligned_source, (R, t)

    # Reflect the source points
    reflected_source_points = source_points * [-1, -1, -1]

    # Align using the reflected source points
    solver.solve(reflected_source_points.T, destination_points.T)
    solution_reflected = solver.getSolution()
    R_reflected = solution_reflected.rotation
    t_reflected = solution_reflected.translation

    aligned_reflected_source = (R_reflected @ reflected_source_points.T + t_reflected[:, np.newaxis]).T
    error_reflected = np.sum(np.linalg.norm(aligned_reflected_source - destination_points, axis=1))

    # Choose the alignment with the lowest error
    if error < error_reflected:
        return aligned_source, (R, t), error, 1
    else:
        return aligned_reflected_source, (R_reflected, t_reflected), error_reflected, -1


if __name__ == "__main__":

    frag_xyz = np.array([[-4.963,  9.532, 13.383],
                         [-6.099, 11.245, 14.211],
                         [-5.569, 10.578, 13.187],
                         [-4.507, 11.545, 11.225],
                         [ -5.78, 11.112, 11.791],
                         [-5.943, 10.771, 15.571]])

    target = np.array([[ 7.231, -7.416, 4.644],
                       [ 6.956, -7.981, 2.516],
                       [ 7.152,  -8.27, 3.803],
                       [ 8.498, -9.902,  4.96],
                       [ 7.265, -9.713, 4.195],
                       [ 6.843, -6.588, 2.075]])
    aligned_xyz, (R, t), error, reflection = teaserpp_rotation(frag_xyz, target, allow_reflection=True)
    print(error, reflection)
    print(target, aligned_xyz, target-aligned_xyz)
    # aligned_xyz, (R, t) = teaserpp_rotation(frag_xyz, target)#, allow_reflection=True)
    plot_molecules_from_xyz_subplots([target, aligned_xyz])# c=[1, 10])

def icp_with_outlier_rejection(source_points, target_points, max_iterations=100, tolerance=1e-6, outlier_rejection=True):
    """
    Perform ICP algorithm with outlier rejection using robust kernels.
    
    Args:
        source_points (ndarray): Source point cloud with shape (N, 3).
        target_points (ndarray): Target point cloud with shape (M, 3).
        max_iterations (int): Maximum number of iterations (default: 100).
        tolerance (float): Convergence tolerance (default: 1e-6).
        outlier_rejection (bool): Flag to enable outlier rejection (default: True).
    
    Returns:
        R (ndarray): Rotation matrix with shape (3, 3).
        t (ndarray): Translation vector with shape (3,).
    """
    assert source_points.shape[1] == 3 and target_points.shape[1] == 3, "Point clouds must have 3 columns."
    
    # Initial transformation
    R = np.eye(3)
    t = np.zeros(3)
    converged = False
    
    for iteration in range(max_iterations):
        # Apply current transformation to source points
        transformed_points = R.dot(source_points.T).T + t
        
        # Find nearest neighbors in target points
        distances, indices = find_nearest_neighbors(transformed_points, target_points)
        
        # Reject outliers using robust kernel
        if outlier_rejection:
            weights = compute_robust_weights(distances)
        else:
            weights = np.ones(len(distances))
        
        # Estimate the transformation using weighted least squares
        R_new, t_new = estimate_transformation(source_points, target_points[indices], weights)
        
        # Update transformation
        R = R_new.dot(R)
        t = R_new.dot(t) + t_new
        
        # Check convergence
        if np.allclose(R_new, np.eye(3)) and np.allclose(t_new, np.zeros(3)):
            converged = True
            break
        
        # Check for small changes in transformation parameters
        if np.linalg.norm(R_new - np.eye(3)) + np.linalg.norm(t_new) < tolerance:
            converged = True
            break
    
    if not converged:
        print("ICP did not converge within the specified number of iterations.")
    
    return R, t

def find_nearest_neighbors(points, target_points):
    """
    Find nearest neighbors in target points for each point in the given points.
    
    Args:
        points (ndarray): Points with shape (N, 3).
        target_points (ndarray): Target points with shape (M, 3).
    
    Returns:
        distances (ndarray): Distances between points and their nearest neighbors with shape (N,).
        indices (ndarray): Indices of nearest neighbors in target points with shape (N,).
    """
    distances = np.sqrt(np.sum((points[:, None, :] - target_points[None, :, :])**2, axis=2))
    indices = np.argmin(distances, axis=1)
    return distances, indices

def compute_robust_weights(distances, c=1.4826):
    """
    Compute robust weights using the Tukey's bisquare kernel.
    
    Args:
        distances (ndarray): Distances with shape (N,).
        c (float): Scale factor (default: 1.4826).
    
    Returns:
        weights (ndarray): Robust weights with shape (N,).
    """
    residuals = distances - np.median(distances)
    weights = np.where(np.abs(residuals) < c, (1 - (residuals / c)**2)**2, 0)
    return weights

def estimate_transformation(source_points: np.ndarray, target_points: np.ndarray):
    """
    Estimate the transformation using weighted least squares.
    
    Args:
        source_points (ndarray): Source points with shape (N, 3).
        target_points (ndarray): Target points with shape (N, 3).
    
    Returns:
        aligned_source (np.ndarray): source points aligned to target_points with shape (N, 3)
        R (ndarray): Rotation matrix with shape (3, 3).
        t (ndarray): Translation vector with shape (3,).
    """
   
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    
    H = source_centered.T.dot(target_centered)
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T.dot(U.T)
    t = centroid_target - R.dot(centroid_source)
    aligned_source = (R @ source_points.T + t[:, np.newaxis]).T
    return aligned_source, (R, t)


def get_vector_rotation(src: np.ndarray, tgt: np.ndarray):
    """
    Estimate the transformation using weighted least squares.
    
    Args:
        source_points (ndarray): Source points with shape (N, 3).
        target_points (ndarray): Target points with shape (N, 3).
        weights (ndarray): Robust weights with shape (N,).
    
    Returns:
        R (ndarray): Rotation matrix with shape (3, 3).
        t (ndarray): Translation vector with shape (3,).
    """
        
    H = src.T @ tgt
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    return R


import numpy as np
from scipy.spatial import KDTree


def generalized_icp(source_points, target_points, max_iterations=100, tolerance=1e-6):
    """
    Generalized Iterative Closest Point (ICP) algorithm.

    Args:
        source_points (ndarray): Array of source point coordinates (N x 3).
        target_points (ndarray): Array of target point coordinates (N x 3).
        max_iterations (int): Maximum number of iterations (default: 100).
        tolerance (float): Convergence tolerance (default: 1e-6).

    Returns:
        ndarray: Rigid transformation matrix (4 x 4) that aligns the source points to the target points.
    """
    assert source_points.shape == target_points.shape, "Source and target point arrays must have the same shape."
    assert source_points.shape[1] == 3, "Source and target point arrays must be Nx3 arrays."

    num_points = source_points.shape[0]

    # Initialize transformation matrix
    transformation = np.eye(4)

    for iteration in range(max_iterations):
        # Apply transformation to source points
        transformed_source = (transformation @ np.hstack((source_points, np.ones((num_points, 1)))).T).T[:, :3]

        # Find nearest neighbors in target points for each transformed source point
        tree = KDTree(target_points)
        _, nearest_indices = tree.query(transformed_source)

        # Compute centroid of nearest target points
        nearest_centroid = np.mean(target_points[nearest_indices], axis=0)

        # Compute centroid of transformed source points
        source_centroid = np.mean(transformed_source, axis=0)

        # Compute cross-covariance matrix
        cross_covariance = np.dot((transformed_source - source_centroid).T, (target_points[nearest_indices] - nearest_centroid))

        # Perform singular value decomposition
        u, _, vh = np.linalg.svd(cross_covariance)

        # Compute rotation matrix
        rotation = np.dot(vh.T, u.T)

        # Compute translation vector
        translation = nearest_centroid - np.dot(rotation, source_centroid)

        # Update transformation matrix
        new_transformation = np.eye(4)
        new_transformation[:3, :3] = rotation
        new_transformation[:3, 3] = translation

        # Check convergence
        delta = np.max(np.abs(new_transformation - transformation))
        transformation = new_transformation

        if delta < tolerance:
            break
        
    # transformed_points = _get_transformed_points(source_points, transformation)
    return transformed_source, transformation

def _get_transformed_points(starting_points, transformation_matrix):
    """
    Get the transformed points using the starting points and the transformation matrix.

    Args:
        starting_points (ndarray): Array of starting point coordinates (N x 3).
        transformation_matrix (ndarray): Rigid transformation matrix (4 x 4).

    Returns:
        ndarray: Transformed points.
    """
    assert starting_points.shape[1] == 3, "Starting point array must be a Nx3 array."

    num_points = starting_points.shape[0]

    # Append ones to the starting points
    starting_points_homogeneous = np.hstack((starting_points, np.ones((num_points, 1))))

    # Apply transformation
    transformed_points_homogeneous = (transformation_matrix @ starting_points_homogeneous.T).T

    # Remove the homogeneous coordinate and return the transformed points
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points

############################################################################################################
### if __name__ == "__main__":                                                                           ###
############################################################################################################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors(vectors):
    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through each vector
    for vector in vectors:
        start = np.array([0,0,0])
        end = vector

        # Plot the vectors
        ax.quiver(*start, *(end - start), color='b')

        # Add labels to the start and end points
        ax.text(*start, 'start', color='r')
        ax.text(*end, 'end', color='r')

    # Set the aspect ratio to equal
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    # Show the plot
    plt.show()

# if __name__=="__main__":
#     # Example arrays
#     array1 = np.random.rand(7, 3)

#     array2 = np.random.rand(7, 3)

#     print(array1)
#     # Calculate dot products of corresponding rows
#     import time
#     st = time.time()
#     dot_products = np.diag(np.dot(array1, array2.T))
#     dt1 = time.time() - st
#     st = time.time()
#     dot_products = np.sum(array1 * array2, axis=1)
#     dt2 = time.time() - st
#     print(dt1, dt2)

#     # Example arrays
#     array1 = np.array([[1, 2, 3],
#                        [4, 5, 6],
#                        [7, 8, 9]])

#     array2 = np.array([[0, 3, 2],
#                        [4, 1, 6],
#                        [5, 5, 2]])

#     # Compute cross products of corresponding rows
#     cross_products = np.cross(array1, array2)

#     # Compute magnitudes of cross products
#     magnitudes = np.linalg.norm(cross_products, axis=1)
#     magnitudes[magnitudes == 0] = 1 # to remove div. by zero problems ('unit' vector will be 0)

#     # Compute unit vectors in the direction of cross products
#     axis = cross_products / magnitudes[:, np.newaxis] 
    
#     # Example XYZ coordinates
#     coordinates = np.array([[1, 2, 3],
#                             [4, 5, 6],
#                             [7, 8, 9]])

#     # Calculate the centroid
#     centroid = np.mean(coordinates, axis=0)
#     print(centroid)

#     # Subtract centroid from coordinates
#     centered_coordinates = coordinates - centroid

#     print(centered_coordinates)


#     import numpy as np

#     # Example arrays
#     array1 = np.array([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])

#     array2 = np.array([[10, 20, 30],
#                     [40, 50, 60],
#                     [70, 80, 90]])

#     # Calculate dot products of corresponding rows
#     dot_products = np.sum(array1 * array2, axis=1)

#     # Calculate magnitudes of each row vector
#     magnitudes1 = np.linalg.norm(array1, axis=1)
#     magnitudes2 = np.linalg.norm(array2, axis=1)
#     print(magnitudes1, magnitudes2)
#     # Calculate angles in radians between corresponding rows
#     angles = np.arccos(dot_products / (magnitudes1 * magnitudes2))

#     # Convert angles to degrees
#     angles_deg = np.degrees(angles)

#     print(angles_deg)
#     # Example usage
#     vectors_array = np.array([[0, 1, -1],
#                             [0, 2, -2],
#                             [0, 3, -3],
#                             [1, 4, -4],
#                             [1, 5, -5],
#                             [1, 6, -6]])

#     plot_vectors(vectors_array)

############################################################################################################

def plot_heatmap(heatmap, title=""):
    aspect_ratio = heatmap.shape[0] / heatmap.shape[1]
    plt.figure(figsize=(12 , 9))  # Adjust the figure size to stretch the smaller dimension
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("molecules")
    plt.ylabel("energies")
    plt.show()

############################################################################################################
### TSP Smoothness based sorting code                                                                    ###
############################################################################################################

def compute_cost_matrix(M, metric='cityblock'):
    cost_matrix = squareform(pdist(M.T, metric='cityblock'))
    return cost_matrix

def solve_tsp(cost_matrix):
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(cost_matrix), 1, 0)

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cost_matrix[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # Get the optimal tour
        index = routing.Start(0)
        optimal_tour = []
        while not routing.IsEnd(index):
            optimal_tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return optimal_tour
    else:
        return None

def tsp_sort(M, metric="cityblock") -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        m (np.ndarray[n,m]): n = length of angle-energy function vector, m = number of compounds.

    Returns:
        ordered m
        order (np.ndarray[m]): ordered indeces that minimize cityblock distance between the compound vectors to maximize smoothness.
    """
    
    cost_matrix = compute_cost_matrix(M, metric)
    best_tour = solve_tsp(cost_matrix)
    M_ordered = M[:, best_tour]
    return M_ordered, best_tour

############################################################################################################
### heatmap smoothness score                                                                             ###
############################################################################################################

def fitness(heatmap):
    similarity = np.sum(np.abs(np.diff(heatmap, axis=1)))
    return -similarity

# +=========================================================================================================+
# |                                               END OF FILE                                               |
# +=========================================================================================================+


