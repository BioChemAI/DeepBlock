from dataclasses import dataclass, field, fields
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union
import warnings

import numpy as np
from numpy import linalg as LA

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolTransforms import ComputeCentroid

from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB import PDBParser, PDBIO, Select

from ..exceptions import EmptyChainException, IllegalSDFException

from . import pretty_dataclass_repr, rdkit_log_handle, pretty_kv

_TResidueId = Tuple[str, int, str]

class residue_constants:
    restype_1to3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    restype_3to1 = {v: k for k, v in restype_1to3.items()}

def chain_to_res3d(chain: Chain) -> Tuple[str, np.ndarray, List[int]]:
    """Convert chain into sequence and atomic point cloud. 
    Refer to the method used by esm.

    https://github.com/aqlaboratory/openfold/blob/main/openfold/np/protein.py
    """
    seq: str = ''
    pos: List[np.ndarray] = []
    ids: List[int] = []

    for res in chain:
        res: Residue
        id: _TResidueId = res.get_id()
        if id[2] != " ":
            raise ValueError(
                f"PDB contains an insertion code at chain {chain.id} and residue "
                f"index {res.id[1]}. These are not supported."
            )
        ids.append(id[1])

        # Refer to this link. The alphabet of esm supports 'X' for unknow residue.
        # https://github.com/facebookresearch/esm/blob/main/esm/constants.py
        resname = res.get_resname()
        res_shortname = residue_constants.restype_3to1.get(resname, "X")
        seq += res_shortname
        pos.append(res.center_of_mass())

    pos = np.array(pos)
    assert len(seq) == len(pos)
    return seq, pos, ids

@dataclass
class RemovedResidues:
    het: List[Residue] = field(default_factory=list)
    mut: List[Residue] = field(default_factory=list)
    wat: List[Residue] = field(default_factory=list)
    unk: List[Residue] = field(default_factory=list)

    def get_all(self):
        return self.het + self.mut + self.wat + self.unk
    
    def __repr__(self) -> str:
        ids = [res.get_id()[0] for res in self.get_all()]
        return pretty_kv(dict(Counter(ids)))

def clean_chain(chain: Chain) -> Tuple[Chain, RemovedResidues]:
    """Remove hetero-residue mutant, and water molecule from chain. 
    Return the cleaned chain and extracted items.
    
    Refer to https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ#what-is-a-residue-id
    """
    _chain = chain.copy()
    removed_residues = RemovedResidues()
    for res in chain:
        res: Residue
        id: _TResidueId = res.get_id()
        if id[0] != " ":
            if id[0].startswith("H_"):
                removed_residues.het.append(res)
            elif id[0] == "W":
                removed_residues.wat.append(res)
            else:
                removed_residues.unk.append(res)
            _chain.detach_child(id)
        elif id[2] != " ":
            removed_residues.mut.append(res)
            _chain.detach_child(id)

    return _chain, removed_residues

class ResidueSelect(Select):
    def __init__(self, 
                residues: Sequence[Union[Residue, _TResidueId]], included=True):
        self.residue_id_lst = []
        for res in residues:
            if isinstance(res, Residue):
                self.residue_id_lst.append(res.id)
            else:
                self.residue_id_lst.append(res)
        self.included = included

    def __repr__(self):
        return f"<Select {len(self.residue_id_lst)} residue(s)>"

    def accept_residue(self, residue):
        return 0 if (residue.id in self.residue_id_lst) ^ (self.included) else 1

def residue_to_pdb(residue: Residue, structure: Structure, included: bool) -> str:
    io = PDBIO()
    io.set_structure(structure)
    residue_select = ResidueSelect(residues=[residue], included=included)

    with StringIO() as f:
        io.save(f, residue_select)
        pdb_str = f.getvalue()
    return pdb_str

def sdf_to_smi(sdf_fn: Path) -> str:
    with rdkit_log_handle() as rdLog:
        mol = next(Chem.SDMolSupplier(str(sdf_fn)))
        if not mol:
            raise IllegalSDFException(rdLog())
        return Chem.MolToSmiles(mol)

@dataclass
class ComplexAAExtract:
    id: str
    seq: str
    len: int = None
    ids: List[int] = None
    pos: np.ndarray = None
    pocket_cent: np.ndarray = None
    pocket_dist: np.ndarray = None

    def __repr__(self) -> str:
        return pretty_dataclass_repr(self)

    def __len__(self):
        return self.len

    def __post_init__(self):
        self.len = len(self.seq)

def get_centroid(mol: Mol) -> np.ndarray:
    """Calculate the centroid of molecule.

    Args:
        mol (Mol): Molecule.

    Returns:
        np.ndarray: Centroid.
    """
    conformer = mol.GetConformer()
    centroid = ComputeCentroid(conformer)
    return np.array(centroid)

def extract_chain(chain: Chain, mol: Mol=None) -> ComplexAAExtract:
    """Input: chain
    Output: ComplexAAExtract(id, len, seq, ids, pos)

    Input: chain
    Output: ComplexAAExtract(id, len, seq, ids, pos, pocket_cent, pocket_dist)
    """
    chain, removed_residues = clean_chain(chain)
    if len(chain) == 0:
        raise EmptyChainException(f"Empty {chain} "
                                  f"after -{len(removed_residues.get_all())} clean! -> {repr(removed_residues)}")
    seq, pos, ids = chain_to_res3d(chain)
    aa = ComplexAAExtract(id=chain.id, seq=seq, ids=ids, pos=pos)
    if mol:
        aa.pocket_cent = get_centroid(mol)
        aa.pocket_dist = LA.norm(aa.pocket_cent - aa.pos, axis=-1)
    return aa

def extract_seq(seq: str, id: str='X'):
    """Input: seq
    Output: ComplexAAExtract(id, len, seq)
    """
    return ComplexAAExtract(id=id, seq=seq)

CHAIN_SPECIAL_IDS = {r"{chain_argmax_length}", r"{chain_close_pocket}", r"{chain_first}"}
def extract(protein_pdb_fn: Path, 
            ligand_sdf_fn: Path=None, chain_id: str=None,
            close_threshold: int=8,
            logging_handle: Callable=lambda msg: None) -> ComplexAAExtract:
    """Extract rich chain information from protein PDB file.

    Args:
        protein_pdb_fn (Path): Protein PDB file path.
        ligand_sdf_fn (Path, optional): Ligand SDF file path. Automatically 
        select chain according to known ligands. Defaults to None. 
        chain_id (str, optional): Chain ID in PDB file like 'A', 'B', etc. 
        Set chain manually. Defaults to None.
        close_threshold (int, optional): Threshold value for judging the 
        number of adjacent residues. Defaults to None.

    Returns:
        ComplexAAExtract: Result.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_pdb_fn.stem, protein_pdb_fn)
    assert len(structure) == 1, "PDB file contains multiple models or no model!"
    model = structure[0]
    chains = list(model)

    if ligand_sdf_fn:
        with rdkit_log_handle() as rdLog:
            mol = next(Chem.SDMolSupplier(str(ligand_sdf_fn)))
            if not mol:
                raise IllegalSDFException(rdLog())
    else:
        mol = None

    aa_lst = []
    for chain in chains:
        try:
            aa = extract_chain(chain, mol)
        except EmptyChainException as err:
            warnings.warn(str(err))
        else:
            aa_lst.append(aa)

    if len(aa_lst) == 0:
        raise Exception("No chain available!")

    id_lst = [aa.id for aa in aa_lst]
    len_lst = [aa.len for aa in aa_lst]

    if chain_id is None:
        if aa.pocket_dist is None:
            chain_id = r"{chain_argmax_length}"
        else:
            chain_id = r"{chain_close_pocket}"
        logging_handle(f"Chain ID not provided, automatically select {repr(chain_id)}")

    if chain_id in CHAIN_SPECIAL_IDS:
        if chain_id == r"{chain_argmax_length}":
            aa_idx = np.argmax(len_lst)
        elif chain_id == r"{chain_close_pocket}":
            assert aa.pocket_dist is not None, "Pocket distance not available!"
            close_dist_arr = np.array(
                [np.sort(aa.pocket_dist)[:close_threshold].mean() for aa in aa_lst])
            aa_idx = close_dist_arr.argmin()
        elif chain_id == r"{chain_first}":
            aa_idx = 0
        else:
            raise NotImplementedError(f"Special Chain ID {chain_id} not implemented!")
        aa = aa_lst[aa_idx]
        logging_handle(f"Special Chain ID resolved to {aa.id=}, {aa.len=}, {aa_idx=}")
    else:
        if chain_id not in id_lst:
            raise ValueError(f"Chain ID {chain_id} not found in {id_lst}, "
                             f"also not in special IDs: {list(CHAIN_SPECIAL_IDS)}")
        else:
            aa = aa_lst[id_lst.index(chain_id)]
    return aa
