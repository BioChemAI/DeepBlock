import copy
from collections import defaultdict
from typing import Any, List
from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem.Descriptors import ExactMolWt

from . import OrderedDFS

bond_type_name_to_char_dict = {
    'SINGLE': '-',
    'DOUBLE': '=',
    'TRIPLE': '#',
    'AROMATIC': ':'
}

def mol_to_mol_by_smi(mol):
    smi = Chem.MolToSmiles(mol)
    assert smi, Exception("mol_to_mol_by_smi: Chem.MolToSmiles(mol) failed!")
    mol = Chem.MolFromSmiles(smi)
    assert mol, Exception(f"mol_to_mol_by_smi: Chem.MolFromSmiles('{smi}') failed!")
    return mol

def smi_to_smi_by_mol(smi: str):
    mol = Chem.MolFromSmiles(smi)
    assert mol, Exception(f"smi_to_smi_by_mol: Chem.MolFromSmiles('{smi}') failed!")
    smi = Chem.MolToSmiles(mol)
    assert smi, Exception("smi_to_smi_by_mol: Chem.MolToSmiles(mol) failed!")
    return smi

def add_atom_map(mol, ignore_dummy=True, start_id=1):
    next_id = start_id
    for atom in mol.GetAtoms():
        if not (ignore_dummy and atom.GetAtomicNum() == 0):
            atom.SetAtomMapNum(next_id)
            next_id += 1
    return mol, next_id

def rm_atom_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def check_mirror_sym(frag, p):
    # May be affected by AtomNote!!
    _frag = copy.deepcopy(frag)
    for i, dummy_atom_idx in enumerate(p):
        _frag.GetAtomWithIdx(dummy_atom_idx).SetIsotope(i+1)
    smi_1 = Chem.MolToSmiles(_frag)
    _frag = copy.deepcopy(frag)
    for i, dummy_atom_idx in enumerate(reversed(p)):
        _frag.GetAtomWithIdx(dummy_atom_idx).SetIsotope(i+1)
    smi_2 = Chem.MolToSmiles(_frag)
    check_flag = smi_1 == smi_2
    return check_flag

def possible_mirror_permutations(groups, items):
    item_to_group = dict()
    for group in groups:
        for item in group:
            item_to_group[item] = group
    assert sorted(item_to_group.keys()) == sorted(items)
    def it(tup_lst, remain_items):
        if len(remain_items) <= 1:
            l_lst, r_lst = zip(*tup_lst)
            yield list(l_lst) + remain_items + list(reversed(r_lst))
        for l_idx, l_item in enumerate(remain_items):
            for r_item in remain_items[-1:l_idx:-1]:
                if r_item in item_to_group[l_item]:
                    tup = (l_item, r_item)
                    new_remain_items = list(remain_items)
                    for x in tup:
                        new_remain_items.remove(x)
                    new_tup_lst = list(tup_lst)
                    new_tup_lst.append(tup)
                    yield from it(new_tup_lst, new_remain_items)
    if len(items) == 1:
        yield items
    else:
        yield from it([], items)

def find_mirror_sym(mol_with_dummy):
    frag = copy.deepcopy(mol_with_dummy)
    dummy_atom_idx_list = []
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_atom_idx_list.append(atom.GetIdx())
            atom.SetIsotope(0)
    num_dummy_atom = len(dummy_atom_idx_list)

    dummy_mark_smi_list = []
    for dummy_atom_idx in dummy_atom_idx_list:
        _frag = copy.deepcopy(frag)
        _frag.GetAtomWithIdx(dummy_atom_idx).SetIsotope(1)
        dummy_mark_smi_list.append(Chem.MolToSmiles(_frag))

    _indices = defaultdict(list)
    for i, x in zip(dummy_atom_idx_list, dummy_mark_smi_list):
        _indices[x].append(i)
    # print('dummy_mark_smi_list', dummy_mark_smi_list)
    # print('_indices', _indices)

    sym_smi_lst = [k for k, v in _indices.items() if len(v) % 2 == 0]
    asym_smi_lst = [k for k, v in _indices.items() if len(v) % 2 == 1]
    # print("sym_smi_lst, asym_smi_lst", sym_smi_lst, asym_smi_lst)

    # print("dummy_atom_idx_list", dummy_atom_idx_list)
    found_flag, p = False, dummy_atom_idx_list
    if num_dummy_atom % 2 == len(asym_smi_lst):
        for p in possible_mirror_permutations(_indices.values(), dummy_atom_idx_list):
            # print(p)
            if check_mirror_sym(frag, p):
                found_flag = True
                break
    return found_flag, p

def fragment(input_mol, abbr=True) -> List[str]:
    input_smi = Chem.MolToSmiles(input_mol)
    mol = Chem.MolFromSmiles(input_smi)

    _tup = tuple(zip(*FindBRICSBonds(mol)))
    if _tup:
        bond_begin_end_atom_idx_list = _tup[0]
    else:
        # No BRICS bond found
        return [Chem.MolToSmiles(mol)]

    # print(bond_begin_end_atom_idx_list)
    bond_idx_list = []
    for begin_atom_idx, end_atom_idx in bond_begin_end_atom_idx_list:
        bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
        bond_idx_list.append(bond.GetIdx())
    # print(bond_idx_list)

    _tmp_mol = Chem.FragmentOnBonds(mol, bondIndices=bond_idx_list, addDummies=True, 
        dummyLabels=list(zip(*(range(1, len(bond_idx_list)+1),)*2)))

    frag_list = Chem.GetMolFrags(_tmp_mol, asMols=True)
    frag_list = [mol_to_mol_by_smi(x) for x in frag_list]
    frag_edge_dict = defaultdict(list)
    lib_frag_list = []
    mirror_sym_node_list = []

    for frag_idx, frag in enumerate(frag_list):
        is_mirror, dummy_atom_idx_list = find_mirror_sym(frag)

        if is_mirror:
            mirror_sym_node_list.append(frag_idx)

        num_dummy_atom = 0
        for dummy_atom_idx in dummy_atom_idx_list:

            dummy_atom = frag.GetAtomWithIdx(dummy_atom_idx)
            dummy_atom_id = dummy_atom.GetIsotope()

            num_dummy_atom += 1
            dummy_atom.SetIsotope(num_dummy_atom)
            frag_edge_dict[dummy_atom_id].append((frag_idx, num_dummy_atom))
                
        lib_frag_list.append((Chem.MolToSmiles(frag), num_dummy_atom, ExactMolWt(frag)))

    assert sorted(frag_edge_dict.keys()) == list(range(1, len(bond_idx_list)+1)), \
        Exception(f"The number of edges cannot correspond "
        f"{sorted(frag_edge_dict.keys())} | {len(bond_idx_list)}")

    frag_edge_list = [(x[0], x[1]-1, y[0], y[1]-1) for x, y in frag_edge_dict.values()]
    # print(frag_edge_list)

    first_lib_frag = max(lib_frag_list, key=lambda x: x[2])
    first_lib_frag_idx = lib_frag_list.index(first_lib_frag)
    num_child_list = [x[1] for x in lib_frag_list]

    # print(mirror_sym_node_list)
    seq = OrderedDFS.serialize(
        num_child_list, frag_edge_list, first_lib_frag_idx, mirror_sym_node_list,
        abbr=abbr)
    frag_seq = [x if x == '.' else lib_frag_list[x][0] for x in seq]

    return frag_seq

def reconstruct(frag_seq: List[str], abbr=True) -> Any:
    num_child_list = []
    frag_smi_list = []
    seq = []
    for x in frag_seq:
        if x != '.':
            num_child_list.append(x.count('*'))
            seq.append(len(frag_smi_list))
            frag_smi_list.append(x)
        else:
            seq.append('.')
    frag_edge_list = OrderedDFS.deserialize(seq, num_child_list, abbr)

    lib_dummy_dict = dict()
    num_dummy_atom = 0
    remain_frag_list = []
    for frag_idx, frag_smi in enumerate(frag_smi_list):
        frag = Chem.MolFromSmiles(frag_smi)
        dummy_atom_id_set = set()
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_atom = atom
                dummy_atom_id = dummy_atom.GetIsotope()

                neigh_atom = dummy_atom.GetNeighbors()[0]
                neigh_atom_id = neigh_atom.GetAtomMapNum()

                bond_type = frag.GetBondBetweenAtoms(dummy_atom.GetIdx(), neigh_atom.GetIdx()).GetBondType()
                num_dummy_atom += 1
                dummy_atom.SetIsotope(num_dummy_atom)
                dummy_atom_id_set.add(num_dummy_atom)

                lib_dummy_dict[(frag_idx, dummy_atom_id)] = num_dummy_atom, bond_type

        remain_frag_list.append((frag, dummy_atom_id_set))

    find_dummy = lambda id: next(j for j, (frag, ids) in enumerate(remain_frag_list) if id in ids)

    for frag_edge in frag_edge_list:
        dummy_id_a, bond_type_a = lib_dummy_dict[(frag_edge[0], frag_edge[1]+1)]
        dummy_id_b, bond_type_b = lib_dummy_dict[(frag_edge[2], frag_edge[3]+1)]
        assert bond_type_a == bond_type_b, Exception("Unmatched bond type")
        bond_char = bond_type_name_to_char_dict[bond_type_a.name]

        frag_a, ids_a = remain_frag_list.pop(find_dummy(dummy_id_a))
        frag_b, ids_b = remain_frag_list.pop(find_dummy(dummy_id_b))

        rs = f'[*:1]{bond_char}[{dummy_id_a}*].[*:2]{bond_char}[{dummy_id_b}*]>>[*:1]{bond_char}[*:2]'
        rxn = ReactionFromSmarts(rs)
        pr = rxn.RunReactants((frag_a, frag_b))
        assert len(pr) > 0, Exception("Error in reaction")
        frag_c = pr[0][0]
        ids_c = (ids_a | ids_b) - set((dummy_id_a, dummy_id_b))
        assert len(ids_c) == len(ids_a) + len(ids_b) - 2, Exception("Error in collection operation")
        remain_frag_list.append((frag_c, ids_c))

    assert len(remain_frag_list) == 1, Exception("There are redundant fragments")
    back_smi = Chem.MolToSmiles(remain_frag_list[0][0])
    assert back_smi.count('*') == 0, Exception("There are redundant dummy atoms")
    back_mol = Chem.MolFromSmiles(back_smi)
    
    return back_mol
