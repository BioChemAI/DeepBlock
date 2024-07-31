import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import logging
import hashlib
from numbers import Number
from pathlib import Path
import shutil
import os
import sys
import re
import time
from typing import Dict, Iterable, List, Sequence, Tuple, Union
from joblib import Parallel, delayed
import warnings
import pexpect
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from meeko import MoleculePreparation, PDBQTMolecule, RDKitMolCreate
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from ..exceptions import IllegalSDFException, IllegalSMILESException, RDKitException

from ..utils import LoggingTrackerJoblibCallback, StrPath, Toc, TqdmTrackerJoblibCallback, \
    auto_dump, auto_load, AutoPackUnpack, \
    child_safe_wait, chmod_plus_x, convert_bytes, delayed_return_exception, hook_joblib, ifn, ob_log_handle, \
    rdkit_log_handle, sequential_file_hash, file_hash, use_path, download_pbar, vin

DEFAULT_VINA_CPU = 4
REMOVE_HETATM_IN_PDB_FILE = os.getenv('RFRAGLI2_DTB_REMOVE_HETATM_IN_PDB_FILE', 
                                      'false').lower() in ['true', '1', 't', 'y', 'yes']

@dataclass
class VinaBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def get_txt(self, ndigits=5) -> bytes:
        return ''.join(f'{k} = {str(round(v, ndigits))}\n' for k, v in asdict(self).items()).encode()

class DockingToolBox:
    """Highly integrated and automated docking tool box"""

    DEFAULT_DOCK_BACKEND = "vina"
    ALL_FEATURES = {'dock', 'prepare_receptor', 'prepare_ligand'}
    QVINA_FAMILY = {"qvina2", "qvinaw", "qvinaws"}
    VINA_FAMILY = {"vina"}
    DOCK_BACKEND_BUILTIN = QVINA_FAMILY | VINA_FAMILY
    assert DEFAULT_DOCK_BACKEND in DOCK_BACKEND_BUILTIN
    DEFAULT_DOCK_TIMEOUT = {
        "qvina2": 10*60, "qvinaw": 20*60, "qvinaws": 40*60, 
        "vina": 30*60,
        "other": 10*60
    }
    DOCK_BACKEND_SUPPORT_MACROCYCLES = {"vina"}
    assert DOCK_BACKEND_SUPPORT_MACROCYCLES.issubset(DOCK_BACKEND_BUILTIN)
    DOCK_BACKEND_DONT_SUPPORT_MACROCYCLES = DOCK_BACKEND_BUILTIN - DOCK_BACKEND_SUPPORT_MACROCYCLES
    QVINA_BINARY_TO_DOCK_BACKEND = {
        "qvina2.1": "qvina2",
        "qvina-w": "qvinaw",
        "qvina-w_serial": "qvinaws"
    }
    assert set(QVINA_BINARY_TO_DOCK_BACKEND.values()) == QVINA_FAMILY
    QVINA_DOWNLOAD_BASE_UTL = {
        "github": "https://github.com/QVina/qvina/raw/f4bb3b1073a0d50bb2f1fdd14d38594f937602ee/bin/",
        "pizyds": "https://pizyds-rfragli2-bucket.bj.bcebos.com/qvina-f4bb3b1073a0d50bb2f1fdd14d38594f937602ee/bin/"
    }
    VINA_BINARY_TO_DOCK_BACKEND = {
        "vina_1.2.3_linux_x86_64": "vina",
    }
    assert set(VINA_BINARY_TO_DOCK_BACKEND.values()) == VINA_FAMILY
    VINA_DOWNLOAD_UTL = {
        "github": "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64",
        "pizyds": "https://pizyds-rfragli2-bucket.bj.bcebos.com/vina_1.2.3_linux_x86_64"
    }
    ADFR_DOWNLOAD_UTL = {
        "ccsb": "https://ccsb.scripps.edu/adfr/download/1038/",
        "pizyds": "https://pizyds-rfragli2-bucket.bj.bcebos.com/ADFRsuite_x86_64Linux_1.0.tar.gz"
    }
    ADFR_TARBALL_NAME = "ADFRsuite_x86_64Linux_1.0.tar.gz"
    OBABEL_BACKEND = "pybel"
    AVALIABLE_OBABEL_BACKEND = {"pybel", "direct", "adfr"}

    is_search_space_volume_warned = False
    is_support_macrocycles_warned = False
    is_add_hs_before_gen_3d_warned = False
    is_dock_macrocycles_but_not_meeko_warned = False

    opt_args = (
        'dock_backend', 'dock_macrocycles',
        'qvina_source', 'vina_source',
        'adfr_backend', 'adfr_source', 'msms_academic',
        'work_dn',
        'debug_child',
        'required_features'
    )
    @classmethod
    def argument_group(cls, group: argparse._ArgumentGroup):
        group.add_argument("--dock-backend", type=str, 
                           default=cls.DEFAULT_DOCK_BACKEND, help='"qvina2", "qvinaw", "qvinaws", "vina" or specify the executable.')
        group.add_argument('--dock_macrocycles', action='store_true', 
                           default=None, help="Automatic or manual selection if None.")
        group.add_argument('--no-dock-macrocycles', dest='dock_macrocycles', action='store_false')
        group.add_argument("--qvina-source", type=str, 
                           default="github", choices=cls.QVINA_DOWNLOAD_BASE_UTL.keys(),
                           help='"github", "pizyds"(If the former has network problems) or None(Disable download).')
        group.add_argument("--vina-source", type=str, 
                           default="github", choices=cls.VINA_DOWNLOAD_UTL.keys(),
                           help='"github", "pizyds" or None(Disable download).')

        group.add_argument("--adfr-backend", type=str, 
                           default="builtin", help='"builtin" or specify the directory.')
        group.add_argument("--adfr-source", type=str, 
                           default="ccsb", choices=cls.ADFR_DOWNLOAD_UTL.keys(),
                           help='"ccsb", "pizyds" or None(Disable download).')
        group.add_argument("--msms-academic", action="store_true")

        group.add_argument("--work-dn", type=str, default="work/docking_toolbox")
        group.add_argument("--debug-child", action="store_true")
        group.add_argument("--required-features", nargs='+', type=str, 
                           default=list(cls.ALL_FEATURES))
        return group

    dock_kwargs_opt_args = (
        'cpu', 'exhaustiveness'
    )
    @classmethod
    def dock_kwargs_argument_group(cls, group: argparse._ArgumentGroup):
        group.add_argument("--cpu", type=int, default=4)
        group.add_argument("--exhaustiveness", type=int, default=32)
        return group

    @classmethod
    @rdkit_log_handle()
    def embed_mol(cls, mol: Mol) -> bool:
        # https://github.com/rdkit/rdkit/issues/2575
        ret_code = AllChem.EmbedMolecule(mol, randomSeed=20230309)
        return ret_code == 0

    @classmethod
    def embed_smi(cls, smi: str) -> bool:
        with rdkit_log_handle() as rdLog:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                raise RDKitException(rdLog())
            mol.UpdatePropertyCache(strict=False)
            mol = Chem.AddHs(mol)
        return cls.embed_mol(mol)

    def __init__(self, 
                 dock_backend: str=DEFAULT_DOCK_BACKEND, dock_macrocycles: bool=None,
                 qvina_source: str="github", vina_source: str="github", 
                 adfr_backend: str="builtin", adfr_source: str="ccsb", msms_academic: bool=False,
                 work_dn: StrPath="work/docking_toolbox",
                 debug_child: bool=False,
                 required_features: Iterable[str]=ALL_FEATURES) -> None:
        """Toolbox for docking

        Pipline: Download utils by *_source -> Check health status of *_backend
        
        - dock_backend: "qvina2", "qvinaw", "qvinaws", "vina" or specify the executable.
        - dock_macrocycles: Automatic or manual selection if None.
            - Vina: True (Vina >= 1.2.3)
            - Other: False
        - qvina_source: "github", "pizyds"(If the former has network problems) or None(Disable download).
        - vina_source: "github", "pizyds" or None(Disable download).

        - adfr_backend: "builtin" or specify the directory.
        - adfr_source: "ccsb", "pizyds" or None(Disable download).
        - msms_academic: True, False.

        - required_features: ('dock', 'prepare_receptor', 'prepare_ligand')
        """

        if sys.platform not in ("linux", "linux2"):
            raise Exception("Docking tool box is only suitable for Linux")

        self._apu = AutoPackUnpack()

        self.dock_backend = dock_backend
        self.dock_macrocycles = ifn(dock_macrocycles, dock_backend in self.DOCK_BACKEND_SUPPORT_MACROCYCLES)

        self.debug_child = debug_child
        self.work_dn = use_path(dir_path=work_dn)

        self.msms_academic = msms_academic
        self.required_features = set(required_features)

        if 'dock' in required_features:
            self.download_vina(qvina_source, vina_source)
            self.vina_fn = self.get_vina_fn(dock_backend)
        if 'prepare_receptor' in required_features:
            self.download_adfr(adfr_source)
            self.adfr_dn = self.get_adfr_dn(adfr_backend)

    def check_required_feature(self, feature: str) -> None:
        if not feature in self.ALL_FEATURES:
            raise Exception(f"Feature {feature} is not support.")
        if not feature in self.required_features:
            raise Exception(f"Feature {feature} is not activated.")

    def health_check_vina(self, vina_fn: Path) -> None:
        child = pexpect.spawn(vina_fn.as_posix(),  ["--version"], timeout=60)
        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)

    def health_check_adfr(self, adfr_dn: Path) -> None:
        child = pexpect.spawn((adfr_dn / "bin/prepare_receptor").as_posix(),  ["-h"], timeout=60)
        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)

    def download_vina(self, qvina_source: str, vina_source: str):
        if qvina_source is not None:
            vin(qvina_source, self.QVINA_DOWNLOAD_BASE_UTL)
            for _bin in self.QVINA_BINARY_TO_DOCK_BACKEND:
                _fn = use_path(file_path=self.work_dn / 'utils/QVina' / _bin)
                if not _fn.exists():
                    _url = f'{self.QVINA_DOWNLOAD_BASE_UTL[qvina_source].rstrip("/")}/{_bin}'
                    print(f"Download -> {_url} to {_fn}")
                    try:
                        _bytes = download_pbar(_url, desc=f"↓ {_bin}")
                    except Exception as err:
                        raise Exception(f"Quick Vina Binary {_url} download failed due to {err}! "
                                        f"Try to use `pizyds` as qvina_source or download manually.")
                    with open(_fn, 'wb') as f:
                        f.write(_bytes)
                    print(f"Done -> size: {convert_bytes(len(_bytes))}")
                    # Grant permissions: chmod +x
                    chmod_plus_x(_fn)
                    print(f"Chmod +x -> {_fn}")

        if vina_source is not None:
            vin(vina_source, self.VINA_DOWNLOAD_UTL)
            _bin = next(iter(self.VINA_BINARY_TO_DOCK_BACKEND))
            _fn = use_path(file_path=self.work_dn / 'utils/AutoDock-Vina' / _bin)
            if not _fn.exists():
                _url = self.VINA_DOWNLOAD_UTL[vina_source]
                print(f"Download -> {_url} to {_fn}")
                try:
                    _bytes = download_pbar(_url, desc=f"↓ {_bin}")
                except Exception as err:
                    raise Exception(f"AutoDock Vina Binary {_url} download failed due to {err}! "
                                    f"Try to use `pizyds` as vina_source or download manually.")
                with open(_fn, 'wb') as f:
                    f.write(_bytes)
                print(f"Done -> size: {convert_bytes(len(_bytes))}")
                # Grant permissions: chmod +x
                chmod_plus_x(_fn)
                print(f"Chmod +x -> {_fn}")

    def get_vina_fn(self, backend: str) -> Path:
        if backend in self.DOCK_BACKEND_BUILTIN:
            if backend in self.QVINA_BINARY_TO_DOCK_BACKEND.values():
                _bin = next(k for k, v in self.QVINA_BINARY_TO_DOCK_BACKEND.items() if v == backend)
                vina_fn = use_path(file_path=self.work_dn / 'utils/QVina' / _bin)
            elif backend in self.VINA_BINARY_TO_DOCK_BACKEND.values():
                _bin = next(k for k, v in self.VINA_BINARY_TO_DOCK_BACKEND.items() if v == backend)
                vina_fn = use_path(file_path=self.work_dn / 'utils/AutoDock-Vina' / _bin)
            self.health_check_vina(vina_fn)
        else:
            vina_fn = Path(backend)
            if not vina_fn.exists():
                raise Exception(f"Dock executable '{vina_fn}' not exists! " +
                                "Do you mean 'qvina2'?" if backend == "qvina" else "")
        return vina_fn

    def download_adfr(self, source: str):
        if source is not None:
            vin(source, self.ADFR_DOWNLOAD_UTL)
            adfr_dn = Path(self.work_dn / 'utils/ADFRsuite-1.0')
            if not adfr_dn.exists():
                # Second check whether ADFRSuite's tarball is downloaded
                tarball_fn = use_path(file_path=self.work_dn / 'download' / self.ADFR_TARBALL_NAME)
                if not tarball_fn.exists():
                    # Download
                    _url = self.ADFR_DOWNLOAD_UTL[source]
                    print(f"Download -> {_url} to {tarball_fn}")
                    try:
                        _bytes = download_pbar(_url, desc=f"↓ {self.ADFR_TARBALL_NAME}")
                    except Exception as err:
                        raise Exception(f"AutodockFR Suite {_url} download failed due to {err}! "
                                        f"Try to use `pizyds_rfragli2_bucket` as adfr_source or download manually.")
                    with open(tarball_fn, 'wb') as f:
                        f.write(_bytes)
                    print(f"Done -> size: {convert_bytes(len(_bytes))}")

                # Install
                dec_dn = use_path(file_path=self.work_dn / 'download' / f'dec_{self.ADFR_TARBALL_NAME}')
                self._apu.auto_unpack(tarball_fn, dec_dn, rm_dn=True)
                print(f"Decompress -> {dec_dn}")

                print("The molecular surface calculation software (MSMS) is freely available for academic research.\n"
                      "For obtainig commercial license usage contact Dr. Sanner at sanner@scripps.edu.")
                if self.msms_academic:
                    print("You have chosen to install this softare for the purpose of academic research.")
                else:
                    print("You rejected the installation of this software.")

                print("The toolbox will automatically answer the prompts in the installation.")
                _time_sleep = 5
                print(f"Installation will start in {_time_sleep} seconds...")
                time.sleep(_time_sleep)

                logfile = open(adfr_dn.with_suffix('.install.log'), 'wb')
                child = pexpect.spawn('bash', 
                                      args=['./install.sh', '-d', adfr_dn.absolute().as_posix()], 
                                      cwd=(dec_dn / 'ADFRsuite_x86_64Linux_1.0').as_posix(),
                                      logfile=logfile, timeout=10*60)
                child.logfile_read = sys.stdout.buffer
                child.expect(r' ACADEMIC INSTALLATION \(Y\/N\) \?')
                child.sendline('Y' if self.msms_academic else 'N')
                child_safe_wait(child)
                logfile.close()

                # Remove
                shutil.rmtree(dec_dn)
                print(f"Remove -> {dec_dn}")
                # Done
                print(f'Done -> {adfr_dn}')

    def get_adfr_dn(self, backend: str) -> Path:
        if backend == "builtin":
            adfr_dn = Path(self.work_dn / 'utils/ADFRsuite-1.0')
            self.health_check_adfr(adfr_dn)
        else:
            adfr_dn = Path(backend)
            self.health_check_adfr(adfr_dn)
        return adfr_dn

    def receptor_pdb_to_pdbqt(self, pdb_fn: Path, pdbqt_fn: Path, 
                              add_hs: bool=True,
                              patch_pdbqt: bool=True,
                              use_reduce: bool=False,
                              reduce_args: List[str]=[]) -> bytes:
        """Receptor: .pdb -> .pdbqt

        > As a prerequisite, a receptor coordinate file must contain all hydrogen atoms.

        If `add_hs` and `use_reduce`, .pdb -> .reduce_add_hs.pdb -> .pdbqt

        `use_reduce` Perhaps more accurate hydrogen atom coordinates can be obtained, 
        but in the program, reduce is prone to collapse and abnormal return values. 

        Notice: If you want to use reduce, please read `reduce -Help` carefully. 
        Some options may affect the results: ['-HIS']
        > -HIS              create NH hydrogens on HIS rings

        I can't completely define the best parameter of reduce, so `use_reduce` is disabled by default.

        Ref: 
        - https://ccsb.scripps.edu/adfr/how-to-create-a-pdbqt-for-my-receptor/
        - https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#preparing-the-receptor
        """
        self.check_required_feature('prepare_receptor')

        if add_hs and use_reduce:
            add_hs_pdb_fn = use_path(file_path=pdbqt_fn.with_suffix(f'.reduce_add_hs{pdb_fn.suffix}'))
            reduce_args_str = f'{" ".join(reduce_args)} ' if len(reduce_args) > 0 else ''
            bash_command = (f'{(self.adfr_dn / "bin/reduce").as_posix()} {reduce_args_str}'
                                                  f'{pdb_fn.as_posix()} > {add_hs_pdb_fn.as_posix()}')
            logfile = open(add_hs_pdb_fn.with_suffix(add_hs_pdb_fn.suffix+'.log'), 'wb')
            child = pexpect.spawn('bash', ['-c', bash_command], 
                                  logfile=logfile, timeout=10*60)
            if self.debug_child: child.logfile_read = sys.stdout.buffer
            err = child_safe_wait(child, is_ignore=True)
            logfile.close()
            if isinstance(err, Exception):
                warnings.warn(f"ChildUnexpectedTerminationException('{err.short_message}')\n"
                              f"Drop reduce result.")
                if add_hs_pdb_fn.exists():
                    add_hs_pdb_fn.rename(add_hs_pdb_fn.with_suffix(add_hs_pdb_fn.suffix+'.drop'))
            else:
                pdb_fn = add_hs_pdb_fn
        
        # For PDBbind pdb files, always raise an error:
        # '  ' apparently composed of not std residues. Deleting
        # AttributeError: member babel_type not found
        # This should be a bug in prepare_receptor, temporarily resolved:
        if REMOVE_HETATM_IN_PDB_FILE:
            remove_hetatm_pdb_fn = use_path(file_path=pdbqt_fn.with_suffix(f'.remove_hetatm{pdb_fn.suffix}'))
            with open(pdb_fn, 'r') as f:
                pdb_lines = f.readlines()
            repaired_lines = [line for line in pdb_lines if not line.startswith('HETATM')]
            if repaired_lines != pdb_lines:
                with open(remove_hetatm_pdb_fn, 'w') as f:
                    f.writelines(repaired_lines)
                warnings.warn(f"Remove {len(pdb_lines) - len(repaired_lines)} HETATM lines in pdb file {pdb_fn}")
                pdb_fn = remove_hetatm_pdb_fn

        # In short, before the commit(Oct 23, 2020), the non-zero code return by reduce may make 
        # the output incomplete, so we should give up the result of reduce.
        # https://github.com/rlabduke/reduce/issues/5
        # ADFR reduce: version 3.23 05/21/2013, Copyright 1997-2013, J. Michael Word
        # Another error: exit(139) - Segmentation fault

        # Read code in ADFRsuite-1.0/CCSBpckgs/AutoDockTools/MoleculePreparation.py:177
        # -A checkhydrogens: When there is at least one hydrogen atom in PDB, 
        # prepare_receptor wouldn't add hydrogens.
        #
        # From the experiment, we found that:
        # -A hydrogens: prepare_receptor will first read the coordinates of hydrogen atoms 
        # (if add with reduce), supplement the protonation state of imidazole 
        # (additional polar hydrogen atoms), and then output pdbqt.
        #
        # So my choice is the latter

        adfr_add_hs_args = ["-A", "hydrogens"] if add_hs else []
        logfile = open(pdbqt_fn.with_suffix(pdbqt_fn.suffix+'.log'), 'wb')
        child = pexpect.spawn((self.adfr_dn / "bin/prepare_receptor").as_posix(),  
                              [*adfr_add_hs_args, "-r", pdb_fn.as_posix(), "-o", pdbqt_fn.as_posix()],
                              logfile=logfile, timeout=10*60)
        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)
        logfile.close()

        # There is still a serious bug in the ADFR prepare_receptor, which will produce 
        # corrupted pdbqt files, such as the two lines:
        # ATOM   1903  HE2A1GLN A 237       4.158   5.378 -54.583  1.00  0.00     0.159 HD
        # ATOM   1904  HE2A2GLN A 237       2.395   5.397 -54.887  1.00  0.00     0.159 HD
        # I think this is because it doesn't handle altLoc correctly. It must have only 
        # one character in COLUMNS 17. This error makes vina unable to parse the line.
        # It should be treated like:
        # ATOM   4030  NE2AHIS A 453      -1.520  36.833 -67.804  0.50 27.89    -0.360 N 
        # ATOM   4031  HE2AHIS A 453      -1.609  35.823 -67.918  1.00  0.00     0.166 HD
        # ATOM   1944 1HE2 GLN A 241       2.736  14.648 -54.193  1.00  0.00     0.159 HD
        # ATOM   1945 2HE2 GLN A 241       1.180  15.363 -53.718  1.00  0.00     0.159 HD
        # work/docking_toolbox/utils/ADFRsuite-1.0/CCSBpckgs/MolKit/pdbWriter.py:254
        # Here, simply patch the output pdbqt file. I hope this is the only case.
        # https://userguide.mdanalysis.org/1.0.0/formats/reference/pdbqt.html

        if patch_pdbqt:
            with open(pdbqt_fn, 'r') as f:
                pdbqt_lines = f.readlines()

            repaired_lines_dic = dict()
            for _idx, _line in enumerate(pdbqt_lines):
                if _line.startswith('ATOM '):
                    if len(_line) == 80:
                        # Normal
                        pass
                    elif len(_line) == 81 and _line[12] == ' ' and _line[20] != ' ':
                        # Can be repaired
                        _seg = _line[12:21]                         # ' HE2A1GLN' len(9)
                        _seg = f'{_seg[5]}{_seg[1:5]}{_seg[6:]}'    # '1HE2AGLN'  len(9)
                        repaired_lines_dic[_idx] = f'{_line[:12]}{_seg}{_line[21:]}'
                    else:
                        raise Exception(f"An error in the pdbqt file was found and cannot be repaired!\n"
                                        f"At row {_idx+1}, len {len(_line)}:\n"
                                        f"{_line}")
                    
            if len(repaired_lines_dic) > 0:
                bak_pdbqt_fn = Path(pdbqt_fn.as_posix())
                bak_pdbqt_fn.rename(pdbqt_fn.with_suffix('.bak_patch'+pdbqt_fn.suffix))
                for _idx, _line in repaired_lines_dic.items():
                    pdbqt_lines[_idx] = _line
                with open(pdbqt_fn, 'w') as f:
                    f.writelines(pdbqt_lines)
                warnings.warn(f"Fixed {len(repaired_lines_dic)} errors in pdbqt file {pdbqt_fn}\n"
                              f"{''.join(repaired_lines_dic.values())}")
        
        with open(pdbqt_fn, 'rb') as f:
            pdbqt_content = f.read()
        
        return pdbqt_content
    
    def ligand_mol_to_pdbqt(self, mol: Mol, pdbqt_fn: Path, 
                            add_hs: bool=True, gen_3d: bool=False,
                            backend=None) -> Tuple[bytes, Mol]:
        """Receptor: mol -> .pdbqt

        If backend == "meeko", mol -> .pdbqt                            *Preferred if dock_macrocycles
        If backend == "adfr", mol -> .rdkit_to_adfr.pdb -> .pdbqt
        If backend == "obabel"
            If OBABEL_BACKEND in ("adfr", "direct"), mol -> .rdkit_to_obabel.sdf -> .pdbqt
            If OBABEL_BACKEND == "pybel", mol -> (mol_block) -> .pdbqt  *Preferred if not dock_macrocycles
        
        > Meeko does not calculate 3D coordinates or assign protonation states. 
        > Input molecules must have explicit hydrogens.

        Ref: 
        - https://github.com/forlilab/Meeko
        - https://www.blopig.com/blog/2022/08/meeko-docking-straight-from-smiles-string/
        - https://autodock-vina.readthedocs.io/en/latest/docking_basic.html#preparing-the-ligand
        """

        # Automatically determine the backend
        backend = ifn(backend, "meeko" if self.dock_macrocycles else "obabel")

        if all((gen_3d, not add_hs,
                not self.__class__.is_add_hs_before_gen_3d_warned)):
            warnings.warn(f"Unless the input mol has been hydrogenated, add_hs -> gen_3d!")
            self.__class__.is_add_hs_before_gen_3d_warned = True

        if not isinstance(mol, Mol):
            raise ValueError(f'Invalid mol')

        if add_hs:
            with rdkit_log_handle() as rdLog:
                mol.UpdatePropertyCache(strict=False)
                mol = Chem.AddHs(mol)
                if not mol:
                    raise RDKitException(rdLog())
        if gen_3d:
            with rdkit_log_handle() as rdLog:
                # https://github.com/rdkit/rdkit/issues/2575
                if not self.embed_mol(mol):
                    raise RDKitException(f'Unable to embed the molecule '
                                         f'{Chem.MolToSmiles(mol)} {rdLog()}')
        
        # If meeko + QVina, QVina will throw exception: 
        # ATOM syntax incorrect: "CG0" is not a valid AutoDock type. Note that AutoDock atom types are case-sensitive.
        # It seems that QVina couldn't dock with macrocycles, but Vina >= 1.2.3 support.
        # Meeko: Sampling of macrocycle conformers (paper) is enabled by default.
        # https://autodock-vina.readthedocs.io/en/latest/docking_macrocycle.html#docking-with-macrocycles
        # https://github.com/forlilab/Meeko/issues/28
        if all((backend == "meeko",
               self.dock_backend in self.DOCK_BACKEND_DONT_SUPPORT_MACROCYCLES,
               not self.__class__.is_support_macrocycles_warned)):
            warnings.warn(f"QVina couldn't dock with macrocycles, don't use Meeko!")
            self.__class__.is_support_macrocycles_warned = True

        # However ADFR prepare_ligand only support .pdb or .mol2 or .pdbq format, 
        # RDKit can only output .pdb and cannot transfer bond information. It is also unqualified.

        # So OpenBabel is the most versatile

        if backend == "meeko":
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()
            pdbqt_content = pdbqt_string.encode()
            with open(pdbqt_fn, 'wb') as f:
                f.write(pdbqt_content)
        elif backend == "adfr":
            pdb_fn = pdbqt_fn.with_suffix('.rdkit_to_adfr.pdb')
            pdb_str = Chem.MolToPDBBlock(mol)
            with open(pdb_fn, 'w') as f:
                f.write(pdb_str)
            # AssertionError: ligand.sdf does't exist
            # There is a bug in ADFR prepare_ligand. The input sdf must be in the current directory
            cwd_dn = pdb_fn.parent
            logfile = open(pdbqt_fn.with_suffix(pdbqt_fn.suffix+'.adfr.log'), 'wb')
            child = pexpect.spawn((self.adfr_dn / "bin/prepare_ligand").absolute().as_posix(),  
                                  ["-l", pdb_fn.relative_to(cwd_dn).as_posix(), 
                                  "-o", pdbqt_fn.absolute().as_posix()],
                                  cwd=cwd_dn.as_posix(),
                                  logfile=logfile, timeout=10*60)
            if self.debug_child: child.logfile_read = sys.stdout.buffer
            child_safe_wait(child)
            logfile.close()

            with open(pdbqt_fn, 'rb') as f:
                pdbqt_content = f.read()
        elif backend == "obabel":
            vin(self.OBABEL_BACKEND, self.AVALIABLE_OBABEL_BACKEND)
            if self.OBABEL_BACKEND in ("adfr", "direct"):
                # Error: libSM.so.6: cannot open shared object file: No such file or directory
                # Cannot directly use the openbabel provided by ADFR (additional libraries need to be installed)
                # You can install openbabel directly using conda
                sdf_fn = pdbqt_fn.with_suffix('.rdkit_to_obabel.sdf')
                with Chem.SDWriter(sdf_fn.as_posix()) as w:
                    w.write(mol)
                logfile = open(pdbqt_fn.with_suffix(pdbqt_fn.suffix+'.obabel.log'), 'wb')
                child = pexpect.spawn((self.adfr_dn / "bin/obabel").as_posix() if self.OBABEL_BACKEND == "adfr" else "obabel", 
                                    ["-i", sdf_fn.as_posix(), "-o", pdbqt_fn.as_posix()],
                                    logfile=logfile, timeout=60)
                if self.debug_child: child.logfile_read = sys.stdout.buffer
                child_safe_wait(child)
                logfile.close()
            elif self.OBABEL_BACKEND == "pybel":
                # Use pybel module as an alternative
                mol_block = Chem.MolToMolBlock(mol)
                with ob_log_handle():
                    obmol = pybel.readstring("mol", mol_block)
                    obmol.write("pdbqt", pdbqt_fn.as_posix(), overwrite=True)
            else:
                raise ValueError(f"Unknow openbabel backend: {self.OBABEL_BACKEND}")

            with open(pdbqt_fn, 'rb') as f:
                pdbqt_content = f.read()
        else:
            raise ValueError(f"Unknow backend: {backend}")
        
        return pdbqt_content, mol
    
    def ligand_smi_to_pdbqt(self, smi: str, pdbqt_fn: Path) -> Tuple[bytes, Mol]:
        """Receptor: smi -> .pdbqt

        Calculate 3D coordinates and add hydrogens
        """
        with rdkit_log_handle() as rdLog:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                raise IllegalSMILESException(rdLog())
        pdbqt_content, mol = self.ligand_mol_to_pdbqt(mol, pdbqt_fn, add_hs=True, gen_3d=True)
        return pdbqt_content, mol
        
    def ligand_sdf_to_pdbqt(self, sdf_fn: Path, pdbqt_fn: Path) -> Tuple[bytes, Mol]:
        """Receptor: .sdf -> .pdbqt

        Add hydrogens
        """
        with rdkit_log_handle() as rdLog:
            mol = next(Chem.SDMolSupplier(sdf_fn.as_posix()))
            if not mol:
                raise IllegalSDFException(rdLog())
        pdbqt_content = self.ligand_mol_to_pdbqt(mol, pdbqt_fn, add_hs=True)
        return pdbqt_content, mol

    def ligand_mol_to_box(self, mol: Mol, 
                          box_txt_fn: Path=None,
                          size: Union[Tuple[float, float, float], float]=25.0) -> VinaBox:
        """Ligand: mol -> VinaBox(center_x, center_y, center_z, size_x, size_y, size_z), .box.txt

        It has nothing to do with this function: some people are studying the influence of 
        the size of the docking box: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4468813/
        """
        if not isinstance(mol, Mol):
            raise ValueError(f'Invalid mol')
        
        if isinstance(size, Number):
            size = (size,) * 3

        search_space_volume = size[0]*size[1]*size[2]
        if search_space_volume > 27000 and not self.__class__.is_search_space_volume_warned:
            warnings.warn(f'Search space volume {search_space_volume} is over 27000 Angstrom^3.\n'
                          f'You may need to increase the value of the exhaustiveness to make up for it.\n'
                          f'Ref: https://autodock-vina.readthedocs.io/en/latest/faq.html\n'
                          f'In addition, it is recommended to use qvina-w in this case.\n'
                          f'Ref: https://github.com/QVina/qvina/tree/f4bb3b1073a0d50bb2f1fdd14d38594f937602ee#important-note')
            self.__class__.is_search_space_volume_warned = True

        coords = mol.GetConformer(0).GetPositions()
        center = (coords.max(axis=0) + coords.min(axis=0)) / 2
        box = VinaBox(center_x=center[0], center_y=center[1], center_z=center[2],
                      size_x=size[0], size_y=size[1], size_z=size[2])
        
        box_txt = box.get_txt()
        if box_txt_fn is not None:
            with open(box_txt_fn, 'wb') as f:
                f.write(box_txt)

        return box, box_txt

    def dock(self, receptor_pdbqt_fn: Path, ligand_pdbqt_fn: Path, box_txt_fn: Path, 
             out_pdbqt_fn: Path, log_fn: Path=None,
             cwd: Path=None,
             cpu: int=DEFAULT_VINA_CPU, exhaustiveness: int=32, seed: int=20230308,
             timeout=None) -> bytes:
        self.check_required_feature('dock')

        timeout = ifn(timeout, self.DEFAULT_DOCK_TIMEOUT.get(self.dock_backend, self.DEFAULT_DOCK_TIMEOUT['other']))
        if log_fn is None:
            log_fn = out_pdbqt_fn.with_name('dock.log')
        pexpect_args = [
            "--receptor", receptor_pdbqt_fn.as_posix(), 
            "--ligand", ligand_pdbqt_fn.as_posix(),
            "--config", box_txt_fn.as_posix(),
            "--cpu", str(cpu),
            "--exhaustiveness", str(exhaustiveness),
            "--seed", str(seed),
            "--out", out_pdbqt_fn.as_posix()
        ]
        logfile = open(log_fn, 'wb')
        child = pexpect.spawn(self.vina_fn.as_posix(), 
                            pexpect_args, cwd=cwd.as_posix() if cwd is not None else None, 
                            logfile=logfile, timeout=timeout)

        if self.debug_child: child.logfile_read = sys.stdout.buffer
        child_safe_wait(child)
        logfile.close()

        with open(out_pdbqt_fn, 'rb') as f:
            out_pdbqt_content = f.read()
        return out_pdbqt_content

    def out_pdbqt_to_affinity(self, pdbqt_fn: Path) -> List[float]:
        p = re.compile(r"REMARK VINA RESULT:\s+(?P<affinity>\S+)")
        with open(pdbqt_fn, 'r') as f:
            pdbqt_str = f.read()
        affinity_lst = [float(m.group("affinity")) for m in p.finditer(pdbqt_str)]
        return affinity_lst

    def out_pdbqt_to_sdf(self, pdbqt_fn: Path, sdf_fn: Path, 
                         backend=None) -> bytes:
        """This function allows the output of the sdf containing only partial molecules or 
        even blanks without throwing an exception.
        """
        backend = ifn(backend, "meeko" if self.dock_backend in self.VINA_FAMILY else "obabel")
        if all((self.dock_macrocycles, backend != "meeko",
                not self.__class__.is_dock_macrocycles_but_not_meeko_warned)):
            warnings.warn(f"Docking macrocycles but not using meeko to parse result is dangerous!")
            self.__class__.is_dock_macrocycles_but_not_meeko_warned = True

        if backend == "meeko":
            try:
                pdbqt_mol = PDBQTMolecule.from_file(pdbqt_fn.as_posix(), skip_typing=True)
                sdf_string, failures = RDKitMolCreate.write_sd_string(pdbqt_mol)
                if len(failures) > 0:
                    warnings.warn(f'meeko error: pdbqt to sdf failed on {failures} for {pdbqt_fn}')
                sdf_content = sdf_string.encode()
                with open(sdf_fn, 'wb') as f:
                    f.write(sdf_content)
            except Exception as err:
                warnings.warn(f'meeko error: {repr(err)}')
                sdf_content = b''
                with open(sdf_fn, 'wb') as f:
                    f.write(sdf_content)

        elif backend == "obabel":
            try:
                failures = []
                with ob_log_handle():
                    with pybel.Outputfile("sdf", sdf_fn.as_posix(), overwrite=True) as sd_f:
                        for i, obmol in enumerate(pybel.readfile('pdbqt', pdbqt_fn.as_posix())):
                            if obmol is not None:
                                sd_f.write(obmol)
                            else:
                                failures.append(i)
                if len(failures) > 0:
                    warnings.warn(f'OpenBabel error: pdbqt to sdf failed on {failures} for {pdbqt_fn}')
                with open(sdf_fn, 'rb') as f:
                    sdf_content = f.read()
            except Exception as err:
                warnings.warn(f'pybel error: {repr(err)}')
                sdf_content = b''
                with open(sdf_fn, 'wb') as f:
                    f.write(sdf_content)

        else:
            raise ValueError(f"Unknow result backend: {backend}")
        return sdf_content
    
    def batch_dock(self, input_fn_or_dn: StrPath, 
                   n_jobs: int=None, n_procs: int=None,
                   is_dev: bool=False) -> '_BatchDock':
        return _BatchDock(self, input_fn_or_dn, n_jobs, n_procs, is_dev)
    
    def batch_prepare(self, work_dn: Path=None) -> '_BatchPrepare':
        return _BatchPrepare(self, work_dn)


class _BatchDock:
    """Batch dock

    Directory structure (Optional)

    - receptor
        - a.receptor.pdbqt
        - b.receptor.pdbqt
    - ligand
        - 1.ligand.pdbqt
        - 2.ligand.pdbqt
    - box
        - A.box.txt
    - complex.json
    - kwargs.json

    Complex list `complex.json`

    ```json
    [
        {
            "id": "test"                                # Unique value (complex_id), such as sha256
            "receptor": "receptor/a.receptor.pdbqt"
            "ligand": "ligand/1.ligand.pdbqt"
            "box": "box/A.box.txt"
        }
    ]
    ```

    Kwargs dict `kwargs.json`

    ```json
    {
        "cpu": 4, 
        "exhaustiveness": 32
    }

    
    You can enter a directory or package it as a tarball `.tar.gz`. 
    Automatically generate `job_id` = `{sha256 of directory Posix absolute path}` or `{sha256 of tarball}`. 
    Tarball will be decompressed to `{work_dn}/jobs/{job_id}`. 
    Results will be saved in `{work_dn}/jobs/{job_id}/docked/` and `{work_dn}/jobs/{job_id}/score.json`

    - `{work_dn}/jobs/{job_id}/docked/`
        - {complex_id}.receptor.pdbqt
        - {complex_id}.docked_ligand.pdbqt
        - {complex_id}.docked_ligand.sdf
        - {complex_id}.dock.box.txt
        - {complex_id}.dock.log

    Score list `{work_dn}/jobs/{job_id}/score.json`
    ```
    [
        {
            "id": "test",
            "affinity": -7.5,       # (kcal/mol)
            "toc": 120              # (seconds)
        }
    ]
    ```
    """
    def __init__(self, dtb: DockingToolBox, input_fn_or_dn: StrPath, 
                   n_jobs: int=None, n_procs: int=None,
                   is_dev: bool=False) -> None:
        self.dtb = dtb
        self.dtb.check_required_feature('dock')
        self._apu = AutoPackUnpack()

        input_fn_or_dn = Path(input_fn_or_dn)
        if input_fn_or_dn.is_dir():
            job_id = hashlib.sha256(self.source_dn.absolute().as_posix().encode()).hexdigest()
            if is_dev: job_id += '_dev'
            self.job_dn = use_path(dir_path=self.dtb.work_dn / 'jobs' / job_id)
            self.source_dn = input_fn_or_dn
        elif input_fn_or_dn.is_file():
            job_id = sequential_file_hash(input_fn_or_dn, hashlib.sha256()).hexdigest()
            if is_dev: job_id += '_dev'
            self.job_dn = use_path(dir_path=self.dtb.work_dn / 'jobs' / job_id)
            self._apu.auto_unpack(input_fn_or_dn, self.job_dn)
            logging.info(f"Decompress -> {input_fn_or_dn} to {self.job_dn}")
            self.source_dn = self.job_dn
        else:
            raise ValueError(f"{input_fn_or_dn.as_posix()} is not dir or file!")
        self.docked_dn = use_path(dir_path=self.job_dn / 'docked')
        
        logging.info(f"job_id: {job_id}, source_dn: {self.source_dn}, job_dn: {self.job_dn}, docked_dn: {self.docked_dn}")

        self.kwargs_dic = auto_load(self.source_dn / "kwargs.json")
        self.complex_lst = auto_load(self.source_dn / "complex.json")
        logging.info(f"kwargs_dic: {self.kwargs_dic}, len(complex_lst): {len(self.complex_lst)}")

        if is_dev:
            self.complex_lst = self.complex_lst[:3]
            logging.info(f"Dev -> len(complex_lst): {len(self.complex_lst)}")

        if n_jobs is None:
            if n_procs is not None:
                self.n_jobs = n_procs // self.kwargs_dic.get('cpu', DEFAULT_VINA_CPU)
            else:
                self.n_jobs = 1
        else:
            self.n_jobs = n_jobs
        logging.info(f"procs_pre_job: {self.kwargs_dic.get('cpu', DEFAULT_VINA_CPU)}, "
                     f"n_procs: {n_procs}, "
                     f"n_jobs: {self.n_jobs}")

    def run_jobs(self):
        
        toc = Toc()
        @delayed_return_exception
        def _par_f_complex(complex):
            toc = Toc()
            complex_id = complex['id']
            out_pdbqt_fn = self.docked_dn / f'{complex_id}.docked_ligand.pdbqt' # 1
            self.dtb.dock(
                receptor_pdbqt_fn=self.source_dn / complex['receptor'],
                ligand_pdbqt_fn=self.source_dn / complex['ligand'],
                box_txt_fn=self.source_dn / complex['box'],
                out_pdbqt_fn=out_pdbqt_fn,
                log_fn=self.docked_dn / f'{complex_id}.dock.log',               # 2
                **self.kwargs_dic
            )
            affinity_lst = self.dtb.out_pdbqt_to_affinity(pdbqt_fn=out_pdbqt_fn)
            self.dtb.out_pdbqt_to_sdf(
                pdbqt_fn=out_pdbqt_fn, 
                sdf_fn=self.docked_dn / f'{complex_id}.docked_ligand.sdf'       # 3
            )
            shutil.copyfile(self.source_dn / complex['box'], 
                            self.docked_dn / f'{complex_id}.dock.box.txt')      # 4
            shutil.copyfile(self.source_dn / complex['receptor'], 
                            self.docked_dn / f'{complex_id}.receptor.pdbqt')    # 5
            result = {
                "id": complex_id,
                "affinity": min(affinity_lst),
                "toc": toc()
            }
            return result

        joblib_cb = LoggingTrackerJoblibCallback(len(self.complex_lst))
        with hook_joblib(joblib_cb):
            result_lst = Parallel(n_jobs=self.n_jobs, verbose=10)(
                _par_f_complex(complex) for complex in self.complex_lst)
        
        _lst = [result for result in result_lst if not isinstance(result, Exception)]

        logging.warning(f'num_failed_complex: {len(self.complex_lst) - len(_lst)}, toc: {toc()}s')

        auto_dump(_lst, self.job_dn / 'score.json', json_indent=True)
        return _lst
    
    def pack_tarball(self, fn: Path=None):
        if fn is None:
            fn = self.job_dn.with_suffix(self._apu.prefer_suffix)
        self._apu.auto_pack(self.job_dn, fn)
        return fn


class _BatchPrepare:

    ReceptorTypeEnum = Enum('ReceptorTypeEnum', 'PDB_FN')
    LigandTypeEnum = Enum('LigandTypeEnum', 'SDF_FN SMI_STR MOL_OBJ')
    BoxTypeEnum = Enum('BoxTypeEnum', 'SDF_FN')

    @dataclass
    class Complex:
        id: str = None
        receptor: str = None    # Note that it needs to be a relative path
        ligand: str = None
        box: str = None

    def __init__(self, dtb: DockingToolBox, work_dn: Path=None) -> None:
        self.dtb = dtb
        self.dtb.check_required_feature('prepare_receptor')
        self.dtb.check_required_feature('prepare_ligand')
        self._apu = AutoPackUnpack()

        if work_dn is None:
            self.work_dn = self.dtb.work_dn / f'tmp_batch-prepare_{int(time.time()*1000)}'
        else:
            self.work_dn = work_dn

        self.receptor_mem = dict()
        self.ligand_mem = dict()
        self.box_mem = dict()

        self.delayed_jobs = list()
        self.complex_lst = list()

    def add_complex(self, 
                    receptor: Tuple[ReceptorTypeEnum, Path], 
                    ligand: Tuple[LigandTypeEnum, Union[Path, str]],
                    box: Tuple[BoxTypeEnum, Path]):

        _hash = hashlib.sha256
        
        complex = self.Complex()
        complex_h = _hash()

        @delayed_return_exception
        def _par_f_receptor_pdb(pdb_fn: Path, pdbqt_fn: Path):
            self.dtb.receptor_pdb_to_pdbqt(pdb_fn=pdb_fn, pdbqt_fn=pdbqt_fn)

        # receptor
        if receptor[0] == self.ReceptorTypeEnum.PDB_FN:
            pdb_fn = receptor[1]
            h = file_hash(pdb_fn, _hash())
            uid = h.hexdigest()
            if uid not in self.receptor_mem:
                pdbqt_fn = use_path(file_path=self.work_dn / "receptor" / f"{uid}.receptor.pdbqt")
                self.receptor_mem[uid] = pdbqt_fn
                self.delayed_jobs.append(dict(t='parallel', f=_par_f_receptor_pdb(pdb_fn=pdb_fn, pdbqt_fn=pdbqt_fn)))
        else:
            raise ValueError(f"Unknown receptor type: {receptor[0]}")
        complex_h.update(h.digest())
        complex.receptor = self.receptor_mem[uid].relative_to(self.work_dn).as_posix()

        @delayed_return_exception
        def _par_f_ligand_sdf(sdf_fn: Path, pdbqt_fn: Path, box_txt_fn: Path):
            _, mol = self.dtb.ligand_sdf_to_pdbqt(sdf_fn=sdf_fn, pdbqt_fn=pdbqt_fn)
            self.dtb.ligand_mol_to_box(mol, box_txt_fn=box_txt_fn)

        @delayed_return_exception
        def _par_f_ligand_smi(smi: str, pdbqt_fn: Path):
            self.dtb.ligand_smi_to_pdbqt(smi, pdbqt_fn=pdbqt_fn)

        @delayed_return_exception
        def _par_f_ligand_mol(mol: Mol, pdbqt_fn: Path):
            self.dtb.ligand_mol_to_pdbqt(mol, pdbqt_fn=pdbqt_fn)

        # ligand
        if ligand[0] == self.LigandTypeEnum.SDF_FN:
            sdf_fn = ligand[1]
            h = file_hash(sdf_fn, _hash())
            uid = h.hexdigest()
            if uid not in self.ligand_mem:
                pdbqt_fn = use_path(file_path=self.work_dn / "ligand" / f"{uid}.ligand.pdbqt")
                box_txt_fn = use_path(file_path=self.work_dn / "box" / f"{uid}.dock.box.txt")
                self.ligand_mem[uid] = pdbqt_fn
                self.box_mem[uid] = box_txt_fn
                self.delayed_jobs.append(dict(t='single', f=_par_f_ligand_sdf(sdf_fn=sdf_fn, pdbqt_fn=pdbqt_fn, box_txt_fn=box_txt_fn)))
        elif ligand[0] == self.LigandTypeEnum.SMI_STR:
            smi = ligand[1]
            h = _hash(smi.encode())
            uid = h.hexdigest()
            if uid not in self.ligand_mem:
                pdbqt_fn = use_path(file_path=self.work_dn / "ligand" / f"{uid}.ligand.pdbqt")
                self.ligand_mem[uid] = pdbqt_fn
                self.delayed_jobs.append(dict(t='single', f=_par_f_ligand_smi(smi=smi, pdbqt_fn=pdbqt_fn)))
        elif ligand[0] == self.LigandTypeEnum.MOL_OBJ:
            mol = ligand[1]
            h = _hash(Chem.MolToMolBlock(mol).encode())
            uid = h.hexdigest()
            if uid not in self.ligand_mem:
                pdbqt_fn = use_path(file_path=self.work_dn / "ligand" / f"{uid}.ligand.pdbqt")
                self.ligand_mem[uid] = pdbqt_fn
                self.delayed_jobs.append(dict(t='single', f=_par_f_ligand_mol(mol=mol, pdbqt_fn=pdbqt_fn)))
        else:
            raise ValueError(f"Unknown ligand type: {ligand[0]}")
        complex_h.update(h.digest())
        complex.ligand = self.ligand_mem[uid].relative_to(self.work_dn).as_posix()

        # box
        if box[0] == self.BoxTypeEnum.SDF_FN:
            sdf_fn = box[1]
            h = file_hash(sdf_fn, _hash())
            uid = h.hexdigest()
            if uid not in self.box_mem:
                pdbqt_fn = use_path(file_path=self.work_dn / "ligand" / f"{uid}.ligand.pdbqt")
                box_txt_fn = use_path(file_path=self.work_dn / "box" / f"{uid}.dock.box.txt")
                self.ligand_mem[uid] = pdbqt_fn
                self.box_mem[uid] = box_txt_fn
                self.delayed_jobs.append(dict(t='single', f=_par_f_ligand_sdf(sdf_fn=sdf_fn, pdbqt_fn=pdbqt_fn, box_txt_fn=box_txt_fn)))
        else:
            raise ValueError(f"Unknown box type: {box[0]}")
        complex_h.update(h.digest())
        complex.box = self.box_mem[uid].relative_to(self.work_dn).as_posix()

        complex.id = complex_h.hexdigest()
        complex_id_set = set(c.id for c in self.complex_lst)
        if complex.id not in complex_id_set:
            self.complex_lst.append(complex)

        return complex
    
    def run_jobs(self, n_jobs=1, desc="Run jobs"):
        if len(self.delayed_jobs) > 0:

            single_jobs = [item['f'] for item in self.delayed_jobs if item['t'] == 'single']
            print(f'{desc} (single) -> {len(single_jobs)}')
            pbar = tqdm(single_jobs, desc=f'{desc} (single)')
            single_result_lst = [function(*args, **kwargs) for function, args, kwargs in pbar]
            err_lst = [result for result in single_result_lst if isinstance(result, Exception)]
            logging.warning(f"{len(err_lst)}/{len(single_result_lst)} tasks throw exceptions")

            parallel_jobs = [item['f'] for item in self.delayed_jobs if item['t'] == 'parallel']
            print(f'{desc} (parallel) -> {len(parallel_jobs)}')
            pbar = tqdm(total=len(parallel_jobs), desc=f'{desc} (parallel)')
            joblib_cb = TqdmTrackerJoblibCallback(pbar)
            with hook_joblib(joblib_cb):
                parallel_result_lst = Parallel(n_jobs=n_jobs)(job for job in parallel_jobs)
            pbar.close()
            err_lst = [result for result in parallel_result_lst if isinstance(result, Exception)]
            logging.warning(f"{len(err_lst)}/{len(parallel_result_lst)} tasks throw exceptions")

            # result_lst is out of order
            result_lst = single_result_lst + parallel_result_lst
            err_lst = [result for result in result_lst if isinstance(result, Exception)]
            logging.warning(f"{len(err_lst)}/{len(result_lst)} tasks throw exceptions")
            self.delayed_jobs.clear()
        else:
            warnings.warn("No jobs being delayed.")

    def pack_tarball(self, fn: Path):
        if fn.suffix == '.auto':
            fn = fn.with_suffix(self._apu.prefer_suffix)
        self._apu.auto_pack(self.work_dn, fn)
        return fn
