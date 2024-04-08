from itertools import chain, islice
from numbers import Number
import re
import io
import base64
import copy
import warnings
import random
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, fields

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import numpy as np

import cairosvg
import svgutils.transform as sg
import svgutils.compose as sc
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Geometry import Point3D

@dataclass
class DrawInfo:
    atom_coords: Dict[int, np.ndarray] = None
    size_int: Tuple[int, int] = None
    size: Tuple[str, str] = None

class svgstr(str):
    def set_draw_info(self, draw_info: DrawInfo=DrawInfo(), old_svg: Union[str, 'TSvgOrSvgstr']=None, obtain_size: bool=True):
        if hasattr(old_svg, 'draw_info'):
            for field in fields(DrawInfo):
                if getattr(draw_info, field.name) is None:
                    setattr(draw_info, field.name, getattr(old_svg.draw_info, field.name))
        if obtain_size:
            root = ET.fromstring(self)
            draw_info.size = root.get('width'), root.get('height')
            draw_info.size_unit = re.search(r'[^\d\s]+$', draw_info.size[0]).group()
            draw_info.size_num = tuple(float(x.rstrip(draw_info.size_unit)) for x in draw_info.size)
        self.draw_info = draw_info
        return self

TSvgOrSvgstr = Union[str, svgstr]

@contextmanager
def namespace_context(namespace_map: Dict={'http://www.w3.org/2000/svg': ''}):
    old_namespace_map = dict(ET._namespace_map)
    ET._namespace_map.update(namespace_map)
    try:
        yield
    finally:
        ET._namespace_map = old_namespace_map

def init_mpl(font_path: Path, use_latex: bool=False, color_cycle: str='light'):
    # https://github.com/garrettj403/SciencePlots
    try:
        import scienceplots
        style_lst = ['science', 'nature', color_cycle]
        if not use_latex:
            style_lst.append('no-latex')
        plt.style.use(style_lst)
    except ImportError as err:
        warnings.warn(f"Please run 'conda install -c conda-forge scienceplots' "
                      f"to enable SciencePlots, now disabled.")
    default_color_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = [mcolors.to_rgb(color['color']) for color in default_color_cycle]
    
    # https://zhuanlan.zhihu.com/p/501395717
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": font_name,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    mpl.rcParams.update({
        "figure.figsize": (3, 1.5),
        "figure.dpi": 150,
    })

    # The font above needs to be installed in the system
    mpl.rcParams.update({
        'svg.fonttype': 'none',
    })

    return default_colors, font_prop

def test_mpl():
    """Generate Matplotlib figure containing Chinese, English, and Latex.
    """
    def model(x, p):
        return x ** (2 * p + 1) / (1 + x ** (2 * p))
    x = np.linspace(0.75, 1.25, 201)
    fig, ax = plt.subplots()
    for p in [10, 15, 20, 30, 50, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.set_title("中英文 Latex Matplotlib 测试")
    ax.legend(title=f'负随机数：{-random.random():.3f}')
    ax.autoscale(tight=True)
    ax.set(xlabel='电压 Voltage (mV)', ylabel='电流 Current ($\\mu$A)')
    return fig, ax

light_colors_c = ['77AADD', 'EE8866', 'EEDD88', 'FFAABB', '99DDFF', '44BB99', 'BBCC33', 'AAAA00', 'DDDDDD']
light_colors = [mcolors.hex2color(f'#{c}') for c in light_colors_c]

get_size_of_elem = lambda elem: tuple(float(elem.get(k).strip('px')) for k in ('width', 'height'))
get_box_of_elem = lambda elem: tuple(map(float, elem.get('viewBox').split()))
pure_num = lambda x: str(x).rstrip('0').rstrip('.')
set_box_of_elem = lambda elem, box: elem.set('viewBox', ' '.join(pure_num(x) for x in box))
set_size_of_elem = lambda elem, size: tuple(elem.set(k, f'{pure_num(size[i])}px') for i, k in enumerate(('width', 'height')))

@namespace_context()
def svg_rm_bg(svg: TSvgOrSvgstr) -> svgstr:
    root = ET.fromstring(svg)
    root_size = get_size_of_elem(root)
    for rect in root.findall(".//ns0:rect", namespaces={'ns0': 'http://www.w3.org/2000/svg'}):
        if get_size_of_elem(rect) == root_size:
            root.remove(rect)
    out_svg = ET.tostring(root).decode()
    return svgstr(out_svg).set_draw_info(old_svg=svg)

def svg_overlap(*svgs: List[TSvgOrSvgstr], size_from: TSvgOrSvgstr=None) -> svgstr:
    sfig_lst = [sg.fromstring(_svg) for _svg in svgs]
    if size_from is None:
        size_from = svgs[0]
    # https://github.com/btel/svg_utils/issues/89
    out_sfig = sg.SVGFigure(*(sc.Unit(x) for x in sg.fromstring(size_from).get_size()))
    for _sfig in sfig_lst:
        # _sfig.set_size(sfig_lst[0].get_size())
        out_sfig.append(_sfig.getroot())
    out_svg = out_sfig.to_str().decode()
    return svgstr(out_svg).set_draw_info(old_svg=svgs[0])

@namespace_context()
def svg_crop(svg: TSvgOrSvgstr, 
             left: float=0, top: float=0, 
             right: float=None, bottom: float=None) -> svgstr:
    root = ET.fromstring(svg)
    root_size = get_size_of_elem(root)
    root_box = get_box_of_elem(root)
    assert root_box[2:] == root_size

    if right is None:
        right = root_box[2]
    elif right < 0:
        right = root_box[2] + right

    if bottom is None:
        bottom = root_box[3]
    elif bottom < 0:
        bottom = root_box[3] + bottom

    new_size = (right - left, bottom - top)
    new_box = (root_box[0] + left, root_box[1] + top, *new_size)

    set_box_of_elem(root, new_box)
    set_size_of_elem(root, new_size)
    
    out_svg = ET.tostring(root).decode()
    return svgstr(out_svg).set_draw_info(old_svg=svg)

def svg_to_pdf(svg: str) -> bytes:
    return cairosvg.svg2pdf(svg.encode())

def save_mpl_img(fig: plt.Figure, format: str="svg", pad_inches: float=0) -> Union[bytes, str]:
    with io.BytesIO() as f:
        fig.savefig(f, format=format, transparent=True, pad_inches=pad_inches)
        img = f.getvalue()
    if format == "svg":
        return svgstr(img.decode()).set_draw_info()
    else:
        return img

def get_mpl_canvas(figsize: Tuple[int, int], axis_off: bool=True) -> Tuple[plt.Figure, plt.Axes, int]:
    dpi = 72
    fig = plt.figure(dpi=dpi)
    fig.set_size_inches((figsize[0] / dpi, figsize[1] / dpi), forward=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis((0, figsize[0], 0, figsize[1]))
    ax.invert_yaxis()
    if axis_off:
        ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

def test_mpl_canvas_overlap(svg: TSvgOrSvgstr):
    svg = svgstr(svg).set_draw_info()

    fig, ax = get_mpl_canvas(svg.draw_info.size_num)
    a, b = svg.draw_info.size_num

    ax.plot([0, a], [0, b])
    ax.plot([0, a], [b, 0])
    ax.scatter([0, a, 0, a], [0, b, b, 0])

    svg2 = save_mpl_img(fig, format="svg")
    svg3 = svg_overlap(svg, svg2)

    return svg3

def get_perpendicular_bisector(a: Tuple[float, float], b: Tuple[float, float], 
                               dash_length: float = None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # By ChatGPT
    midpoint = [(a[0]+b[0])/2, (a[1]+b[1])/2]
    unit_vector = np.array([(b[0]-a[0])/np.linalg.norm(b-a), 
                            (b[1]-a[1])/np.linalg.norm(b-a)])
    perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])
    if dash_length is None:
        dash_length = np.linalg.norm(b-a)
    start_point = midpoint - 0.5*dash_length*perpendicular_vector
    end_point = midpoint + 0.5*dash_length*perpendicular_vector
    return start_point, end_point


def angle_to_unit_vector(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    x = np.cos(angle_radians)
    y = np.sin(angle_radians)
    return np.array([x, y])

def random_iterator(start: Number, stop: Number, seed: Any):
    r = random.Random(seed)
    while True:
        yield r.uniform(start, stop)

def get_open_point(cloud: np.ndarray, center: np.ndarray, 
                   cr: Number, sr: Number, stop: int=1<<10, 
                   width: Number=None, height: Number=None,
                   reverse_x: bool=False, reverse_y: bool=False) -> np.ndarray:
    
    alpha = np.array([-1 if reverse_x else 1, -1 if reverse_y else 1])
    limit = np.array([
        [sr, sr],
        [np.inf if width-sr is None else width-sr, np.inf if height is None else height-sr]
    ])

    predefined_angles = [-30, 30, 150, -150]
    random_angles = random_iterator(-180, 180, center.tobytes())

    angles = chain(predefined_angles, random_angles)
    for angle in islice(angles, stop):
        unit_vector = angle_to_unit_vector(angle)
        target = center + unit_vector * cr * alpha
        distance = np.linalg.norm(target - cloud, axis=1)
        if np.all(distance > sr) and np.all((target < limit[1]) & (limit[0] < target)):
            return target
    raise Exception(f"Maximum attempts reached: {stop}")

def insert_newline(x, n):
    return '\n'.join([x[i:i+n] for i in range(0, len(x), n)])


def formula_to_latex(formula: str):
    formula = re.sub(r'(\d+)', r'_{\1}', formula)
    formula = re.sub(r'([A-Z][a-z]*)', r'\\mathrm{\1}', formula)
    return f"${formula}$"

def mol_to_md(mol, *args, **kwargs) -> str:
    img = Draw.MolToImage(mol, *args, **kwargs)
    with io.BytesIO() as output:
        img.save(output, format='PNG')
        png_bytes = output.getvalue()
    img_base64 = base64.b64encode(png_bytes).decode('utf-8')
    return f"![{Chem.MolToSmiles(mol)}](data:image/png;base64,{img_base64})"


# https://www.rdkit.org/docs/Cookbook.html#rings-aromaticity-and-kekulization
# drawer.drawOptions().addAtomIndices
# Draw molecule with atom index (see RDKitCB_0)
def add_atom_index_to_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def rm_atom_index_to_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def mol_to_svg(mol: Mol, 
               width: int=600, height: int=300, 
               is_add_atom_index_to_map: bool = False, is_add_atom_indices: bool = False,
               is_add_bond_indices: bool = False,
               is_rm_bg: bool = True,
               minFontSize: int=12, annotationFontScale: float=0.8,
               highlightAtoms: List[int]=None, highlightBonds: List[int]=None, 
               highlightAtomColors: Dict[int, Tuple[float, float, float]]=None, 
               highlightBondColors: Dict[int, Tuple[float, float, float]]=None, 
               highlightAtomRadii: Dict[int, float]=None) -> svgstr:
    
    mol = copy.deepcopy(mol)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.atomHighlightsAreCircles = True
    if is_add_atom_index_to_map:
        add_atom_index_to_map(mol)
    if is_add_atom_indices:
        opts.addAtomIndices = True
    if is_add_bond_indices:
        opts.addBondIndices = True

    opts.minFontSize = minFontSize
    opts.annotationFontScale = annotationFontScale

    drawer.DrawMolecule(mol, 
                        highlightAtoms=highlightAtoms, highlightBonds=highlightBonds, 
                        highlightAtomColors=highlightAtomColors, highlightBondColors=highlightBondColors, 
                        highlightAtomRadii=highlightAtomRadii)
    atom_draw_coords = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        coord = drawer.GetDrawCoords(idx)
        atom_draw_coords[idx] = np.array([coord.x, coord.y])

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if is_rm_bg:
        svg = svg_rm_bg(svg)

    return svgstr(svg).set_draw_info(DrawInfo(atom_coords=atom_draw_coords))

def get_mol_atom_pos_arr(mol, dims: int=2) -> np.ndarray:
    assert dims in (2, 3)
    lst = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        assert atom.GetIdx() == len(lst)
        pos = conf.GetAtomPosition(atom_idx)
        lst.append((pos.x, pos.y, pos.z)[:dims])
    arr = np.array(lst)
    return arr

def set_mol_atom_pos_arr(mol, arr: np.ndarray):
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = arr[atom_idx]
        conf.SetAtomPosition(atom_idx, Point3D(*(*pos, 0)[:3]))
    return mol
