# DeepBlock

Reactive Fragment-based Ligand Generation (Version 2)

## Evaluation Script

## Script for docking

## Design Art

prepare -> serialize -> compute


https://github.com/QVina/qvina

> Important note:
>
> - If you Don't know the Docking site, then QuickVina-W is your choice with ability to dock WIDE search box.
> - However if you know the target search box, we recommend that you use QuickVina 2.

https://autodock-vina.readthedocs.io/en/latest/faq.html

> How big should the search space be?
>
> As small as possible, but not smaller. The smaller the search space, the easier it is for the docking algorithm to explore it. On the other hand, it will not explore ligand and flexible side chain atom positions outside the search space. You should probably avoid search spaces bigger than 30 x 30 x 30 Angstrom, unless you also increase “–exhaustiveness”.

https://projects.volkamerlab.org/teachopencadd/talktorials/T015_protein_ligand_docking.html

https://chem-workflows.com/articles/2021/09/18/1-molecular-docking/

https://ccsb.scripps.edu/adfr/how-to-create-a-pdbqt-for-my-receptor/


AutoDock Suite -> AutoDockFR

https://github.com/bioconda/bioconda-recipes/tree/master/recipes/autodock
https://anaconda.org/bioconda/autodock
https://autodocksuite.scripps.edu/
https://ccsb.scripps.edu/adfr


Final:

ligand:
meeko, openbabel

receptor:
adfr
prepare_receptor

# PyMol

https://pymolwiki.org/index.php/Windows_Install