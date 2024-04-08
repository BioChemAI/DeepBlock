from setuptools import setup, find_packages
import importlib

def test_import_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False

install_requires = [
    'torch', 'numpy', 'scikit-learn', 'matplotlib', 'pandas', 'tpot', 'xgboost', 'fair-esm',
    'rdkit', 'biopython', 'meeko', 'openbabel',
    'easydict', 'orjson', 'ormsgpack', 'pyyaml', 'bidict',
    'joblib', 'wandb', 'tqdm', 'halo', 'pexpect', 'psutil', 'requests'
]

# Some modules, even if installed by conda, still do not enter the index of pip.
# Or its dependencies are still required to be installed by pip.
# We will dynamically import to test them.
module_to_be_test_import = ['rdkit', 'openbabel', 'tpot', 'xgboost']
satisfied_requires = set(filter(test_import_module, module_to_be_test_import))
install_requires = [x for x in install_requires if x not in satisfied_requires]

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='deepblock',
    version='0.2.0',
    description='Reactive Fragment-based Ligand Generation (Version 2)',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kaihao Zhang',
    author_email='khzhang@stu.xjtu.edu.cn',
    url="https://github.com/BioChemAI/DeepBlock",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.7',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
