import os, sys
from setuptools import setup, find_packages

setup(name='circuitq',
      version='1.1.0',
      description='Automated design of superconducting qubits',
      url='https://github.com/PhilippAumann/circuitq',
      author='CircuitQ Team',
      author_email='philipp.aumann@uibk.ac.at',
      license='MIT',
      # packages=['circuitq']
      packages=find_packages(),
      install_requires= ['numpy', 'networkx', 'sympy', 'scipy'],
      )

local_package_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_package_dir)