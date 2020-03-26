from setuptools import setup, find_packages

setup(name='circuitq',
      version='0.1',
      description='Automated design of superconducting qubits',
      url='https://git.uibk.ac.at/c7051093/circuitq.git',
      # packages=['circuitq']
      packages=find_packages(),
      # install_requires= [
      #     'qutip', 'numpy', 'networkx', 'sympy', 'scipy'
      #                   ],
      )
