import circuitq as cq

import numpy as np
import networkx as nx
import scqubits as sc
import matplotlib.pyplot as plt
import scipy.linalg as lg

# =============================================================================
# Define Circuit as Graph
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J');

# =============================================================================
# Numerical Implementation
# =============================================================================
circuit = cq.CircuitQ(graph)
dim = 200
h_num = circuit.get_numerical_hamiltonian(dim)
eigv, eigs = circuit.get_eigensystem()
print(circuit.get_T1_quasiparticles())



