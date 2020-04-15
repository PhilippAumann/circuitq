import circuitq as cq

import numpy as np
import networkx as nx
import scqubits as sc
import matplotlib.pyplot as plt

# =============================================================================
# Define Circuit as Graph
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J')

# =============================================================================
# Create circuit instance with the symbolic classical Hamiltonian
# =============================================================================
circuit = cq.CircuitQ(graph, offset_nodes=[1])
print(circuit.h_parameters)
# =============================================================================
# Numerical Hamiltonian
# =============================================================================
h_num = circuit.get_numerical_hamiltonian(10,
                                          parameter_values=[False, False, 2*circuit.c_v['e']])
eigv, eigs = circuit.get_eigensystem(5)





