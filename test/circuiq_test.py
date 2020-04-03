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
graph.add_edge(1,2, element = 'L')
graph.add_edge(2,0, element = 'J')

# =============================================================================
# Create circuit instance with the symbolic classical Hamiltonian
# =============================================================================
circuit = cq.CircuitQ(graph)

# =============================================================================
# Numerical Hamiltonian
# =============================================================================
h_num = circuit.get_numerical_hamiltonian(50)
eigv, eigs = circuit.get_eigensystem(50)





