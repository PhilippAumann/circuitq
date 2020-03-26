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
graph.add_edge(0,1, element = 'L');
# my.visualize_circuit(graph, 'circuitq_test_graph')

# =============================================================================
# Create circuit instance with the symbolic classical Hamiltonian
# =============================================================================
circuit = cq.CircuitQ(graph)
print(circuit.h_parameters)

# =============================================================================
# Diagonalization data
# =============================================================================
phi_ex_list = np.linspace(-2*np.pi,2*np.pi,30)
cq_eigv_list = []
for phi_ex in phi_ex_list:
    parameters_list = [False, False, False, phi_ex]
    h_num = circuit.get_numerical_hamiltonian(100,parameter_values=parameters_list)
    eigv, eigs = circuit.get_eigensystem()
    cq_eigv_list.append(eigv[:10])

