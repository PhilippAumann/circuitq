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
# Shifted (by one elemantary charge) state in flux basis by operator
# =============================================================================
circuit = cq.CircuitQ(graph)
dim = 400
h_num = circuit.get_numerical_hamiltonian(dim)
eigv, eigs = circuit.get_eigensystem()
circuit.get_T1_quasiparticles()
shifted_state_operator = circuit.shifted_state


shifted_state_array = shifted_state_operator.toarray()
norm = np.sqrt(np.sum([abs(element) ** 2 for element in shifted_state_operator.data]))
shifted_state_operator_normalized = shifted_state_array/norm

# =============================================================================
# Shifted (by one elemantary charge) state in flux basis by offset
# =============================================================================
circuit = cq.CircuitQ(graph, offset_nodes=[1])
parameters_list = [False, False, 1*circuit.e]
circuit.get_numerical_hamiltonian(dim, parameter_values=parameters_list)
circuit.get_eigensystem()
circuit.transform_charge_to_flux()
shifted_state_offset = circuit.estates_in_phi_basis[1]

plt.style.use('default')
plt.plot(abs(shifted_state_offset)**2, label='offset')
plt.plot(abs(shifted_state_operator.data)**2, label='operator')
plt.plot(abs(shifted_state_operator_normalized)**2, label='operator normalized')
plt.legend()
plt.show()

# =============================================================================
# Size of matrix element as a function of dimension
# =============================================================================
# mtx_elements = []
# dims = np.arange(10,100,1)
# # dims=[50]
# for dim in dims:
#
#     print(dim, '*********')
#     circuit = cq.CircuitQ(graph)
#     # =============================================================================
#     # Numerical Hamiltonian
#     # =============================================================================
#     dim = int(dim)
#     h_num = circuit.get_numerical_hamiltonian(dim)
#     eigv, eigs = circuit.get_eigensystem()
#     # =============================================================================
#     # T1
#     # =============================================================================
#     circuit.get_T1_quasiparticles()
#     mtx_elements.append(circuit.mtx_element)
#
# plt.plot(dims, mtx_elements)
# plt.show()



