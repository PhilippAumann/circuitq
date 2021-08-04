import os
import circuitq as cq
import numpy as np
import networkx as nx

# =============================================================================
# LC Circuit
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'L')
circuit = cq.CircuitQ(graph)
circuit.get_numerical_hamiltonian(200)
eigv_lc, eigs_lc = circuit.get_eigensystem(30)

# =============================================================================
# Transmon
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J')
circuit = cq.CircuitQ(graph, offset_nodes=[1])
h_num = circuit.get_numerical_hamiltonian(401, grid_length=np.pi*circuit.phi_0)
eigv_transmon, eigs_transmon = circuit.get_eigensystem(30)

# =============================================================================
# Fluxonium
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J')
graph.add_edge(0,1, element = 'L')
circuit = cq.CircuitQ(graph)
EJ = circuit.c_v["E"]/2
phi_ext = np.pi*circuit.phi_0
h_num = circuit.get_numerical_hamiltonian(401,
                                parameter_values=[False, EJ, False, phi_ext ])
eigv_fluxonium, eigs_fluxonium = circuit.get_eigensystem(30)

# =============================================================================
# Flux Qubit
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J')
graph.add_edge(1,2, element = 'C')
graph.add_edge(1,2, element = 'J')
graph.add_edge(0,2, element = 'C')
graph.add_edge(0,2, element = 'J')
circuit = cq.CircuitQ(graph)
dim = 50
EJ = 2.5*circuit.c_v["E"]
alpha = 0.7
C = circuit.c_v["C"]
phi_ext = np.pi*circuit.phi_0
h_num = circuit.get_numerical_hamiltonian(dim,
                        parameter_values=[C,C,alpha*C,EJ,EJ,alpha*EJ,phi_ext])
eigv_fluxqubit, eigs_fluxqubit = circuit.get_eigensystem(30)

# =============================================================================
# Save data
# =============================================================================
main_dir = os.path.abspath('..')
data_path = os.path.join(main_dir, 'test/unittest_data.npy')
np.save(data_path,
        np.array([[eigv_lc, eigv_transmon, eigv_fluxonium, eigv_fluxqubit],
                  [eigs_lc, eigs_transmon, eigs_fluxonium, eigs_fluxqubit]], dtype=object))
