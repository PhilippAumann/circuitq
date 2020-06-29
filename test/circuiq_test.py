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
# graph.add_edge(1,2, element = 'C')
# graph.add_edge(1,2, element = 'J')
# graph.add_edge(0,2, element = 'C')
# graph.add_edge(0,2, element = 'J')
# graph.add_edge(1,2, element = 'C')
# graph.add_edge(2,3, element = 'L')
# graph.add_edge(0,3, element = 'C')

# =============================================================================
# Create circuit instance with the symbolic classical Hamiltonian
# =============================================================================
circuit = cq.CircuitQ(graph)
print(circuit.h_parameters)
# =============================================================================
# Numerical Hamiltonian
# =============================================================================
dim = 10
# EJ = 0.1*circuit.c_v["E"] #2.5*circuit.c_v["E"]
# alpha = 0.7
# C = circuit.c_v["C"]
# phi_ext = 0.5*np.pi*circuit.phi_0 #np.pi*circuit.phi_0
#
# # Parenthesis begin -----------------------------------------------------------
# # =============================================================================
# # SC Qubit
# # =============================================================================
# EC = circuit.c_v["E_C"]
# fluxqubit = sc.FluxQubit(EJ1 = EJ, EJ2 = EJ, EJ3 = alpha*EJ,
#                          ECJ1 = EC, ECJ2 = EC, ECJ3 = EC/alpha,
#                          ECg1 = 1e25, ECg2 = 1e25, ng1 = 0, ng2 = 0,
#                          flux = phi_ext/(circuit.phi_0*2*np.pi), ncut = int(dim/2))
# esys = fluxqubit.eigensys(evals_count=30)
# # Parenthesis end -------------------------------------------------------------

h_num = circuit.get_numerical_hamiltonian(dim),
#                            parameter_values=[C,C,alpha*C,EJ,EJ,alpha*EJ,phi_ext])
cutoff = 5
eigv, eigs = circuit.get_eigensystem(cutoff)
circuit.transform_charge_to_flux()


# =============================================================================
# Comparison Plot
# =============================================================================
# plt.figure(figsize=(7,5))
# plt.plot(np.arange(cutoff), eigv[:cutoff], 'rv', label="CircuitQ")
# plt.plot(np.arange(cutoff), esys[0][:cutoff], 'g^', label="SC Qubit")
# plt.legend()
# plt.xlabel("Eigenvalue No.")
# plt.ylabel("Energy")
# for n in range(25):
#     plt.axhline(esys[0][n], lw=0.5)
# plt.ticklabel_format(style='scientific', scilimits=(0, 0))
# plt.show()

print(circuit.get_T1_quasiparticles())




