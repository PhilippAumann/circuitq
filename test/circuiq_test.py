import circuitq as cq
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as spa

# =============================================================================
# Define Circuit as Graph
# =============================================================================
graph = nx.MultiGraph()
graph.add_edge(0,1, element = 'C')
graph.add_edge(0,1, element = 'J')
graph.add_edge(0,1, element = 'L')
# graph.add_edge(1,2, element = 'C')
# graph.add_edge(1,2, element = 'J')
# graph.add_edge(0,2, element = 'C')
# graph.add_edge(0,2, element = 'J')

# =============================================================================
# Numerical Implementation
# =============================================================================
circuit = cq.CircuitQ(graph)
phi_ext = 0#np.pi*circuit.phi_0
circuit.get_numerical_hamiltonian(400,
          parameter_values=[False, False, False, phi_ext])
eigv, eigs = circuit.get_eigensystem()
print(circuit.get_T1_quasiparticles())
print(circuit.get_T1_charge())
print(circuit.get_T1_flux())

# =============================================================================
# current operator as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(2,20,9)
# current_operators = []
# plt.figure(figsize=(10,10))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     current_operators.append(circuit.current_operators_num[0])
#     plt.subplot(3,3,int(n+1))
#     plt.title("dim = " + str(circuit.n_dim))
#     plt.imshow(np.imag(circuit.current_operators_num[0].toarray()))
#     plt.ylim(0,100)
#     plt.xlim(0,100)
#     plt.colorbar()
# plt.show()

# =============================================================================
# Plot diagonals of current operator in flux basis as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(20,100,9)
# current_operators = []
# plt.figure(figsize=(10,10))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph, force_flux_nodes=[0,1,2])
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     current_operators.append(circuit.current_operators_num[0])
#     plt.subplot(3,3,int(n+1))
#     plt.xlim(0,50)
#     plt.title("dim = " + str(circuit.n_dim))
#     plt.plot(np.real(circuit.current_operators_num[0].toarray().diagonal()))
# plt.show()

# =============================================================================
# Plot prefactor of flux T1 as a function of dimension
# =============================================================================
# dim_list = np.linspace(30,60,7)
# prefactor_list = []
# plt.style.use('default')
# plt.figure(figsize=(8,5))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     circuit.get_eigensystem()
#     circuit.get_T1_quasiparticles()
#     circuit.get_T1_charge()
#     circuit.get_T1_flux()
#     prefactor_list.append(circuit.S_phi)
# plt.plot(dim_list, prefactor_list)
# plt.show()

# =============================================================================
# matrix element (flux T1) as a function of dimension
# =============================================================================
# dim_list = np.linspace(30,100,20)
# matrix_elements = []
# plt.figure(figsize=(8,5))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     circuit.get_eigensystem()
#     circuit._qubit_states_energy()
#     ground_state = spa.csr_matrix(circuit.ground_state)
#     excited_state = spa.csr_matrix(circuit.estates[:, circuit.excited_level])
#     matrix_elements.append(abs((excited_state.conjugate()*circuit.current_operators_num[0]*
#                                              ground_state.transpose()).data[0])**2)
# plt.title("Matrix element (flux T1)")
# plt.plot(dim_list, matrix_elements)
# plt.xlabel("Dimension")
# plt.show()

# =============================================================================
# fidelity ground - excited state as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(30,100,20)
# fidelity_list = []
# plt.figure(figsize=(8,5))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     circuit.get_eigensystem()
#     circuit._qubit_states_energy()
#     fidelity_list.append(abs(np.dot(circuit.ground_state.transpose().conjugate(),
#                                     circuit.estates[:, circuit.excited_level]))**2)
# plt.title("Fidelity ground - exited state")
# plt.plot(dim_list, fidelity_list)
# plt.xlabel("Dimension")
# plt.show()

# =============================================================================
# states as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(10,80,9)
# plt.figure(figsize=(10,10))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     circuit.get_eigensystem()
#     # circuit._qubit_states_energy()
#     circuit.get_T1_flux()
#     plt.subplot(3,3,int(n+1))
#     plt.title("dim = " + str(int(circuit.n_dim)))
#     plt.plot(abs(circuit.ground_state)**2, '-', label="ground")
#     plt.plot(abs(circuit.excited_state_np) ** 2, '-', label="excited")
#     # plt.plot(abs(circuit.estates[:, circuit.excited_level])**2, '-', label="excited")
#     # plt.plot(abs(circuit.estates[:, circuit.excited_level+1]) ** 2, '--' ,
#     #          label="2nd excited")
#     plt.xlim((int(circuit.n_dim**2/2 - 1*circuit.n_dim),
#              int(circuit.n_dim**2/2 + 1*circuit.n_dim)))
# plt.legend()
# # plt.savefig("/Users/philipp/Dropbox (Personal)/"
# #             "CircuitDesign/figures/excited_states_flux.pdf")
# plt.show()

# =============================================================================
# current operators applied to ground state as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(10,100,9)
# plt.figure(figsize=(30,30))
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     circuit.get_eigensystem()
#     circuit._qubit_states_energy()
#     ground_state = spa.csr_matrix(circuit.ground_state)
#     transformed_state = (circuit.current_operators_num[0]*
#                          ground_state.transpose()).toarray()
#     plt.subplot(3,3,int(n+1))
#     plt.title("dim = " + str(int(circuit.n_dim)))
#     plt.plot(abs(transformed_state)**2)
#     plt.xlim((int(circuit.n_dim**2/2 - 5*circuit.n_dim),
#              int(circuit.n_dim**2/2 + 5*circuit.n_dim)))
# plt.show()

# =============================================================================
# eigenvalues as a function of dimension
# =============================================================================
# circuit = cq.CircuitQ(graph)
# dim_list = np.linspace(30,60,7)
# evals_list = []
# for n, dim in enumerate(dim_list):
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     eigv, eigs = circuit.get_eigensystem()
#     evals_list.append(eigv[:3])
# plt.plot(dim_list, evals_list)
# plt.show()

# =============================================================================
# T1 times as a function of dimension with runtime
# =============================================================================

# dim_list = np.linspace(30,120,15)
# T1_qp,T1_c,T1_f,time_list = [],[],[],[]
# for dim in dim_list:
#     start_time = time.time()
#     circuit = cq.CircuitQ(graph)
#     h_num = circuit.get_numerical_hamiltonian(int(dim))
#     eigv, eigs = circuit.get_eigensystem()
#     T1_qp.append(circuit.get_T1_quasiparticles())
#     T1_c.append(circuit.get_T1_charge())
#     # T1_f.append(circuit.get_T1_flux())
#     time_list.append(time.time()-start_time)
#     print(circuit.excited_subspace)
#
# plt.style.use('default')
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(dim_list, T1_qp, label="quasiparticle")
# ax1.plot(dim_list, T1_c, label="charge")
# # ax1.plot(dim_list, T1_f, label="flux")
# ax2.plot(dim_list,time_list, 'r-', label="runtime")
# ax1.set_xlabel("Dimension")
# ax1.set_ylabel(r'$T_1$ in s')
# ax2.set_ylabel('Runtime in s')
# ax1.legend()
# ax2.legend()
# # plt.savefig("/Users/philipp/Dropbox (Personal)/"
# #             "CircuitDesign/figures/T1_dim_time.pdf")
# plt.show()

# =============================================================================
# T1 times as a function of dimension
# =============================================================================

dim_list = np.linspace(30,400,10)
T1_qp,T1_c,T1_f = [],[],[]
for dim in dim_list:
    print("\nDimension =", int(dim))
    circuit = cq.CircuitQ(graph)
    # EJ = 2.5 * circuit.c_v["E"]
    # alpha = 0.7
    # C = circuit.c_v["C"]
    # phi_ext = np.pi * circuit.phi_0
    h_num = circuit.get_numerical_hamiltonian(dim,
            parameter_values=[False, False, False, phi_ext])
    eigv, eigs = circuit.get_eigensystem()
    T1_qp.append(circuit.get_T1_quasiparticles())
    T1_c.append(circuit.get_T1_charge())
    T1_f.append(circuit.get_T1_flux())

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dim_list, T1_qp, label="quasiparticle")
ax1.plot(dim_list, T1_c, label="charge")
ax2.plot(dim_list, T1_f, 'r-', label="flux")
ax1.set_xlabel("Dimension")
ax1.set_ylabel(r'$T_1$ in s')
ax2.set_ylabel('$T_1$ in s')
ax1.legend()
ax2.legend()
# plt.savefig("/Users/philipp/Dropbox (Personal)/"
#             "CircuitDesign/figures/T1_dim_time.pdf")
plt.show()