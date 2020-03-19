import copy
import networkx as nx
import functionsAndClasses.functions_file as my
import sympy as sp
from sympy.physics.quantum import TensorProduct, tensor_product_simp
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
import qutip as qt
import numpy as np
import scipy.sparse as spa
from scipy.linalg import eigh

class CircuitQ:
    """
    A class corresponding to a superconducting circuit.
    """
    def __init__(self, circuit_graph, e =  1, hbar = 1,
                 ground_nodes = [], print_feedback=False):
        """
        Creates a circuit from a given graph.

        Parameters
        ----------
        circuit_graph: NetworkX Graph
            Graph representation of the circuit
            The edges have to have a specified element with keyword 'element'
            set to either 'C', 'L' or 'J'.
        e: int or float (Default 1)
            Value of the elementary charge
        hbar: int or float (Default 1)
            Value of the reduced Planck constant.
        dimension: int (Default 5)
            Dimension of the Sub-Hilbert spaces
        ground_nodes: list
            The nodes set as ground can be specified in a list.
            They have to be active nodes. If they are not specified, one node
            will be chosen automatically as ground.
        print_feedback: bool (Default False)
            Bool to control printed feedback
        """
        self.circuit_graph = circuit_graph
        self.e = e
        self.hbar = hbar
        #Characteristic Values
        c_v = dict()
        c_v["hbar"] = self.hbar
        c_v["e"] = self.e
        c_v["h"] = 2 * np.pi * c_v["hbar"]
        c_v["t_conversion"] = 2.4188843265 * 10 ** (-17)
        c_v["C"] = c_v["e"] ** 2 / (2 * c_v["h"] * (10 ** 9) * c_v["t_conversion"])
        c_v["omega"] = 2 * np.pi * (10 ** 9) * c_v["t_conversion"]
        c_v["L"] = 1 / ((c_v["omega"] ** 2) * c_v["C"])
        c_v["E_C"] = c_v["e"] ** 2 / (2 * c_v["C"])
        c_v["E"] = 50 * c_v["E_C"]
        self.c_v = c_v
        self.ground_nodes = ground_nodes
        self.print_feedback = print_feedback
        self.phi_0 = self.hbar/(2*self.e)
        self.delete_edges = []
        self.deleted_edges = []
        self.c_matrix_inv = False
        self.inductances = dict()
        self.josephson_energies = dict()
        self.get_classical_hamiltonian_run = False
        self.nodes = []
        self.nodes_wo_ground = []
        self.taylor_variables = []
        self.phi_dict = dict()
        self.phi_periodic = dict()
        self.q_dict = dict()
        self.q_quadratic_dict = dict()
        self.h, self.h_parameters, self.h_sep = self.get_classical_hamiltonian()
        self.coord_list = []
        self.parameter_values = []
        self.input_num_list = []
        self.h_num = False

    def get_classical_hamiltonian(self):
        """
        Returns a Hamiltonian as a Sympy-function for the circuit.

        Parameters
        ----------
        No external parameters

        Returns
        ----------
        h: Sympy Add
            Symbolic Hamiltonian of the circuit
        h_parameters: list
            List of parameters in the Hamiltonian
        """

        # Test Visualisation of the graph
        # my.visualize_circuit(self.circuit_graph, 'test_circuit_start')

        # Check if method has already been run
        if self.get_classical_hamiltonian_run:
            raise Exception("get_classical_hamiltonian is already executed "
                            "with instance initialisation and should not "
                            "run twice.")
        else:
            self.get_classical_hamiltonian_run = True

        # =============================================================================
        # Merge parallel C
        # =============================================================================
        graph_l = copy.deepcopy(self.circuit_graph)
        visited_branches = []
        for e in graph_l.edges(data=True, keys=True):
            element = e[3]['element']
            if 'C' in element:
                u = e[0]
                v = e[1]
                key = e[2]
                if (u,v) in visited_branches:
                    continue
                removed = False
                # Remove parallel capacitive branches
                for edge in graph_l.edges(data=True, keys=True):
                    if ('C' in edge[3]['element'] and
                        ((edge[0] == u and edge[1] == v) or
                         (edge[1] == u and edge[0] == v))
                        and edge[2] != key):
                            self.circuit_graph.remove_edge(edge[0], edge[1], key=edge[2])
                            removed = True
                # Rename element to indicate that the capacitances have been merged
                if removed:
                    for item in self.circuit_graph.edges(data=True, keys=True):
                        if item[0] == u and item[1] == v and 'C' in item[3]['element']:
                            item[3]['element'] = 'C_mp' # mp = "merged parallel"
                visited_branches.append((u,v))
                visited_branches.append((v,u))

        # =============================================================================
        # Find purely capacitive branches
        # =============================================================================
        purely_capacitive_branches = []
        for e in self.circuit_graph.edges(data=True, keys=True):
            u = e[0]
            v = e[1]
            element = e[3]['element']
            if 'C' in element:
                # check element of parallel edges
                parallel_edges = list(nx.edge_boundary(self.circuit_graph, [u, v], [v, u],
                                                   data=True, keys=True))
                parallel_inductance = False
                for p_e in parallel_edges:
                    if 'C' not in p_e[3]['element']:
                        parallel_inductance = True
                if not parallel_inductance:
                    purely_capacitive_branches.append(e)

        # =============================================================================
        # Merge serial C
        # =============================================================================

        # Look for serial C, put purely capicitive edges on delete list
        # and shift investigated node
        def _find_edges_to_delete(i, e_f):
            u_f = i
            while True:
                # Check whether edges connecting u_f are purely capacitive
                connecting_edges = list(self.circuit_graph.edges(u_f,
                                                data=True, keys=True))
                if len(connecting_edges)!=2:
                    break
                for c_e in connecting_edges:
                    if not ((c_e[0] == e_f[0] and c_e[1] == e_f[1]) or
                            (c_e[0] == e_f[1] and c_e[1] == e_f[0])):
                        o_e = c_e
                o_e_p_c = False
                for p_c_e in purely_capacitive_branches:
                    if ((o_e[0] == p_c_e[0] and o_e[1] == p_c_e[1])
                        or (o_e[0] == p_c_e[1] and o_e[1] == p_c_e[0])):
                        o_e_p_c = True
                if not o_e_p_c:
                    break
                # Delete neighbouring purely capacitive branch
                self.delete_edges.append(o_e)
                # Look at next node (shift u_f)
                if o_e[0] == u_f:
                    u_f = o_e[1]
                else:
                    u_f = o_e[0]
                if u_f == e_f[0] or u_f == e_f[1]:
                    raise Exception("Circle of capacitances")
                e_f = o_e
            return u_f

        # Look for serial C and merge them (using previous function)
        def _discover_and_delete(i_0, e_f):
            i = _find_edges_to_delete(i_0, e_f)
            if len(self.delete_edges) > 0:
                for d_e in self.delete_edges:
                    self.circuit_graph.remove_edge(d_e[0], d_e[1], key=d_e[2])
                    self.deleted_edges.append(d_e)
                self.delete_edges = []
                e_f[3]['element'] = 'C_ms' # mp = "merged serial"
                self.circuit_graph = nx.contracted_nodes(self.circuit_graph, i, i_0)
                graph_loop = copy.deepcopy(self.circuit_graph)
                for n_l in graph_loop:
                    connecting_edges = list(graph_loop.edges(n_l))
                    if len(connecting_edges) == 0:
                        self.circuit_graph.remove_node(n_l)

        # Execute the above functions for the purely capacitive branches
        for e in purely_capacitive_branches:
            if e in self.deleted_edges:
                continue
            u_0 = e[0]
            v_0 = e[1]
            _discover_and_delete(u_0, e)
            _discover_and_delete(v_0, e)

        # =============================================================================
        # Introduce parasitic capacitances
        # =============================================================================
        graph_l = copy.deepcopy(self.circuit_graph)
        visited_branches = []
        for e in graph_l.edges(data=True, keys=True):
            # Check if edge has parallel capacitance
            u = e[0]
            v = e[1]
            if (u, v) in visited_branches:
                continue
            if 'C' in e[3]['element']:
                visited_branches.append((u, v))
                visited_branches.append((v, u))
                continue
            parallel_edges = nx.edge_boundary(self.circuit_graph, [u,v], [v,u],
                                              data=True, keys=True)
            parallel_c = False
            for p_e in parallel_edges:
                if 'C' in p_e[3]['element']:
                    parallel_c = True
                    break
            # If there is no parallel capacitance, add a parsitic capacitance
            if not parallel_c:
                self.circuit_graph.add_edge(u, v, element='Cp')
            visited_branches.append((u, v))
            visited_branches.append((v, u))

        # =============================================================================
        # Hide all purely capacative edges (open loop - 0 loop flux)
        # =============================================================================
        reduced_graph = copy.deepcopy(self.circuit_graph)
        for e in self.circuit_graph.edges(data=True, keys=True):
            u = e[0]
            v = e[1]
            key = e[2]
            parallel_edges = list(nx.edge_boundary(self.circuit_graph, [u, v], [v, u],
                                                   data=True, keys=True))
            if len(parallel_edges) < 2:
                reduced_graph.remove_edge(u,v,key=key)

        # =============================================================================
        # Find reduced connected subgraphs
        # =============================================================================
        red_subgraphs = [reduced_graph.subgraph(c_c).copy() for
                         c_c in nx.connected_components(reduced_graph)]

        # =============================================================================
        # Define reduced capactive sub-network(s)
        # =============================================================================
        c_graphs = []
        for sg in red_subgraphs:
            graph_l = copy.deepcopy(sg)
            c_graph = copy.deepcopy(sg)
            for e in graph_l.edges(data=True, keys=True):
                u = e[0]
                v = e[1]
                key = e[2]
                if 'C' not in e[3]['element']:
                    c_graph.remove_edge(u,v,key=key)
            c_graphs.append(c_graph)

        # =============================================================================
        # Find spanning tree(s) of reduced capactive sub-network(s)
        # =============================================================================
        spanning_trees = []
        for n, c_graph in enumerate(c_graphs):
            spanning_tree = nx.minimum_spanning_edges(c_graph, keys=True, data=True)
            spanning_trees.append(spanning_tree)
        spanning_trees_edges = [list(s_t) for s_t in spanning_trees]

        # =============================================================================
        # Define an active node as ground if no ground nodes are given.
        # Otherwise: Check that given ground nodes are active.
        # =============================================================================
        self.nodes = list(self.circuit_graph.nodes())
        for node in self.nodes:
            active_node = False
            incoming_edges = self.circuit_graph.edges(node, keys=True, data=True)
            for e in incoming_edges:
                if 'C' not in e[3]['element']: # Active node (There are C anyways)
                    active_node = True
                    if len(self.ground_nodes) == 0:
                            self.ground_nodes.append(0)
                            break
            if not active_node and node in self.ground_nodes:
                raise Exception("Specified ground node " + str(node) +
                                " is not an active node.")

        # =============================================================================
        # Define flux variables
        # =============================================================================
        for node in self.nodes:
            if node in self.ground_nodes:
                self.phi_dict[node] = 0
            else:
                phi = sp.symbols('\Phi_{' + str(node) + '}')
                self.phi_dict[node] = phi
                self.phi_periodic[node] = True

        # =============================================================================
        # Define potential energy
        # =============================================================================
        pot_energy = 0
        loop_fluxes = dict()
        used_c = []
        nbr_loop_fluxes = dict()

        for n_sg, sg in enumerate(red_subgraphs):
            for e in sg.edges(data=True, keys=True):
                element = e[3]['element']
                if 'C' not in element:
                    # Check whether the parallel C is in spanning tree and define
                    # the variable according to that
                    u = e[0]
                    v = e[1]
                    parallel_edges = nx.edge_boundary(self.circuit_graph,
                                                      [u, v], [v, u], data=True, keys=True)
                    for p_e in parallel_edges:
                        if 'C' in p_e[3]['element']:
                            parallel_c = p_e
                            break
                    if parallel_c in spanning_trees_edges[n_sg] and parallel_c not in used_c:
                        var = self.phi_dict[v] - self.phi_dict[u]
                        # Store capacitances that have been used s.t. parallel L/J have loop flux
                        used_c.append(parallel_c)
                    else:
                        if (str(u) + str(v) not in nbr_loop_fluxes.keys()
                            or str(v) + str(u) not in nbr_loop_fluxes.keys()):
                            nbr_l_f = 0
                        else:
                            nbr_l_f = nbr_loop_fluxes[str(u) + str(v)] + 1
                        nbr_loop_fluxes[str(u) + str(v)] = nbr_l_f
                        nbr_loop_fluxes[str(v) + str(u)] = nbr_l_f
                        loop_flux = sp.symbols(r'\tilde{\Phi}_{' + str(u) + str(v) + str(nbr_l_f) + '}')
                        loop_fluxes[str(u) + str(v) + str(nbr_l_f)] = loop_flux
                        loop_fluxes[str(v) + str(u) + str(nbr_l_f)] = loop_flux
                        loop_flux_var = 0
                        for n in range(nbr_l_f+1):
                            loop_flux_var += loop_fluxes[str(u) + str(v) + str(n)]
                        var = self.phi_dict[v] - self.phi_dict[u] + loop_flux_var
                    # Add terms for junctions or inductances to the potential energy
                    if 'L' in element:
                        if str(u) + str(v) not in self.inductances.keys():
                            self.inductances[str(u) + str(v)] = []
                        number_ind = len(self.inductances[str(u) + str(v)])
                        inductance = sp.symbols('L_{' + str(u) + str(v) + str(number_ind) +'}')
                        self.inductances[str(u) + str(v)].append(inductance)
                        pot_energy += var**2/(2*inductance)
                        self.phi_periodic[u] = False
                        self.phi_periodic[v] = False
                    if 'J' in element:
                        if str(u) + str(v) not in self.josephson_energies.keys():
                            self.josephson_energies[str(u) + str(v)] = []
                        number_j = len(self.josephson_energies[str(u) + str(v)])
                        josesphson_energy = sp.symbols('E_{J' + str(u) + str(v) + str(number_j) + '}')
                        self.josephson_energies[str(u) + str(v)].append(josesphson_energy)
                        pot_energy -= josesphson_energy * sp.cos(var/self.phi_0)
                        self.taylor_variables.append(var)


        # =============================================================================
        # Define C matrix and q vector
        # =============================================================================
        nbr_nodes = len(self.nodes)
        capacitances = dict()
        c_matrix = sp.zeros(nbr_nodes, nbr_nodes)
        # Define non-reduced capactive sub-network
        graph_l = copy.deepcopy(self.circuit_graph)
        c_full_graph = copy.deepcopy(self.circuit_graph)
        for e in graph_l.edges(data=True, keys=True):
            u = e[0]
            v = e[1]
            key = e[2]
            if 'C' not in e[3]['element']:
                c_full_graph.remove_edge(u,v,key=key)
        # Define off-diagonal elements according to the connectivity
        for n, u in enumerate(self.nodes):
            for k, v in enumerate(self.nodes):
                edge = list(nx.edge_boundary(c_full_graph, [u], [v], data=True, keys=True))
                nbr_edges = len(edge)
                if nbr_edges == 0:
                    continue
                elif nbr_edges != 1:
                    raise Exception("More than one capacity is connecting two nodes.")
                else:
                    if str(v) + str(u) in list(capacitances.keys()):
                        capacitance = capacitances[str(v) + str(u)]
                    else:
                        if 'Cp' in edge[0][3]['element']:
                            capacitance = sp.symbols('Cp_{' + str(u) + str(v) + '}')
                        else:
                            capacitance = sp.symbols('C_{' + str(u) + str(v) + '}')
                        capacitances[str(u) + str(v)] = capacitance
                    c_matrix[n,k] = - capacitance
        # Define diagonal elements as the sum of the row entries
        for n in range(nbr_nodes):
            row_sum = 0
            for k in range(nbr_nodes):
                if k!=n:
                    row_sum += c_matrix[n,k]
            c_matrix[n,n] = - row_sum
        # Define q vector
        q_vec_list = []
        self.q_dict = dict()
        for node in self.nodes:
            q = sp.symbols('q_{' + str(node) + '}')
            self.q_dict[node] = q
            q_vec_list.append(q)
        q_vec = sp.Matrix(q_vec_list)
        # Delete rows and columns corresponding to the ground nodes
        nodes_l = copy.deepcopy(self.nodes)
        for g_n in self.ground_nodes:
            ground_idx = nodes_l.index(g_n)
            c_matrix.row_del(ground_idx)
            c_matrix.col_del(ground_idx)
            q_vec.row_del(ground_idx)
            nodes_l.remove(g_n)

        self.nodes_wo_ground = nodes_l
        self.c_matrix_inv = c_matrix.inv()

        # =============================================================================
        # Set parasitic capacitances to zero
        # =============================================================================
        subs_list = []
        for cap in capacitances.values():
            if 'Cp' in str(cap):
                subs_list.append((cap,0))
        self.c_matrix_inv = self.c_matrix_inv.subs(subs_list)

        # =============================================================================
        # Define classical Hamiltonian
        # =============================================================================
        h = 0.5*(q_vec.transpose()*self.c_matrix_inv*q_vec)[0] + pot_energy

        # =============================================================================
        # Define Hamiltonian for numerical implementation with
        # seperated quadratic charges
        # =============================================================================
        c_matrix_inv_sep = copy.deepcopy(self.c_matrix_inv)
        h_kin_sep = 0
        for n, node_l in enumerate(self.nodes_wo_ground):
            c_matrix_inv_sep[n,n] = 0
            q_q = sp.symbols('q^{q}_{' + str(node_l) + '}')
            self.q_quadratic_dict[node_l] = q_q
            h_kin_sep += 0.5*(q_q * self.c_matrix_inv[n,n])
        h_kin_sep += 0.5*(q_vec.transpose()*c_matrix_inv_sep*q_vec)[0]
        h_sep = h_kin_sep + pot_energy

        # =============================================================================
        # Get list of parameter in the Hamiltonian
        # =============================================================================
        h_parameters = list(sorted(h.free_symbols, key=str))
        h_parameters_loop = copy.deepcopy(h_parameters)
        for element in h_parameters_loop:
            if (str(element).startswith('\Phi_') or
                str(element).startswith('q_')):
                h_parameters.remove(element)
        if self.print_feedback:
            print("The parameters of the circuit are " + str(h_parameters))

        return h, h_parameters, h_sep


    def get_numerical_hamiltonian(self, n_dim, parameter_values = None):
        """
        Creates a numerical representation of the Hamiltonian using the
        finite difference method.

        Parameters
        ----------
        n_dim: int
            Dimension of numerical matrix of all subsystems
        parameter_values: list
            Numerical values of system parameters (corresponds to self.h_parameters)

        Returns
        ----------
        h_num: numpy array
            Numeric Hamiltonian of the circuit given as a matrix in array form
        """
        # =============================================================================
        # Define matrix functions
        # =============================================================================
        # Central derivative matrix
        def der_mtx(coord_list, periodic=True):
            dim = len(coord_list)
            delta = abs(coord_list[1] - coord_list[0])
            m = np.zeros((dim, dim))
            for n in range(dim):
                if n + 1 <= dim - 1:
                    m[n, n + 1] = 1
                if n - 1 >= 0:
                    m[n, n - 1] = -1
            if periodic:
                m[0, dim - 1] = -1
                m[dim - 1, 0] = 1
            m = m / (2*delta)

            return spa.csr_matrix(m)

        # Second derivative matrix
        def scnd_der_mtx(coord_list, periodic=True):
            dim = len(coord_list)
            delta = abs(coord_list[1] - coord_list[0])
            m = np.zeros((dim, dim))
            for n in range(dim):
                m[n, n] = -2
                if n + 1 <= dim - 1:
                    m[n, n + 1] = 1
                if n - 1 >= 0:
                    m[n, n - 1] = 1
            if periodic:
                m[0, dim - 1] = 1
                m[dim - 1, 0] = 1
            m = m / (delta ** 2)

            return spa.csr_matrix(m)

        # Phi matrix
        def phi_mtx(coord_list):
            dim = len(coord_list)
            m = np.zeros((dim, dim))
            for n, item in enumerate(coord_list):
                m[n, n] = item

            return spa.csr_matrix(m)

        # cos function
        def mtx_cos(m):
            m_dia = m.diagonal()
            return spa.diags(np.cos(m_dia), format='csr')

        # =============================================================================
        # Define numerical matrices
        # =============================================================================
        phi_matrices, q_matrices, q_quadratic_matrices = [], [], []
        phi_list, q_list, q_quadratic_list = [], [], [] # without ground, input for lambdify
        nbr_subsystems = len(self.nodes) - len(self.ground_nodes)
        self.coord_list = np.arange(-4*np.pi, 4*np.pi, 8*np.pi/n_dim)
        mtx_id_list = [np.identity(n_dim) for n in range(nbr_subsystems)]
        n_mtx_list = 0
        for n, phi in self.phi_dict.items():
            if phi==0:
                continue
            periodic_bool = self.phi_periodic[n]
            for var_type in ['phi', 'q', 'q_quadratic']:
                mtx_list = copy.deepcopy(mtx_id_list)
                if var_type=='phi':
                    mtx_list[n_mtx_list] = phi_mtx(self.coord_list)
                elif var_type=='q':
                    mtx_list[n_mtx_list] = -1j*self.hbar*der_mtx(self.coord_list,
                                                                 periodic=periodic_bool)
                elif var_type=='q_quadratic':
                    mtx_list[n_mtx_list] = -1*(self.hbar**2)*scnd_der_mtx(self.coord_list,
                                                                          periodic=periodic_bool)
                if nbr_subsystems==1:
                    mtx_num = mtx_list[0]
                else:
                    mtx_num = spa.kron(mtx_list[0],mtx_list[1])
                    for i in range(2, nbr_subsystems):
                        mtx_num = spa.kron(mtx_num, mtx_list[i])
                if var_type=='phi':
                    phi_matrices.append(mtx_num)
                    phi_list.append(phi)
                elif var_type=='q':
                    q_matrices.append(mtx_num)
                    q_list.append(self.q_dict[n])
                elif var_type=='q_quadratic':
                    q_quadratic_matrices.append(mtx_num)
                    q_quadratic_list.append(self.q_quadratic_dict[n])
            n_mtx_list += 1

        # =============================================================================
        # Define default parameter values if not given
        # =============================================================================
        if parameter_values is None:
            parameter_values = [False] * len(self.h_parameters)
        parameter_values_l = copy.deepcopy(parameter_values)
        for n, parameter in enumerate(parameter_values_l):
            if parameter is False:
                for key, value in self.c_v.items():
                    if key in str(self.h_parameters[n])\
                            and key != "E_C":
                        parameter_values[n] = value
                        break
                if 'tilde' in str(self.h_parameters[n]):
                    parameter_values[n] = 0
        if any(p is False for p in parameter_values):
            raise Exception("Parameter type might have not been recognized.")
        self.parameter_values = parameter_values

        # =============================================================================
        # Define numerical Hamiltonian via lambdify
        # =============================================================================
        input_list = q_list + q_quadratic_list + phi_list + self.h_parameters
        h_num_lambda = sp.lambdify(input_list, self.h_sep, modules=[{'cos': mtx_cos}, 'numpy'])
        self.input_num_list = q_matrices + q_quadratic_matrices + phi_matrices + self.parameter_values
        self.h_num = h_num_lambda(*self.input_num_list)

        return self.h_num

    def get_eigensystem(self, n_eig = 30):
        """
        Calculates eigenvectors and eigenstates of the numerical Hamiltonian.

        Parameters
        ----------
        n_eig: int
            Number of returned eigenvalues.

        Returns
        ----------
        evals: array
            Array of eigenvalues
        estates: array
            Array of eigenstates
        """

        evals, estates = spa.linalg.eigsh(self.h_num, k=n_eig, which='SA')
        # evals, estates = np.linalg.eigh((self.h_num).toarray())
        idx_sort = np.argsort(evals)
        evals = evals[idx_sort]
        estates = estates[:, idx_sort]

        return evals, estates



