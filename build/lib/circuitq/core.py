import copy
import warnings
import networkx as nx
import math
import sympy as sp
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spa_linalg

class CircuitQ:
    """
    A class corresponding to a superconducting circuit.
    """
    def __init__(self, circuit_graph, ground_nodes = None,
                 offset_nodes = None, force_flux_nodes=None,
                 print_feedback=False):
        """
        Creates a circuit from a given graph.

        Parameters
        ----------
        circuit_graph: NetworkX Graph
            Graph representation of the circuit
            The edges have to have a specified element with keyword 'element'
            set to either 'C', 'L' or 'J'.
        ground_nodes: list
            The nodes set as ground can be specified in a list.
            They have to be active nodes. If they are not specified, one node
            will be chosen automatically as ground.
        offset_nodes: list
            The nodes specified in this list will have an offset charge
            which values can be specified via parameter_values in
            self.get_numerical_hamiltonian()
        print_feedback: bool (Default False)
            Bool to control printed feedback
        """
        self.circuit_graph = circuit_graph
        self.e = 1.602176634e-19
        self.hbar = 1.054571817e-34
        #Characteristic Values
        c_v = {}
        c_v["C"] =1e-13 #F
        c_v["L"] = 1e-7 #H
        c_v["E_C"] = self.e ** 2 / (2 * c_v["C"])
        c_v["E"] = 50 * c_v["E_C"]
        self.c_v = c_v
        # The definition of phi_0 is not consistent in the literature (variant: h/(2e) )
        self.phi_0 = self.hbar/(2*self.e)
        self.ground_nodes = ground_nodes if ground_nodes is not None else []
        self.offset_nodes = offset_nodes if offset_nodes is not None else []
        self.force_flux_nodes = force_flux_nodes if force_flux_nodes is not None else []
        self.print_feedback = print_feedback
        self.delete_edges = []
        self.deleted_edges = []
        self.c_matrix_inv = None
        self.inductances = {}
        self.inductances_full_index = {}
        self.edge_flux_inductance = {}
        self.edge_flux_inductance_num = {}
        self.loop_fluxes_in_cos_arg = {}
        self.get_classical_hamiltonian_run = False
        self.spanning_trees_edges = []
        self.nodes = []
        self.capacitances = {}
        self.nodes_wo_ground = []
        self.phi_dict = {}
        self.q_dict = {}
        self.q_quadratic_dict = {}
        self.q_matrices_num_dict = {}
        self.periodic = {}
        self.josephson_energies_global_dict = {}
        self.charge_basis_nodes = []
        self.loop_fluxes = {}
        self.cos_charge_dict = {}
        self.sin_charge_dict = {}
        self.sin_phi_half_operators_dict = {}
        self.offset_dict = {}
        self.current_operators_imp = []
        self.current_operators_all_imp = []
        self.sin_phi_half_operators = []
        self.v = None
        self.pot_energy_imp = None
        self.h, self.h_parameters, self.h_imp = self.get_classical_hamiltonian()
        self.n_dim = None
        self.grid_length = None
        self.n_cutoff = None
        self.flux_list = []
        self.charge_list = []
        self.mtx_id_list = []
        self.subspace_pos = {}
        self.charge_subspaces = []
        self.loop_fluxes_num = {}
        self.parameter_values = []
        self.parameter_values_dict = {}
        self.input_num_list = []
        self.phi_num_dict = {}
        self.potential = None
        self.h_num = None
        self.v_num_list = []
        # self.capacitances_sum_dict = {}
        # self.capacitances_sum_dict_num = {}
        self.current_operators_num = []
        self.current_operators_all_num = []
        self.sin_phi_half_operators_num = []
        self.n_eig = None
        self.evals = None
        self.estates = None
        self.anharmonicity = None
        self.excited_level = None
        self.excited_subspace = []
        self.degenerated = False
        self.T_mtx = None
        self.estates_in_phi_basis = []
        self.ground_state = None
        self.omega_q = None
        self.T1_quasiparticle = None
        self.T1_dielectric_loss = None
        self.T1_inductive_loss = None

    def get_classical_hamiltonian(self):
        """
        Returns a Hamiltonian as a Sympy-function for the circuit.

        Parameters
        ----------
        No external parameters.

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
                if u_f in (e_f[0], e_f[1]):
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
            e_already_deleted = False
            for d_e in self.deleted_edges:
                if ((d_e[0]==e[0] and d_e[1]==e[1] and d_e[2]==e[2]) or
                    (d_e[1]==e[0] and d_e[0]==e[1] and d_e[2]==e[2])):
                    e_already_deleted = True
            if not e_already_deleted:
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
        self.spanning_trees_edges = spanning_trees_edges

        # =============================================================================
        # Define a node with only one neighbour as a ground node if the neighbour is
        # not a ground node yet.
        # Define an active node as ground if no ground nodes are given.
        # Otherwise: Check that given ground nodes are active.
        # =============================================================================
        self.nodes = list(self.circuit_graph.nodes())
        active_nodes = []
        for node in self.nodes:
            neighbor_nodes = list(self.circuit_graph.neighbors(node))
            if len(neighbor_nodes) == 1:
                neighbor = neighbor_nodes[0]
                if (node not in self.ground_nodes and
                    neighbor not in self.ground_nodes):
                    self.ground_nodes.append(node)
            incoming_edges = self.circuit_graph.edges(node, keys=True, data=True)
            for e in incoming_edges:
                if 'C' not in e[3]['element']: # Active node (There are C anyways)
                    active_nodes.append(node)
            if node in self.ground_nodes and node not in active_nodes:
                raise Exception("Specified ground node " + str(node) +
                                " is not an active node.")
        if len(self.ground_nodes) == 0:
            self.ground_nodes.append(active_nodes[0])

        # =============================================================================
        # Define flux variables
        # =============================================================================
        for node in self.nodes:
            if node in self.ground_nodes:
                self.phi_dict[node] = 0
            else:
                phi = sp.symbols(r'\Phi_{' + str(node) + '}')
                self.phi_dict[node] = phi

        # =============================================================================
        # Define periodicity of nodes to choose between flux and charge basis
        # =============================================================================
        for n in self.nodes:
            periodicity = True
            connecting_edges = self.circuit_graph.edges(n,data=True, keys=True)
            for c_e in connecting_edges:
                if c_e[3]['element'] == 'L':
                    periodicity = False
                    break
            if n in self.force_flux_nodes:
                periodicity = False
            self.periodic[n] = periodicity

        # Only periodic if neighbouring nodes which are connected
        # via a Josephson junction are periodic as well
        periodic_dict = copy.deepcopy(self.periodic)
        changed_dict = True
        while changed_dict:
            periodic_loop = copy.deepcopy(periodic_dict)
            loop_break = False
            for key, value in periodic_loop.items():
                if value is True:
                    connecting_edges = self.circuit_graph.edges(key, data=True, keys=True)
                    for edge in connecting_edges:
                        neighbour = edge[1]
                        if neighbour == key:
                            neighbour = edge[0]
                        if (periodic_dict[neighbour] is False
                            and edge[3]['element'] == 'J'
                            and neighbour not in self.ground_nodes):
                            periodic_dict[key] = False
                            loop_break = True
                            break
                if loop_break:
                    break
            if periodic_loop == periodic_dict:
                changed_dict = False
        # Set all ground nodes to periodic, to prevent the ground node from affecting
        # the charge basis implementation
        for node in self.ground_nodes:
            periodic_dict[node] = True

        self.periodic = periodic_dict

        # =============================================================================
        # Define potential energy
        # =============================================================================
        pot_energy = 0
        pot_energy_imp = 0 #The potential energy for implementation w/ different basis
        used_c = []
        josephson_energies = {}
        nbr_loop_fluxes = {}
        phi_0_symbol = sp.symbols(r'\Phi_{o}')

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
                    loop_flux_var_indices = []
                    for p_e in parallel_edges:
                        if 'C' in p_e[3]['element']:
                            parallel_c = p_e
                            break
                    closure_branch = False
                    # Nodes of parallel_c could be permuted
                    parallel_c_permuted = (parallel_c[1], parallel_c[0], parallel_c[2], parallel_c[3])
                    if (parallel_c in spanning_trees_edges[n_sg] and
                        parallel_c not in used_c) \
                            or (parallel_c_permuted in spanning_trees_edges[n_sg]
                                and parallel_c_permuted not in used_c):
                        var = self.phi_dict[v] - self.phi_dict[u]
                        # Store capacitances that have been used s.t. parallel L/J have loop flux
                        used_c.append(parallel_c)
                    else:
                        closure_branch = True
                        if (str(u) + str(v) not in nbr_loop_fluxes
                            or str(v) + str(u) not in nbr_loop_fluxes):
                            nbr_l_f = 0
                        else:
                            nbr_l_f = nbr_loop_fluxes[str(u) + str(v)] + 1
                        nbr_loop_fluxes[str(u) + str(v)] = nbr_l_f
                        nbr_loop_fluxes[str(v) + str(u)] = nbr_l_f
                        loop_flux = sp.symbols(r'\tilde{\Phi}_{' + str(u) + str(v) + str(nbr_l_f) + '}')
                        self.loop_fluxes[(u, v, nbr_l_f)] = loop_flux
                        loop_flux_var = 0
                        for n in range(nbr_l_f+1):
                            loop_flux_var += self.loop_fluxes[(u, v, n)]
                            loop_flux_var_indices.append((u, v, n))
                        var = self.phi_dict[v] - self.phi_dict[u] + loop_flux_var
                    # Add terms for junctions or inductances to the potential energy
                    if 'L' in element:
                        if str(u) + str(v) not in self.inductances:
                            self.inductances[str(u) + str(v)] = []
                        number_ind = len(self.inductances[str(u) + str(v)])
                        inductance = sp.symbols('L_{' + str(u) + str(v) + str(number_ind) +'}')
                        self.inductances[str(u) + str(v)].append(inductance)
                        self.inductances_full_index[(u, v, number_ind)] = inductance
                        self.edge_flux_inductance[(u, v, number_ind)] = var
                        pot_energy += var**2/(2*inductance)
                        pot_energy_imp += var**2/(2*inductance)
                        # Current operator
                        current_operator = var/inductance
                        self.current_operators_all_imp.append(current_operator)
                        if closure_branch:
                            self.current_operators_imp.append(current_operator)
                    if 'J' in element:
                        if ((u,v)) not in josephson_energies:
                            josephson_energies[(u,v)] = []
                        number_j = len(josephson_energies[(u,v)])
                        josephson_energy = sp.symbols('E_{J' + str(u) + str(v) + str(number_j) + '}')
                        josephson_energies[(u,v)].append(josephson_energy)
                        self.josephson_energies_global_dict[(u, v, number_j)] = josephson_energy
                        pot_energy -= josephson_energy * sp.cos(var/phi_0_symbol)
                        if len(loop_flux_var_indices) > 0:
                            self.loop_fluxes_in_cos_arg[(u,v,number_j)] = loop_flux_var_indices
                        if self.periodic[u] is True and self.periodic[v] is True:
                            # Will be implemented in charge basis
                            # Cosine operator for charge basis implementation
                            cos_charge = sp.symbols('cos_{' + str(u) + str(v) + str(number_j) + '}')
                            if u not in self.charge_basis_nodes:
                                self.charge_basis_nodes.append(u)
                            if v not in self.charge_basis_nodes:
                                self.charge_basis_nodes.append(v)
                            self.cos_charge_dict[(u,v,number_j)] = cos_charge
                            pot_energy_imp -= josephson_energy * cos_charge
                            # Sine phi half operator for charge basis implementation
                            sin_phi_half_charge = sp.symbols('sin_phi_half_{' + str(u) + str(v) +
                                                             str(number_j) + '}')
                            self.sin_phi_half_operators_dict[(u, v, number_j)] = sin_phi_half_charge
                            self.sin_phi_half_operators.append(sin_phi_half_charge)
                            # Current operator for charge basis implementation
                            sin_charge = sp.symbols('sin_{' + str(u) + str(v) + str(number_j) + '}')
                            self.sin_charge_dict[(u, v, number_j)] = sin_charge
                            current_operator = (2 * self.e * josephson_energy / self.hbar
                                                * sin_charge)
                            self.current_operators_all_imp.append(current_operator)
                            if closure_branch:
                                self.current_operators_imp.append(current_operator)
                        else:
                            # Will be implemented in flux basis
                            pot_energy_imp -= josephson_energy * sp.cos(var/self.phi_0)
                            sin_phi_half_flux = sp.sin(var/(2*self.phi_0))
                            self.sin_phi_half_operators.append(sin_phi_half_flux)
                            self.sin_phi_half_operators_dict[(u, v, number_j)] = sin_phi_half_flux
                            # Current operator for flux basis implementation
                            current_operator = (2*self.e*josephson_energy/self.hbar
                                                * sp.sin(var/self.phi_0))
                            self.current_operators_all_imp.append(current_operator)
                            if closure_branch:
                                self.current_operators_imp.append(current_operator)

        # =============================================================================
        # Define C matrix and q vector
        # =============================================================================
        nbr_nodes = len(self.nodes)
        capacitances = {}
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
                    if (v,u) in list(capacitances):
                        capacitance = capacitances[(v, u)]
                    else:
                        if 'Cp' in edge[0][3]['element']:
                            capacitance = sp.symbols('Cp_{' + str(u) + str(v) + '}')
                        else:
                            capacitance = sp.symbols('C_{' + str(u) + str(v) + '}')
                        capacitances[(u, v)] = capacitance
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
        q_vec_list_without_offset = []
        self.q_dict = {}
        for node in self.nodes:
            q = sp.symbols('q_{' + str(node) + '}')
            self.q_dict[node] = q
            q_vec_list_without_offset.append(q)
            if node in self.offset_nodes:
                o = sp.symbols(r'\tilde{q}_{' + str(node) + '}')
                self.offset_dict[node] = o
                q_vec_list.append(q + o)
            else:
                q_vec_list.append(q)
        q_vec = sp.Matrix(q_vec_list)
        q_vec_without_offset = sp.Matrix(q_vec_list_without_offset)
        # Delete rows and columns corresponding to the ground nodes
        nodes_l = copy.deepcopy(self.nodes)
        for g_n in self.ground_nodes:
            ground_idx = nodes_l.index(g_n)
            c_matrix.row_del(ground_idx)
            c_matrix.col_del(ground_idx)
            q_vec.row_del(ground_idx)
            q_vec_without_offset.row_del(ground_idx)
            nodes_l.remove(g_n)

        self.capacitances = capacitances
        self.nodes_wo_ground = nodes_l
        self.c_matrix_inv = c_matrix.inv()

        # =============================================================================
        # Define classical Hamiltonian
        # =============================================================================
        h = 0.5*(q_vec.transpose()*self.c_matrix_inv*q_vec)[0] + pot_energy

        # =============================================================================
        # Define Hamiltonian for numerical implementation with
        # seperated quadratic charges.
        # (The offset is not explicitly given here,
        # but is implemented in the numerical matrices in get_numerical_hamiltonian)
        # =============================================================================
        c_matrix_inv_sep = copy.deepcopy(self.c_matrix_inv)
        h_kin_sep = 0
        for n, node_l in enumerate(self.nodes_wo_ground):
            c_matrix_inv_sep[n,n] = 0
            q_q = sp.symbols('q^{q}_{' + str(node_l) + '}')
            self.q_quadratic_dict[node_l] = q_q
            h_kin_sep += 0.5*(q_q * self.c_matrix_inv[n,n])
        h_kin_sep += 0.5*(q_vec_without_offset.transpose()*c_matrix_inv_sep*
                          q_vec_without_offset)[0]
        h_imp = h_kin_sep + pot_energy_imp
        self.pot_energy_imp = pot_energy_imp

        # =============================================================================
        # Get list of parameter in the Hamiltonian
        # =============================================================================
        h_parameters = list(sorted(h.free_symbols, key=str))
        h_parameters_loop = copy.deepcopy(h_parameters)
        for element in h_parameters_loop:
            if (str(element).startswith(r'\Phi_') or
                str(element).startswith('q_')):
                h_parameters.remove(element)
        if self.print_feedback:
            print("The parameters of the circuit are " + str(h_parameters))

        # =============================================================================
        # Define voltage operator
        # =============================================================================
        self.v = q_vec_without_offset.transpose()*self.c_matrix_inv

        return h, h_parameters, h_imp

    @staticmethod
    def _kron_product(mtx_list):
        """
        Create the kronecker product of list of matrices. Used within CircuitQ to
        construct the total operators out of a list of operators acting on the subspaces.

        Parameters
        ----------
        mtx_list: list
            List of sparse matrices, which should be combined by the kronecker product.

        Returns
        ----------
        mtx_num: scipy sparse matrix
            Total operator constructed by the kronecker product.

        """
        nbr_subsystems = len(mtx_list)
        if nbr_subsystems == 1:
            mtx_num = mtx_list[0]
        else:
            mtx_num = spa.kron(mtx_list[0], mtx_list[1])
            for i in range(2, nbr_subsystems):
                mtx_num = spa.kron(mtx_num, mtx_list[i])
        return mtx_num

    def get_numerical_hamiltonian(self, n_dim, grid_length = None,
                                  parameter_values = None, default_zero = True):
        """
        Creates a numerical representation of the Hamiltonian using the
        finite difference method.

        Parameters
        ----------
        n_dim: int
            Dimension of numerical matrix of each subsystem. If an even number is given,
            the value is set to the next odd number. This is done to match the
            dimension of matrices between the charge and flux basis.
        grid_length: float (Default 4*np.pi*phi_0)
            The coordinate grid is taken from -grid_length to +grid_length in n_dim steps.
        parameter_values: list
            Numerical values of system parameters (corresponds to self.h_parameters).
        default_zero: bool (Default True)


        Returns
        ----------
        h_num: numpy array
            Numeric Hamiltonian of the circuit given as a matrix in array form
        """
        # =============================================================================
        # Setting default paramter value for grid_length and set n_dim to odd value
        # =============================================================================
        if grid_length is None:
            grid_length = 4 * np.pi * self.phi_0
        n_dim = int(n_dim)
        if n_dim % 2 == 0:
            n_dim = int(n_dim+1)
        self.n_dim = int(n_dim)
        self.grid_length = grid_length

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

        # sin function
        def mtx_sin(m):
            m_dia = m.diagonal()
            return spa.diags(np.sin(m_dia), format='csr')

        # Charge matrix
        def q_mtx(n_cutoff):
            diagonal = 2*self.e*np.arange(-n_cutoff, n_cutoff+1)
            return spa.diags(diagonal, format='csr')

        # e^{i \Phi} matrix
        def cmplx_exp_phi_mtx(n_cutoff):
            dim = 2*n_cutoff+1
            m = np.zeros((dim, dim))
            for n in range(1,dim):
                m[n,n-1] = 1
            return spa.csr_matrix(m)

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
                        if "Cp" in str(self.h_parameters[n]):
                            parameter_values[n] = value*10**(-4)
                        else:
                            parameter_values[n] = value
                        break
                if ('tilde' in str(self.h_parameters[n])
                    and default_zero is True):
                    parameter_values[n] = 0
                elif r'tilde{\Phi}' in str(self.h_parameters[n]):
                    parameter_values[n] = np.pi*self.phi_0
                elif 'tilde{q}' in str(self.h_parameters[n]):
                    parameter_values[n] = 2*self.e

        if any(p is False for p in parameter_values):
            raise Exception("Parameter type might have not been recognized.")
        self.parameter_values = parameter_values
        # Define parameter value dictionary
        # =============================================================================
        for n, parameter in enumerate(self.h_parameters):
            self.parameter_values_dict[parameter] = self.parameter_values[n]

        # =============================================================================
        # Define numerical matrices
        # =============================================================================
        phi_matrices, q_matrices, q_quadratic_matrices = [], [], []
        cos_charge_matrices = []
        sin_charge_matrices = []
        sin_phi_half_matrices = []
        phi_list, q_list, q_quadratic_list = [], [], [] # without ground, input for lambdify
        cos_charge_list = []
        sin_charge_list = []
        sin_phi_half_list = []
        nbr_subsystems = len(self.nodes_wo_ground)
        self.n_cutoff = int((self.n_dim-1) / 2 )
        self.flux_list = np.linspace(-self.grid_length, self.grid_length, self.n_dim)
        self.charge_list = 2*self.e*np.arange(-self.n_cutoff, self.n_cutoff+1)
        self.mtx_id_list = [spa.identity(self.n_dim) for n in range(nbr_subsystems)]
        n_mtx_list = 0
        # q-matrices in charge and flux basis, phi-matrices (flux basis)
        # =============================================================================
        for n, phi in self.phi_dict.items():
            if phi==0:
                self.phi_num_dict[n] = 0
                continue
            self.subspace_pos[n] = n_mtx_list
            if n in self.charge_basis_nodes:
                self.charge_subspaces.append(n_mtx_list)
            if n in self.offset_nodes:
                parameter_pos = self.h_parameters.index(self.offset_dict[n])
                offset = parameter_values[parameter_pos]
            for var_type in ['phi', 'q', 'q_quadratic']:
                mtx_list = copy.deepcopy(self.mtx_id_list)
                if var_type=='phi':
                    mtx_list[n_mtx_list] = phi_mtx(self.flux_list)
                elif var_type=='q':
                    if n in self.charge_basis_nodes:
                        mtx_list[n_mtx_list] = q_mtx(self.n_cutoff)
                    else:
                        mtx_list[n_mtx_list] = -1j*self.hbar*der_mtx(self.flux_list,
                                                                     periodic=False)
                    if n in self.offset_nodes:
                        mtx_list[n_mtx_list] += offset * spa.identity(self.n_dim)
                elif var_type=='q_quadratic':
                    if n in self.charge_basis_nodes:
                        mtx_list[n_mtx_list] = q_mtx(self.n_cutoff)**2
                        if n in self.offset_nodes:
                            mtx_list[n_mtx_list] = (q_mtx(self.n_cutoff) +
                                                         offset * spa.identity(self.n_dim)) ** 2
                    else:
                        mtx_list[n_mtx_list] = -1*(self.hbar**2)*scnd_der_mtx(self.flux_list,
                                                                            periodic=False)
                        if n in self.offset_nodes:
                            mtx_list[n_mtx_list] += (-2*offset *1j*self.hbar*der_mtx(self.flux_list,
                                                                     periodic=False) +
                                                    offset**2 * spa.identity(self.n_dim) )
                mtx_num = self._kron_product(mtx_list)
                if var_type=='phi':
                    self.phi_num_dict[n] = mtx_num
                    if n not in self.charge_basis_nodes:
                        phi_matrices.append(mtx_num)
                        phi_list.append(phi)
                elif var_type=='q':
                    q_matrices.append(mtx_num)
                    q_list.append(self.q_dict[n])
                    self.q_matrices_num_dict[n] = mtx_num
                elif var_type=='q_quadratic':
                    q_quadratic_matrices.append(mtx_num)
                    q_quadratic_list.append(self.q_quadratic_dict[n])
            n_mtx_list += 1

        # cos-, sin-matrices and sin-phi-half-matrices (charge basis)
        # =============================================================================
        for indices, cos in self.cos_charge_dict.items():
            mtx_list = copy.deepcopy(self.mtx_id_list)
            mtx_list_single_charge = copy.deepcopy(self.mtx_id_list)
            if indices[0] not in self.ground_nodes:
                pos_u = self.subspace_pos[indices[0]]
                mtx_list[pos_u] = cmplx_exp_phi_mtx(self.n_cutoff).getH()
                mtx_list_single_charge[pos_u] = cmplx_exp_phi_mtx(2*self.n_cutoff).getH()
            if indices[1] not in self.ground_nodes:
                pos_v = self.subspace_pos[indices[1]]
                mtx_list[pos_v] = cmplx_exp_phi_mtx(self.n_cutoff)
                mtx_list_single_charge[pos_v] = cmplx_exp_phi_mtx(2*self.n_cutoff)
            mtx_num = self._kron_product(mtx_list)
            mtx_num_single_charge = self._kron_product(mtx_list_single_charge)
            loop_flux = 0
            if indices in self.loop_fluxes_in_cos_arg:
                for loop_flux_index in self.loop_fluxes_in_cos_arg[indices]:
                    l_f = self.loop_fluxes[loop_flux_index]
                    parameter_pos = self.h_parameters.index(l_f)
                    loop_flux += self.parameter_values[parameter_pos]
            mtx_num = np.exp(-1j*loop_flux/self.phi_0) * mtx_num
            mtx_num_single_charge = np.exp(-1j*loop_flux/(2*self.phi_0)) * mtx_num_single_charge
            mtx_num_cos = 0.5*(mtx_num + mtx_num.getH())
            cos_charge_matrices.append(mtx_num_cos)
            cos_charge_list.append(cos)
            mtx_num_sin_phi_half = -0.5j*(mtx_num_single_charge.getH()-mtx_num_single_charge)
            sin_phi_half_matrices.append(mtx_num_sin_phi_half)
            sin_phi_half_list.append(self.sin_phi_half_operators_dict[indices])
            if indices in self.sin_charge_dict:
                mtx_num_sin = 0.5j*(mtx_num.getH()-mtx_num)
                sin_charge_matrices.append(mtx_num_sin)
                sin_charge_list.append(self.sin_charge_dict[indices])

        # numerical matrices for offset flux (flux basis)
        # =============================================================================
        _parameter_values = copy.deepcopy(self.parameter_values)
        for key, value in self.loop_fluxes.items():
            if (key[0] in self.charge_basis_nodes and
                key[1] in self.charge_basis_nodes):
                continue
            node_list = [key[0], key[1]]
            mtx_list = copy.deepcopy(self.mtx_id_list)
            parameter_pos = self.h_parameters.index(value)
            for n in node_list:
                if self.phi_dict[n] == 0:
                    continue
                mtx_list[self.subspace_pos[n]] = (self.parameter_values[parameter_pos]*
                                                    spa.identity(self.n_dim))
                # The offset should only be added once
                break
            mtx_num = self._kron_product(mtx_list)
            self.loop_fluxes_num[key] = mtx_num
            _parameter_values[parameter_pos] = mtx_num

        # =============================================================================
        # If problem is 1D and formulated in phi-basis:
        # Potential as a List
        # =============================================================================
        if nbr_subsystems == 1 and len(self.charge_basis_nodes) == 0:
            potential_lambda = sp.lambdify(phi_list + self.h_parameters,
                                           self.pot_energy_imp)
            potential = [potential_lambda(flux, *self.parameter_values) for flux in
                         self.flux_list]
            self.potential = potential

        # =============================================================================
        # Define numerical Hamiltonian via lambdify
        # =============================================================================
        input_list = q_list + q_quadratic_list + phi_list + cos_charge_list + self.h_parameters
        h_num_lambda = sp.lambdify(input_list, self.h_imp, modules=[{'cos': mtx_cos}, 'numpy'])
        self.input_num_list = q_matrices + q_quadratic_matrices + phi_matrices + \
                              cos_charge_matrices + _parameter_values
        self.h_num = h_num_lambda(*self.input_num_list)

        # =============================================================================
        # Define numerical voltage operator via lambdify
        # =============================================================================
        for element in self.v:
            num_element = sp.lambdify(q_list + self.h_parameters, element, modules = ['numpy'])
            input_num = q_matrices + self.parameter_values
            self.v_num_list.append(num_element(*input_num))

        # =============================================================================
        # Define sum of capacitances connected to a node
        # =============================================================================
        # for node in self.nodes_wo_ground:
        #     capcitances_sum = 0
        #     for key in self.capacitances:
        #         if str(node) in key:
        #             capcitances_sum += self.capacitances[key]
        #     self.capacitances_sum_dict[node] = capcitances_sum
        #
        # for node, element in self.capacitances_sum_dict.items():
        #     num_element = sp.lambdify(self.h_parameters, element)
        #     input_num = self.parameter_values
        #     self.capacitances_sum_dict_num[node] = num_element(*input_num)

        # =============================================================================
        # Define numerical current operator via lambdify
        # =============================================================================
        for element in self.current_operators_imp:
            num_element = sp.lambdify(phi_list + sin_charge_list + self.h_parameters,
                                      element, modules = [{'sin': mtx_sin}, 'numpy'])
            input_num = phi_matrices + sin_charge_matrices + _parameter_values
            self.current_operators_num.append(num_element(*input_num))
        for element in self.current_operators_all_imp:
            num_element = sp.lambdify(phi_list + sin_charge_list + self.h_parameters,
                                      element, modules = [{'sin': mtx_sin}, 'numpy'])
            input_num = phi_matrices + sin_charge_matrices + _parameter_values
            self.current_operators_all_num.append(num_element(*input_num))

        # =============================================================================
        # Define sin-phi-half operator via lambdify
        # =============================================================================
        for element in self.sin_phi_half_operators:
            num_element = sp.lambdify(phi_list + sin_phi_half_list + self.h_parameters,
                                      element, modules = [{'sin': mtx_sin}, 'numpy'])
            input_num = phi_matrices + sin_phi_half_matrices + _parameter_values
            self.sin_phi_half_operators_num.append(num_element(*input_num))

        # =============================================================================
        # Define edge flux operators for linear inductive branches via lambdify
        # =============================================================================
        for indices, element in self.edge_flux_inductance.items():
            num_element = sp.lambdify(phi_list + self.h_parameters, element)
            input_num = phi_matrices + _parameter_values
            self.edge_flux_inductance_num[indices] = num_element(*input_num)

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
        dim_total = self.h_num.shape[0]
        if n_eig > dim_total-2:
            n_eig = dim_total - 2
        self.n_eig = n_eig
        v0 = [0]*dim_total
        v0[0] = 1
        evals, estates = spa_linalg.eigsh(self.h_num, k=self.n_eig, which='SA'#)
                                          ,v0=v0)
        idx_sort = np.argsort(evals)
        self.evals = evals[idx_sort]
        self.estates = estates[:, idx_sort]

        return self.evals, self.estates


    def get_spectrum_anharmonicity(self, nbr_check_levels = 3):
        """
        Calculates the anharmonicity of the eigenspectrum. The method can handle
        degenerated eigenenergies. It considers the transition that is the closest to
        the qubit transition (groundstate-first excited states) and subsequently
        calculates the quotient between these two transitions and returns abs(1-quotient).

        Parameters
        ----------
        nbr_check_levels: int (Default 3)
            Number of levels that should be considered for this analysis. The
            counting includes groundstate and first excited state.

        Returns
        ----------
        anharmonicity: float
            abs(1-quotient) as described above
        """
        quotients = []
        distances_to_1 = []
        excited_level = 1
        # Define first non-degenerate state which is higher then groundstate
        # as excited level
        tolerance = 0.001    #*100 gives tolerance in percentage for comparison of the
                            # states to check for degeneracy
        while math.isclose(self.evals[excited_level], self.evals[0], rel_tol=tolerance):
            excited_level += 1
        if excited_level >1:
            warnings.warn("The ground state seems to be degenerated. "
                          "This has an unfortunate effect on the T1 times.")
            self.degenerated = True
        if excited_level == len(self.evals)-1:
            raise Exception("All states in self.evals are degenerated")
        self.excited_level = excited_level
        self.excited_subspace.append(excited_level)
        current_level = excited_level
        check_level = 2
        for k in range(excited_level + 1, len(self.evals)):
            if math.isclose(self.evals[current_level], self.evals[k], rel_tol=tolerance):
                if current_level == excited_level:
                    self.excited_subspace.append(k)
                    warnings.warn("The excited state seems to be degenerated. "
                                  "This has an unfortunate effect on the T1 times.")
                    self.degenerated = True
                continue
            else:
                current_level = k
                check_level += 1
            for l in range(excited_level,k):
                quotient = abs((self.evals[k]-self.evals[l])/(self.evals[excited_level]-self.evals[0]))
                quotient = min(quotient, 2)
                quotients.append(quotient)
            if check_level >= nbr_check_levels:
                break
        if check_level < nbr_check_levels:
            if self.degenerated:
                warnings.warn("Not enough eigenenergies to check spectrum for nbr_check_levels."
                              "However at least one of the qubit states is degenerated anyway.")
            else:
                raise Exception("Not enough eigenenergies to check spectrum for nbr_check_levels")
        if len(quotients) > 0:
            for quotient in quotients:
                distances_to_1.append(abs(1-quotient))
            self.anharmonicity = min(distances_to_1)

        return self.anharmonicity

    def transform_charge_to_flux(self):
        """
        Transforms the eigenvectors into the flux basis. This is necessary to plot the
        states as a function of flux if the numerical Hamiltonian was (partially)
        implemented in the charge basis.

        Parameters
        ----------
        No external parameters

        Returns
        ----------
        estates_in_phi_basis: array
            An array that contains the eigenstates in the flux basis.
        """
        # =============================================================================
        # Create a dictionary that labels the basis states
        # =============================================================================
        n_subs = len(self.nodes_wo_ground)
        state_indices = n_subs * [0]
        state_indices[-1] = -1
        position = n_subs - 1
        position_shifted = False
        basis_states = {}
        count = 0
        while position >= 0:
            value = state_indices[position] + 1
            if value > self.n_dim - 1:
                for n in range(position, n_subs):
                    state_indices[n] = 0
                position -= 1
                position_shifted = True
            else:
                state_indices[position] = value
                if position_shifted:
                    position = n_subs - 1
                current_indices = copy.deepcopy(state_indices)
                basis_states[count] = current_indices
                count += 1

        # =============================================================================
        # Define the transformation matrix T
        # =============================================================================
        length = len(self.estates[:,0])
        T = np.ones((length, length), dtype=np.complex)
        for i in range(length):
            for j in range(length):
                i_states = basis_states[i]
                j_states = basis_states[j]
                for n, j_state in enumerate(j_states):
                    if n in self.charge_subspaces:
                        T[i,j] *= np.exp(-1j*self.flux_list[i_states[n]]*
                                    self.charge_list[j_state]/self.hbar)
                    else:
                        if i_states[n]!=j_state:
                            T[i,j] = 0
                            break
        T = T/np.sqrt(self.n_dim)
        self.T_mtx = T

        # =============================================================================
        # Transform the states
        # =============================================================================
        transformed_estates = []
        for n in range(self.n_eig):
            transformed_estates.append(np.dot(T,self.estates[:,n]))

        # =============================================================================
        # Normalize the states
        # =============================================================================
        # transformed_estates_loop = copy.deepcopy(transformed_estates)
        # for n, state in enumerate(transformed_estates_loop):
        #     norm = np.sqrt(np.sum([abs(element)**2 for element in state]))
        #     transformed_estates[n] = transformed_estates[n]/norm

        self.estates_in_phi_basis = transformed_estates
        return self.estates_in_phi_basis

    def _qubit_states_energy(self):
        """
        Returns the ground state, the number of the excited level and the qubit energy
        and writes these parameters in self.

        Parameters
        ----------
        No external parameters.

        Returns
        ----------
        ground_state: array
            Ground state as an array
        excited_level: int
            Number of default excited level, which is the first state above ground state.
        omega_q: float
            Qubit energy (difference between excited level and ground state energy.
        """
        ground_state = self.estates[:,0]
        self.get_spectrum_anharmonicity()
        excited_level = self.excited_level
        omega_q = abs(self.evals[excited_level]-self.evals[0])
        self.ground_state = ground_state
        self.omega_q = omega_q
        return self.ground_state, self.excited_level, self.omega_q

    def get_T1_quasiparticles(self, excited_level=None):
        """
        Estimates the T1 contribution due to quasiparticles. See the preprint on arXiv
        for more details about the formulas that have been used for this method.

        Parameters
        ----------
        excited_level: int (Default: Number of first state above the groundstate)
            Number of state, which is considered to be the excited state.

        Returns
        ----------
        T1_quasiparticle: float
            T1 contribution due to quasiparticles.
        """
        # =============================================================================
        # Define transformation from cooper pair charge basis
        # to odd and even single charge basis
        # =============================================================================
        def cooper_to_even(dim):
            dim_single = 2*dim-1
            mtx = np.zeros((dim_single, dim))
            for n in range(dim):
                mtx[2*n, n] = 1
            return spa.csr_matrix(mtx)

        def cooper_to_odd(dim):
            dim_single = 2*dim-1
            mtx = np.zeros((dim_single, dim))
            for n in range(dim-1):
                mtx[2*n+1, n] = 1
            return spa.csr_matrix(mtx)

        # =============================================================================
        # Set numerical values for parameters
        # =============================================================================
        self._qubit_states_energy()
        if excited_level is None:
            excited_level = self.excited_level
        T_c = 1.2 # K
        k_b = 1.380649e-23 # J/K
        delta = 1.76*T_c*k_b # superconducting gap
        x_qp = 1e-8 #np.sqrt(2*np.pi*k_b*T/delta)*np.exp(-delta/(k_b*T))
        # Define Noise Spectral density without E_J (in accordance with
        # Catelani et al. https://doi.org/10.1103/PhysRevB.84.064517)
        S_qp = x_qp * (8 / (self.hbar * np.pi)) * np.sqrt(
            2 * delta / self.omega_q)
        # print("CircuitQ: Omega {:e}".format(self.omega_q / (self.hbar * 2 * np.pi)))

        # =============================================================================
        # Set ground state and excited state
        # =============================================================================
        ground_state = spa.csr_matrix(self.ground_state)
        excited_state = spa.csr_matrix(self.estates[:,excited_level])

        # =============================================================================
        # Calculate T1 contribution
        # =============================================================================
        T1_inv = 0
        transform_even = cooper_to_even(self.n_dim)
        transform_odd = cooper_to_odd(self.n_dim)

        for indices, sin_phi_half in self.sin_phi_half_operators_dict.items():
            mtx_list_even = copy.deepcopy(self.mtx_id_list)
            mtx_list_odd = copy.deepcopy(self.mtx_id_list)
            for node in indices[:2]:
                if ((node not in self.ground_nodes) and
                    (node in self.charge_basis_nodes)):
                    pos = self.subspace_pos[node]
                    mtx_list_even[pos] = transform_even
                    mtx_list_odd[pos] = transform_odd
            transform_even_num = self._kron_product(mtx_list_even)
            transform_odd_num = self._kron_product(mtx_list_odd)
            operator_pos = self.sin_phi_half_operators.index(sin_phi_half)
            sin_phi_half_operator = self.sin_phi_half_operators_num[operator_pos]
            braket = (ground_state.conjugate() * transform_even_num.transpose() *
                      sin_phi_half_operator *
                      transform_odd_num * excited_state.transpose()).data[0]
            E_J = self.parameter_values_dict[self.josephson_energies_global_dict[indices]]
            S_qp_E = S_qp * E_J
            # print("CircuitQ: E_J: {:e}".format(E_J))
            # print("CircuitQ: S_qp Quasiparticles Junction: {:e}".format(S_qp_E))
            # print("CircuitQ: Braket Junction: {:e}".format(abs(braket)))
            T1_inv += S_qp_E * abs(braket)**2
            # print("CircuitQ: Junction T1 contribution: {:e}".format(1/(S_qp_E * abs(braket)**2)))

        for indices, edge_flux in self.edge_flux_inductance_num.items():
            inductance_symbol = self.inductances_full_index[indices]
            inductance = self.parameter_values_dict[inductance_symbol]
            E_L = self.phi_0**2/inductance
            S_qp_L = S_qp * E_L
            braket = (ground_state.conjugate() *
                      edge_flux/(2*self.phi_0) *
                      excited_state.transpose()).data[0]
            # print("CircuitQ: E_L: {:e}".format(E_L))
            # print("CircuitQ: S_qp Quasiparticles Inductance: {:e}".format(S_qp_L))
            # print("CircuitQ: Braket Inductance: {:e}".format(abs(braket)))
            T1_inv += S_qp_L * abs(braket) ** 2
            # print("CircuitQ: Inductance T1 contribution: {:e}".format(1 /(S_qp_L * abs(braket) ** 2)))

        if T1_inv==0:
            T1 = None
        else:
            T1 = 1/T1_inv
        self.T1_quasiparticle = T1
        return self.T1_quasiparticle



    def get_T1_dielectric_loss(self, excited_level=None):
        """
        Estimates the T1 contribution due to dielectric loss. See the preprint on arXiv
        for more details about the formulas that have been used for this method.

        Parameters
        ----------
        excited_level: int (Default: Number of first state above the groundstate)
            Number of state, which is considered to be the excited state.

        Returns
        ----------
        T1_dielectric_loss: float
            T1 contribution due to dielectric loss.
        """
        # =============================================================================
        # Set numerical values for parameters
        # =============================================================================
        self._qubit_states_energy()
        Q_cap = 3e6*(2*np.pi*self.hbar*6e9/self.omega_q)**0.7
        k_b = 1.380649e-23
        T = 0.015
        thermal_factor = self.omega_q/(k_b*T)
        # Define Noise Spectral Density without C
        # Smith et al.:
        # S_q = 2*self.hbar/Q_cap * 1/np.tanh(thermal_factor/2) / (1 +
        #                                                      np.exp(thermal_factor))
        # Nguyen et al.:
        S_q = self.hbar/Q_cap * (1 + 1/np.tanh(thermal_factor/2) )

        # print("CircuitQ: Omega {:e}".format(self.omega_q/(self.hbar*2*np.pi)))
        # print("CircuitQ: Thermal Factor {:e}".format(thermal_factor))
        # print("CircuitQ: Q_cap {:e}".format(Q_cap))

        # =============================================================================
        # Set ground state and excited state
        # =============================================================================
        if excited_level is None:
            excited_level = self.excited_level
        ground_state = spa.csr_matrix(self.ground_state)
        excited_state = spa.csr_matrix(self.estates[:,excited_level])

        # =============================================================================
        # Calculate T1 contribution
        # =============================================================================
        T1_inv = 0

        for nodes, capacitance in self.capacitances.items():
            charge_operators_of_branch = []
            for node in nodes:
                if node in self.q_matrices_num_dict:
                    charge_operators_of_branch.append(self.q_matrices_num_dict[node])
                else:
                    charge_operators_of_branch.append(0)
            charge_operators_difference = (charge_operators_of_branch[1] -
                                          charge_operators_of_branch[0])
            braket = (ground_state.conjugate() * charge_operators_difference *
                      excited_state.transpose()).data[0]
            capacitance_value = self.parameter_values_dict[capacitance]
            T1_inv += S_q / (self.hbar ** 2 * capacitance_value) * abs(braket) ** 2
            # print("CircuitQ: NSD rescaled {:e}".format(S_q / self.hbar ** 2 / capacitance_value *
            #                                            (2 * self.e)**2))
            # print("CircuitQ: S_q rescaled (to compare to S_qp): {:e}".format(S_q/
            #                     (self.hbar ** 2 * 2 * np.pi * capacitance_value)*(2*self.e)**2))
            # print("CircuitQ: Braket Rescaled {:e}".format((abs(braket) / (2 * self.e))))
            # print("CircuitQ: T1_inv {:e}".format(T1_inv))

        if T1_inv==0:
            T1 = None
        else:
            T1 = 1/T1_inv
        self.T1_dielectric_loss = T1

        return self.T1_dielectric_loss

    def get_T1_flux(self, excited_level=None, lower_bound=False):
        """
        Estimates the T1 contribution due to flux noise. See the preprint on arXiv
        for more details about the formulas that have been used for this method.

        Parameters
        ----------
        excited_level: int (Default: Number of first state above the groundstate)
            Number of state, which is considered to be the excited state.
        lower_bound: bool (Default:False)
            If set to True, a lower bound for the T1 contribution will be calculated
            by summing over all inductive branches
            (not just closure branches as in default mode)

        Returns
        ----------
        T1_flux: float
            T1 contribution due to flux noise.
        """
        # =============================================================================
        # Set numerical values for parameters
        # =============================================================================
        self._qubit_states_energy()

        A_phi = 2*np.pi*self.phi_0*1e-6
        # Yan et al.:
        # gamma_phi = .9
        # S_phi = A_phi**2 * (2*np.pi*self.hbar/self.omega_q)**gamma_phi
        # Nguyen et al.:
        S_phi = (self.hbar**2/(2*self.e)**2
                 * (1/self.phi_0**2)
                 * 2 * np.pi * A_phi ** 2 * self.hbar / self.omega_q)
        # print("CircuitQ: Omega {:e}".format(self.omega_q / (self.hbar * 2 * np.pi)))

        # =============================================================================
        # Set ground state and excited state
        # =============================================================================
        if excited_level is None:
            excited_level = self.excited_level
        ground_state = spa.csr_matrix(self.ground_state)
        excited_state = spa.csr_matrix(self.estates[:,excited_level])

        # =============================================================================
        # Calculate T1 contribution
        # =============================================================================
        T1_inv = 0
        if lower_bound is False:
            for current_element in self.current_operators_num:
                braket = (excited_state.conjugate()*current_element*
                                                 ground_state.transpose()).data[0]
                # print("CircuitQ: Braket Rescaled {:e}".format(abs(braket)
                #                               *self.parameter_values[2]/self.phi_0))
                # print("CircuitQ: S_phi rescaled {:e}".format(S_phi/self.hbar**2
                #                         *self.parameter_values[2]**2/self.phi_0**2))
                T1_inv += S_phi/self.hbar**2 * abs(braket)**2
        else:
            for current_element in self.current_operators_all_num:
                braket = (ground_state.conjugate()*current_element*
                                                 excited_state.transpose()).data[0]
                # print("CircuitQ: Braket Rescaled {:e}".format(abs(braket)))
                #                               #*self.parameter_values[2]))
                T1_inv += S_phi/self.hbar**2 * abs(braket)**2
        if T1_inv==0:
            T1 = None
        else:
            T1 = 1/T1_inv
        self.T1_inductive_loss = T1

        return self.T1_inductive_loss