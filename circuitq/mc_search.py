import circuitq as cq
import numpy as np
import networkx as nx
import os
import copy
import pickle

# =============================================================================
# Directory handling
# =============================================================================
def create_directories(file_name):
    """
    Function that creates directories for data and figures if they don't exist
    and deletes files within those directories.

    Parameters
    ----------
    file_name: str
        Name of directory to store data and figures

    Returns
    ----------
    absolute data_path: str
    absolute figures_path: str
    """
    main_dir = os.path.abspath('..')
    data_path ='data/mc_search_' + str(file_name) + '/'
    figures_path = 'figures/mc_search_' + str(file_name) + '/'
    for path in [data_path,figures_path]:
        file_dir = os.path.join(main_dir, path)
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        else:
            for file in os.listdir(file_dir):
                file_path = os.path.join(file_dir, file)
                os.remove(file_path)

    return (os.path.join(main_dir, data_path),
           os.path.join(main_dir, figures_path) )

# =============================================================================
# Search run
# =============================================================================
def mc_search(n_dim, circuit_steps, parameter_steps, my_random,
              file_name, temperature, max_edges = 9, filter=1e30):
    """
    Optimization algorithm using the Metropolis-Hastings algorithm to search
    for circuits

    Parameters
    ----------
    n_dim: int
        Dimension of the Hilbert space for 1-dimensional problems. The dimension
        gets scaled down if the dimensionality increases.
    circuit_steps: int
        Number of iteration steps to vary the circuit (outer loop).
    parameter_steps: int
        Number of iteration steps to vary the parameters (inner loop).
    my_random: random.Random() instance
        The random instance for reproducability. Set a seed with
        random.seed(int) before.
    file_name: str
        This name will be added to the file and figure directories.
    tempreature: float
        Temperatur of the Metropolis-Hastings algorithm
    max_edges: int
        Maximal amount of allowed edges for circuit variation.
        If the graph has more edges, 'remove_edge' will be executed
        when calling do_action
    filter: float
        If the cost-function jumps by a value that is higher than the filter value,
        the change will be rejected.

    Returns
    ----------
    winner_instance: numpy array
        Lists the value of the cost function, the paramter values and
        the graph of the optimized instance.
    plot_data: list
        List of lists for monitor plotting with the following order:
        initial_cost, accepted_circuits, refused_circuits,
        accepted_parameters, refused_parameters, graph_list
    """
    abs_data_path, abs_figure_path = create_directories(file_name)
    circuit, h_num, graph = initialize_fluxonium(n_dim)
    accepted_list = []
    cost_ct, _ = get_cost(circuit, graph)
    cost_ct += abs(cost_ct / 2)
    initial_cost = [0, cost_ct]
    deletion = True
    circuit_counter = 0
    parameter_counter = 0
    accepted_circuits, refused_circuits = [[],[]] , [[],[]]
    accepted_parameters, refused_parameters = [[],[]] , [[],[]]
    cost_contributions_list = []
    graph_list = []
    with open(os.path.join(abs_data_path, 'search_data.pickle')
                ,'wb') as data_file:
        for c_s in range(circuit_steps):
            print("\n===== Circuit variation " + str(c_s) + " =====")
            new_graph = do_action(graph, my_random, max_edges=max_edges, deletion=True)
            pickle.dump({"graph" + str(c_s) : new_graph} , data_file)

            # Savely delete next paragraph after debug
            if c_s ==4:
                print("Debug")

            new_circuit = cq.CircuitQ(new_graph)
            new_graph = new_circuit.circuit_graph
            if not prepared_for_T1(new_circuit):
                deletion = False
                continue
            else:
                deletion = True
            dimension = int(n_dim/(len(new_circuit.nodes_wo_ground)**2))
            new_circuit.get_numerical_hamiltonian(dimension, default_zero=False)
            parameter_values = new_circuit.parameter_values
            new_circuit.get_eigensystem()
            new_cost_ct, new_cost_ct_contributions = get_cost(new_circuit, new_graph)
            if new_circuit.degenerated is True:
                print("\n================="
                      "\nSkip circuit due to degenerate qubit state."
                      "\n=================")
                continue
            cost_par = new_cost_ct
            cost_list = []
            values_list = []
            parameter_counter = circuit_counter + 1
            cost_contributions_list.append(new_cost_ct_contributions)
            for p_s in range(parameter_steps):
                print("===== Parameter variation " + str(p_s) + " =====")
                varied_parameter_values = vary_parameter(new_circuit, my_random)
                pickle.dump({"parameter{}.{}".format(c_s,p_s): varied_parameter_values},
                            data_file)
                new_circuit = cq.CircuitQ(new_graph)
                varied_h_num = new_circuit.get_numerical_hamiltonian(dimension,
                                           parameter_values=varied_parameter_values)
                new_circuit.get_eigensystem()
                new_cost_par, new_cost_par_contributions = get_cost(new_circuit, new_graph)
                if new_circuit.degenerated is True:
                    print("\n================="
                          "\nSkip parameter variation due to degenerate qubit state."
                          "\n=================")
                    continue
                accept = accept_refuse(new_cost_par, cost_par, temperature/9, my_random, filter)

                if accept:
                    parameter_values = varied_parameter_values
                    cost_list.append(new_cost_par)
                    values_list.append(varied_parameter_values)
                    cost_par = new_cost_par
                    accepted_parameters[0].append(parameter_counter)
                    accepted_parameters[1].append(new_cost_par)
                else:
                    refused_parameters[0].append(parameter_counter)
                    refused_parameters[1].append(new_cost_par)
                cost_contributions_list.append(new_cost_par_contributions)
                parameter_counter += 1
            if accept_refuse(cost_par, cost_ct, temperature, my_random, filter):
                print("\n graph accepted\n")
                graph = new_graph
                cost_ct = new_cost_ct
                cq.visualize_circuit_general(new_graph, abs_figure_path + str(c_s) + '_accepted_circuit' )
                graph_list.append((new_graph, str(c_s) + " accepted"))
                if len(cost_list) > 0:
                    accepted_list.append([min(cost_list),values_list[cost_list.index(min((cost_list)))],
                                          new_graph, c_s])
                accepted_circuits[0].append(circuit_counter)
                accepted_circuits[1].append(new_cost_ct)
            else:
                print("\n graph refused\n")
                cq.visualize_circuit_general(new_graph, abs_figure_path + str(c_s) + '_refused_circuit')
                graph_list.append((new_graph, str(c_s) + " refused"))
                refused_circuits[0].append(circuit_counter)
                refused_circuits[1].append(new_cost_ct)
            circuit_counter += parameter_steps + 1
    if len(accepted_list) > 0:
        accepted_list = np.array(accepted_list)
        lowest_cost = min(accepted_list[:,0])
        lowest_cost_pos = accepted_list[:,0].tolist().index(lowest_cost)
        winner_instance = accepted_list[lowest_cost_pos,:]
    else:
        winner_instance = None
    plot_data = [initial_cost, accepted_circuits, refused_circuits,
                 accepted_parameters, refused_parameters, graph_list]

    return winner_instance, plot_data, cost_contributions_list

# =============================================================================
# Initialize Fluxonium
# =============================================================================
def initialize_fluxonium(n_dim):
    print("\n======================================"
          "\n======================================"
          "\nInitialize Fluxonium")
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, element='C')
    graph.add_edge(0, 1, element='J')
    graph.add_edge(0, 1, element='L')
    circuit = cq.CircuitQ(graph)
    h_num = circuit.get_numerical_hamiltonian(n_dim)
    circuit.get_eigensystem()
    return circuit, h_num, graph

# =============================================================================
# Get value of the cost function for given instance
# =============================================================================
def get_cost(circuit, graph):
    anharmonicity = circuit.get_spectrum_anharmonicity()
    if anharmonicity is None:
        return None, None
    anharmonicity_scaled = -anharmonicity*1e2
    nbr_edges = len(graph.edges)
    nbr_edges_scaled = nbr_edges*2e1
    T1_quasiparticles = circuit.get_T1_quasiparticles()
    # T1_quasiparticles_scaled = -T1_quasiparticles*1e4
    T1_charge = circuit.get_T1_charge()
    # T1_charge_scaled = -T1_charge * 1e4
    T1_flux = circuit.get_T1_flux()
    # T1_flux_scaled = -T1_flux*1e1
    T1_ges = 1/( 1/T1_quasiparticles + 1/T1_flux + 1/T1_charge )
    T1_ges_scaled = -T1_ges*1e6
    cost = (anharmonicity_scaled + nbr_edges_scaled + T1_ges_scaled)
    cost_contributions = [T1_quasiparticles, T1_charge, T1_flux, T1_ges_scaled,
                          anharmonicity_scaled, nbr_edges_scaled]
    print("Anharmonicity =", anharmonicity, "scaled:", anharmonicity_scaled)
    print("Number of edges =", nbr_edges, "scaled:", nbr_edges_scaled)
    print("T1_quasiparticles = {:e}".format(T1_quasiparticles))
          # ,"scaled: {:e}".format(T1_quasiparticles_scaled))
    print("T1_charge = {:e}".format(T1_charge))
          # ,"scaled: {:e}".format(T1_charge_scaled))
    print("T1_flux = {:e}".format(T1_flux))
          # ,"scaled: {:e}".format(T1_flux_scaled))
    print("T1_ges = {:e}".format(T1_ges),
          "scaled: {:e}".format(T1_ges_scaled))
    print("Cost = {:e}".format(cost))
    return cost, cost_contributions

# =============================================================================
# Possible graph modifications
# =============================================================================

# Add element in between two existing nodes
# =============================================================================
def add_edge(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    u = my_random.choice(nodes)
    nodes.remove(u)
    v = my_random.choice(nodes)
    rdm_element = my_random.choice(elements)
    graph.add_edge(u, v, element = rdm_element)
    print("Added element:", rdm_element, "in between nodes", u, "and", v )
    return graph

# Insert element in between two existing elements
# =============================================================================
def insert_edge(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    rdm_node = my_random.choice(nodes)
    rdm_edge = my_random.choice(list(graph.edges(rdm_node, keys=True, data=True)))
    graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
    new_node = max(nodes) + 1
    rdm_case_nbr = my_random.uniform(0, 1)
    rdm_element = my_random.choice(elements)
    if rdm_case_nbr < 0.5:
        graph.add_edge(rdm_edge[0], new_node, **rdm_edge[3])
        graph.add_edge(new_node, rdm_edge[1], element = rdm_element)
        print("Inserted edge in between nodes", new_node, "and", rdm_edge[1],
              "with element", rdm_element)
    elif 0.5 <= rdm_case_nbr <= 1:
        graph.add_edge(new_node, rdm_edge[1], **rdm_edge[3])
        graph.add_edge(rdm_edge[0], new_node, element = rdm_element)
        print("Inserted edge in between nodes", rdm_edge[0], "and", new_node,
              "with element", rdm_element)
    return graph

# Add new loop to existing node
# =============================================================================
def add_loop(graph, my_random, elements):
    graph = copy.deepcopy(graph)
    nodes = list(graph.nodes)
    rdm_node = my_random.choice(nodes)
    new_node = max(nodes) + 1
    rdm_element = my_random.choice(elements)
    graph.add_edge(rdm_node, new_node, element=rdm_element)
    rdm_element_new = my_random.choice(elements)
    while rdm_element_new == rdm_element:
        rdm_element_new = my_random.choice(elements)
    graph.add_edge(rdm_node, new_node, element=rdm_element_new)
    print("Added loop with nodes", rdm_node, "and", new_node,
          "with elements", rdm_element, "and", rdm_element_new)
    return graph

# Remove edge
# =============================================================================
def remove_edge(graph, my_random, no_deletion = 0):
    graph = copy.deepcopy(graph)
    edges = list(graph.edges.data(keys=True))
    rdm_edge = my_random.choice(edges)
    if len(edges) == 2:
        print("Only two edges. No deletion!")
        return do_action(graph, my_random, deletion=False)
    ngb1 = list(graph.neighbors(rdm_edge[0]))
    nbr_ngb1 = len(ngb1)
    ngb2 = list(graph.neighbors(rdm_edge[1]))
    nbr_ngb2 = len(ngb2)
    if nbr_ngb1 > 1 and nbr_ngb2 > 1:
        graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
        print("Removed edge in between nodes", rdm_edge[0], "and", rdm_edge[1],
              "with element", rdm_edge[3]["element"])
        if not graph.has_edge(rdm_edge[0], rdm_edge[1]):
            graph = nx.contracted_nodes(graph, rdm_edge[0], rdm_edge[1])
        return graph
    # If the edge has one boundary node:
    else:
        nbr_edges = graph.number_of_edges(rdm_edge[0], rdm_edge[1])
        if nbr_edges > 2:
            graph.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
            print("Removed edge in between nodes", rdm_edge[0], "and", rdm_edge[1],
                  "with element", rdm_edge[3]["element"])
            return graph
        # Delete loop if it's connected to the rest of the graph with at
        # least two edges
        elif nbr_edges == 2:
            if nbr_ngb1 > 1:
                connected_node = rdm_edge[0]
                same_edge_node = rdm_edge[1]
            elif nbr_ngb2 > 1:
                connected_node = rdm_edge[1]
                same_edge_node = rdm_edge[0]
            else:
                raise Exception("Isolated subgraph detected!")
            connected_ngb = list(graph.neighbors(connected_node))
            connected_ngb.remove(same_edge_node)
            if len(connected_ngb)>1:
                delete = True
            else:
                nbr_connect_edges = graph.number_of_edges(connected_node, connected_ngb[0])
                if nbr_connect_edges >1 :
                    for edge in edges:
                        if ((edge[0] == rdm_edge[0] and edge[1] == rdm_edge[1])
                                or (edge[0] == rdm_edge[1] and edge[1] == rdm_edge[0])):
                            graph.remove_edge(edge[0], edge[1], key=edge[2])
                            print("Removed edge in between nodes", edge[0], "and", edge[1],
                                  "with element", edge[3]["element"])
                        isolates = list(nx.isolates(graph))
                        for node in isolates:
                            graph.remove_node(node)
                    return graph
                else:
                    print("Loop could not be deleted (weakly connected)")
                    no_deletion_new = no_deletion + 1
                    if no_deletion_new > 20:
                        raise Exception("Might be hard or impossible to delete further elements.")
                    return remove_edge(graph, my_random, no_deletion_new)
        else:
            raise Exception("Only one edge connected to boundary node")

# =============================================================================
# Perform a graph modification
# =============================================================================
def do_action(graph, my_random, max_edges=9, deletion = True):
    elements = ['C','L','J']
    actions = ['add_edge', 'insert_edge', 'add_loop']
    if deletion:
        actions.append('remove_edge')
    if len(graph.edges) > max_edges:
        print("------- Maximal number of edges reached -> deleting -------")
        rdm_action = 'remove_edge'
    else:
        rdm_action = my_random.choice(actions)
        print("\n======================================"
              "\n======================================")
        print("Action = " + rdm_action)
    if rdm_action == 'add_edge':
        return add_edge(graph, my_random, elements)
    elif rdm_action == 'insert_edge':
        return insert_edge(graph, my_random, elements)
    elif rdm_action == 'add_loop':
        return add_loop(graph, my_random, elements)
    elif rdm_action == 'remove_edge':
        return remove_edge(graph, my_random)
    else:
        raise Exception("Action not recognized.")

# =============================================================================
# Vary parameter of circuit elements
# =============================================================================
def vary_parameter(circuit_instance, my_random):
    parameter_values = copy.deepcopy(circuit_instance.parameter_values)
    parameter = my_random.choice(parameter_values)
    parameter_idx = parameter_values.index(parameter)
    print("======================================")
    print("Changed parameter",
          circuit_instance.h_parameters[parameter_idx],
          "with value {:e}".format(parameter))
    if parameter == 0:
        parameter = 1
    else:
        rdm_factor = my_random.choice([4/5, 6/5])
        parameter = rdm_factor*parameter
    parameter_values[parameter_idx] = parameter
    print(" to {:e}".format(parameter))
    return parameter_values

# =============================================================================
# Accept or Refuse
# =============================================================================
def accept_refuse(new_cost, old_cost, temperature, my_random, filter):
    if abs(old_cost-new_cost) >= filter:
        print("\n----- Refuse: Jump in cost-function too large "
              "according to filter -----\n")
        return False
    exp_arg = float((old_cost - new_cost) / temperature)
    p_accept = min([1, np.exp(exp_arg)])
    print("Acceptance probability: " + str(p_accept))
    random_nbr = my_random.uniform(0, 1)
    if random_nbr <= p_accept:
        print("accepting parameter change")
        return True
    print("refusing parameter change")
    return False

# =============================================================================
# Check if all T1 contributions can be calculated
# =============================================================================
def prepared_for_T1(circuit):
    prepared = True
    # Check if there is a junction in the circuit
    if not any('J'== item[2]['element'] for item
               in circuit.circuit_graph.edges.data()):
        print("There is no junction in the circuit! "
              "T1_quasiparticles can not be computed.")
        prepared = False
    # Check if there is a closed superconducting loop in the circuit
    if len(circuit.loop_fluxes) == 0:
        print("There is no junction in the circuit! "
              "T1_quasiparticles can not be computed.")
        prepared = False
    return prepared





