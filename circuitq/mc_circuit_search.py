import circuitq as cq

import numpy as np
import networkx as nx
import itertools as it
import os
import sympy as sp
import copy
import pickle

# =============================================================================
# Directory handling
# =============================================================================
"""
Function that creates directories for data and figures if they don't exist
and deletes files within those directories.

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
def create_directories(file_name):
    main_dir = os.path.abspath('..')
    data_path ='data/mc_search' + str(file_name)
    figures_path = 'figures/mc_search' + str(file_name)
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
"""
Optimization algorithm using the Metropolis-Hastings algorithm to search
for circuits
"""

def mc_search(instance_name, my_random, temperature = 0.05, n_max = 10):
    accepted_f = [[],[]]
    accepted_instances = []
    refused_f = [[],[]]
    file_dir, figures_dir = create_directories(instance_name)
    data_path = os.path.join(file_dir, 'mc_search_data.pickle')
    n = 0
    max_input_dimension = get_max_dim(terms)
    instance, circuit = initialize_transmon(characteristic_values)
    current_f, h_pauli, decimal_powers = get_fvalue(instance, terms, max_input_dimension)

    with open(data_path, 'wb') as data_file:
        pickle.dump({"characteristic_values": characteristic_values}, data_file)
        while n<n_max:
            print("\n \nRound {}:".format(n))
            new_circuit = do_action(circuit,  my_random)
            loop_count = 0
            while True:
                circuit_data = {"Circuit {}.{}".format(n, loop_count): new_circuit}
                pickle.dump(circuit_data, data_file)

                new_instance = Circuit(new_circuit, characteristic_values["e"],
                                       characteristic_values["hbar"], print_feedback=False)
                parameters_values = get_parameters_values(new_instance, characteristic_values)

                instance_data = {"Instance {}.{}".format(n, loop_count) : new_instance,
                                "Parameters {}.{}".format(n, loop_count) : parameters_values}
                pickle.dump(instance_data, data_file)


                new_instance.get_qutip_hamiltonian(parameters_values)
                h_pauli = new_instance.get_pauli_hamiltonian()
                current_dimension = get_max_dim(h_pauli.args)

                if (current_dimension > max_input_dimension
                        and current_dimension >= 5):
                    print("Action = remove_edge (due to dimension)")
                    new_circuit = remove_edge(new_circuit, my_random)
                    loop_count += 1
                else:
                    break

            decision, new_f, new_H_pauli = accept_refuse(new_instance, current_f, terms, max_input_dimension,
                                                         temperature, my_random, decimal_powers_f=decimal_powers)

            print("\nDecision = {}".format(decision))
            if decision == "accept":
                circuit = new_circuit
                current_f = new_f
                h_pauli = new_H_pauli
                visualize_circuit_general(circuit, os.path.join(figures_dir, 'AcceptedCircuit' + str(n)))
                accepted_instances.append(new_instance)
                accepted_f[0].append(n)
                accepted_f[1].append(new_f)

            else:
                visualize_circuit_general(new_circuit, os.path.join(figures_dir, 'RefusedCircuit' + str(n)))
                refused_f[0].append(n)
                refused_f[1].append(new_f)

            n += 1

    optimized_instance = accepted_instances[accepted_f[1].index(min(accepted_f[1]))]

    return optimized_instance, accepted_f, refused_f, file_dir, figures_dir, data_path



# =============================================================================
# Initialize Transmon
# =============================================================================
def initialize_transmon():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1, element='C')
    graph.add_edge(0, 1, element='J')
    circuit = cq.CircuitQ(graph)

    return None


# =============================================================================
# Get f-value (correct coefficients + anharmonicity)
# =============================================================================

def get_fvalue(Circuit_instance, terms: list, max_input_dim_f,
               decimal_powers_f = None):

    d_p_f = dict()
    instance = copy.deepcopy(Circuit_instance)
    operators = [1, Pauli(1), Pauli(2), Pauli(3)]
    instance_dimension = len(instance.nodes_wo_ground)
    all_terms_components = list(it.product(operators, repeat=instance_dimension))
    all_other_terms = [TensorProduct(*item) for item in all_terms_components]
    for term in terms:
        if term in all_other_terms:
            all_other_terms.remove(term)
    h_pauli = instance.get_pauli_hamiltonian()
    print("\nh_pauli = {}\n".format(h_pauli), flush=True)
    c_list = []
    for term in terms:
        c_list.append(abs(h_pauli.coeff(term)))
    c = sum(c_list)
    print("c = {}".format(c))
    k_list = []
    for item in all_other_terms:
        item_coeff = abs(h_pauli.coeff(item))
        k_list.append(item_coeff)
    k = float(sum(k_list))
    print("k = {}".format(k))

    nbr_entries = 0
    for item in k_list:
        if item != 0:
            nbr_entries += 1

    ratio = k / ((c*10**2) + k)

    anharmonicities = instance.get_anharmonicities()
    print("anharmonicities = {}".format(anharmonicities))
    h_list = [1 - anharmonicity for anharmonicity in anharmonicities]
    h = sum(h_list)/len(h_list)
    print("h = {}".format(h))

    l = len(Circuit_instance.circuit_graph.edges)
    print("l = {}".format(l))

    d = abs(max_input_dim_f - instance_dimension)
    print("d = {}".format(d))

    def decimal_power(arg):
        if arg == 0:
            arg += 1
        arg = float(arg)
        return np.floor(np.log10(abs(arg)))

    if decimal_powers_f is None:
        d_p_f["h"] = decimal_power(h)
        d_p_f["l"] = decimal_power(l)
    else:
        d_p_f = decimal_powers_f

    f = float( ratio + h*10**(-d_p_f["h"]-2)
              + l*10**(-2) + d*10**(-1))
    print("ratio k / ((c*10**2) + k) = {}".format(ratio))
    print("d_p_f h = " + str(d_p_f["h"]) + " --- d_p_f l = " + str(d_p_f["l"]))
    print("h scaled = {}".format(float(h*10**(-d_p_f["h"]-2))))
    print("l scaled = {}".format(float(l*10**(-2))))
    print("d scaled = {}".format(float(d*10**(-1))))
    print("f = {}".format(f))
    return f, h_pauli, d_p_f

# =============================================================================
# Possible actions
# =============================================================================

# Add element in between two existing nodes
def add_edge(circuit_f, my_random_f, elements_f):
    circuit_f = copy.deepcopy(circuit_f)
    nodes = list(circuit_f.nodes)
    u = my_random_f.choice(nodes)
    nodes.remove(u)
    v = my_random_f.choice(nodes)
    rdm_element = my_random_f.choice(elements_f)
    circuit_f.add_edge(u, v, element = rdm_element)
    return circuit_f

# Insert element in between two existing elements
def insert_edge(circuit_f, my_random_f, elements_f):
    circuit_f = copy.deepcopy(circuit_f)
    nodes = list(circuit_f.nodes)
    rdm_node = my_random_f.choice(nodes)
    rdm_edge = my_random_f.choice(list(circuit_f.edges(rdm_node, keys=True, data=True)))
    circuit_f.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
    new_node = max(nodes) + 1
    rdm_case_nbr = my_random_f.uniform(0,1)
    rdm_element = my_random_f.choice(elements_f)
    if rdm_case_nbr < 0.5:
        circuit_f.add_edge(rdm_edge[0], new_node, **rdm_edge[3])
        circuit_f.add_edge(new_node, rdm_edge[1], element = rdm_element)
    elif 0.5 <= rdm_case_nbr <= 1:
        circuit_f.add_edge(new_node, rdm_edge[1], **rdm_edge[3])
        circuit_f.add_edge(rdm_edge[0], new_node, element = rdm_element)
    return circuit_f

# Add new loop to existing node
def add_loop(circuit_f, my_random_f, elements_f):
    circuit_f = copy.deepcopy(circuit_f)
    nodes = list(circuit_f.nodes)
    rdm_node = my_random_f.choice(nodes)
    new_node = max(nodes) + 1
    rdm_element = my_random_f.choice(elements_f)
    circuit_f.add_edge(rdm_node, new_node, element=rdm_element)
    rdm_element_new = my_random_f.choice(elements_f)
    while rdm_element_new == rdm_element:
        rdm_element_new = my_random_f.choice(elements_f)
    circuit_f.add_edge(rdm_node, new_node, element=rdm_element_new)
    return circuit_f

# Remove edge
def remove_edge(circuit_f, my_random_f, no_deletion = 0):
    circuit_f = copy.deepcopy(circuit_f)
    edges = list(circuit_f.edges.data(keys=True))
    rdm_edge = my_random_f.choice(edges)
    if 'protected' in rdm_edge[3]:
        if rdm_edge[3]['protected'] == 'y':
            print("PROTECTED!")
            return circuit_f

    if len(edges) == 2:
        print("Only two edges. No deletion!")
        return circuit_f

    ngb1 = list(circuit_f.neighbors(rdm_edge[0]))
    nbr_ngb1 = len(ngb1)
    ngb2 = list(circuit_f.neighbors(rdm_edge[1]))
    nbr_ngb2 = len(ngb2)

    if nbr_ngb1 > 1 and nbr_ngb2 > 1:
        circuit_f.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
        if not circuit_f.has_edge(rdm_edge[0], rdm_edge[1]):
            circuit_f = nx.contracted_nodes(circuit_f, rdm_edge[0], rdm_edge[1])
        return circuit_f
    else:
        nbr_edges = circuit_f.number_of_edges(rdm_edge[0],rdm_edge[1])
        if nbr_edges > 2:
            circuit_f.remove_edge(rdm_edge[0], rdm_edge[1], key=rdm_edge[2])
            return circuit_f
        elif nbr_edges == 2:
            if nbr_ngb1 > 1:
                connected_node = rdm_edge[0]
                same_edge_node = rdm_edge[1]
            elif nbr_ngb2 > 1:
                connected_node = rdm_edge[1]
                same_edge_node = rdm_edge[0]
            else:
                raise Exception("Isolated subgraph detected!")
            connected_ngb = list(circuit_f.neighbors(connected_node))
            connected_ngb.remove(same_edge_node)
            if len(connected_ngb)>1:
                delete = True
            else:
                nbr_connect_edges = circuit_f.number_of_edges(connected_node,connected_ngb[0])
                if nbr_connect_edges >1 :
                    delete = True
                else:
                    delete = False

            if delete is True:

                for edge in edges:
                    if ( (edge[0] == rdm_edge[0] and edge[1] == rdm_edge[1])
                        or (edge[0] == rdm_edge[1] and edge[1] == rdm_edge[0]) ):
                        circuit_f.remove_edge(edge[0], edge[1], key=edge[2])
                    isolates = list(nx.isolates(circuit_f))
                    for node in isolates:
                        circuit_f.remove_node(node)
                return circuit_f

            else:
                print("Loop could not be deleted (weakly connected)")
                no_deletion_new = no_deletion + 1
                if no_deletion_new > 20:
                    raise Exception("Might be hard or impossible to delete further elements.")
                return remove_edge(circuit_f, my_random_f, no_deletion_new)

        else:
            raise Exception("Only one edge connected to boundary node")


def dim_check(expr, dim_list):
    if type(expr) == TensorProduct:
        dim_list.append(len(expr.args))
    elif (type(expr) == sp.Add or type(expr) == sp.Mul
          or type(expr) == sp.Pow):
        for arg in expr.args:
            dim_check(arg, dim_list)

def get_max_dim(expr_list):
    for expr in expr_list:
        dimensions = []
        dim_check(expr, dimensions)
    if len(dimensions) == 0:
        dimensions.append(1)
    return max(dimensions)

def do_action(circuit_f, my_random_f):
    circuit_f = copy.deepcopy(circuit_f)
    elements = ['C','L','J']
    actions = ['add_edge', 'insert_edge', 'add_loop', 'remove_edge']
    rdm_action = my_random_f.choice(actions)
    print("\nAction = " + rdm_action)
    if rdm_action == 'add_edge':
        return add_edge(circuit_f, my_random_f, elements)
    elif rdm_action == 'insert_edge':
        return insert_edge(circuit_f, my_random_f, elements)
    elif rdm_action == 'add_loop':
        return add_loop(circuit_f, my_random_f, elements)
    elif rdm_action == 'remove_edge':
        return remove_edge(circuit_f, my_random_f)
    else:
        raise Exception("Action not recognized.")

# =============================================================================
# Accept or Refuse
# =============================================================================

def accept_refuse(Circuit_instance, current_f_f, terms: list,
                  max_input_dim_f, temperature_f, my_random_f,
                  decimal_powers_f=None):

    new_f, H_pauli, d_p_f = get_fvalue(Circuit_instance, terms, max_input_dim_f,
                                       decimal_powers_f=decimal_powers_f)
    exp_arg = float((current_f_f - new_f) / temperature_f)
    p_accept = min([1, np.exp(exp_arg)])
    print("p_accept: {}".format(p_accept))
    random_nbr = my_random_f.uniform(0, 1)
    print("random_nbr: {}".format(random_nbr))
    if random_nbr <= p_accept:
        return "accept", new_f, H_pauli
    return "refuse", new_f, H_pauli

