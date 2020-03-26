
# from functionsAndClasses.CCircuitGeneral import *
import networkx as nx
import itertools as it
import copy
import random
import sys
import numpy as np
import sympy as sp
import pickle # For debuging
from sympy.physics.paulialgebra import Pauli
from sympy.physics.quantum import TensorProduct

def visualize_circuit(circuit_name, save_as):
    """
    Visualises a circuit by creating a figure.

    Parameters
    ----------
    circuit_name : NetworkX Graph
        Input circuit
    save_as : String
        Name of saved file in figures folder
    """

    circuit_func = copy.deepcopy(circuit_name)
    for e in circuit_func.edges.data():
        e[2]['label'] = e[2]['element']
    circuit_vis = nx.nx_agraph.to_agraph(circuit_func)
    circuit_vis.node_attr['width'] = 0.25
    circuit_vis.node_attr['height'] = 0.25
    circuit_vis.node_attr['fixedsize'] = True
    if sys.platform == 'linux':
        circuit_vis.edge_attr['fontpath'] = '/usr/share/fonts/dejavu'
        circuit_vis.edge_attr['fontname'] = 'DejaVuSans'
        circuit_vis.node_attr['fontpath'] = '/usr/share/fonts/dejavu'
        circuit_vis.node_attr['fontname'] = 'DejaVuSans'
    circuit_vis.draw(
        '/Users/philipp/Dropbox (Personal)/CircuitDesign/figures/' + save_as + '.png',
        prog='circo')


def visualize_circuit_general(circuit_name, save_as):
    """
    Visualises a circuit by creating a figure to an arbitrary path.

    Parameters
    ----------
    circuit_name : NetworkX Graph
        Input circuit
    save_as : String
        Arbitrary figure path
    """

    circuit_func = copy.deepcopy(circuit_name)
    for e in circuit_func.edges.data():
        e[2]['label'] = e[2]['element']
    circuit_vis = nx.nx_agraph.to_agraph(circuit_func)
    circuit_vis.node_attr['width'] = 0.25
    circuit_vis.node_attr['height'] = 0.25
    circuit_vis.node_attr['fixedsize'] = True
    circuit_vis.edge_attr['fontpath'] = '/usr/share/fonts/dejavu'
    circuit_vis.edge_attr['fontname'] = 'DejaVuSans'
    circuit_vis.node_attr['fontpath'] = '/usr/share/fonts/dejavu'
    circuit_vis.node_attr['fontname'] = 'DejaVuSans'
    circuit_vis.draw(save_as + '.png', prog='dot')


def extremize_coeff(CCircuit_instance, term, operation = "maximize"):

    if not CCircuit_instance.qutip_hamiltonian_flag:
        raise Exception("get_qutip_hamiltonian() has to be run before"
                        + " calling this function.")

    instance = copy.deepcopy(CCircuit_instance)
    operators = [1, Pauli(1), Pauli(2), Pauli(3)]
    all_terms_components = list(it.product(operators, repeat=len(instance.free_Phis)))
    all_other_terms = [TensorProduct(*item) for item in all_terms_components]
    all_other_terms.remove(term)
    parameters_values = instance.parameters_values
    parameters_values_keys = instance.parameters_values.keys()
    number_parameters = len(parameters_values.keys())
    factorlist = [n*10**(p) for n in range(1,2) for p in range(-1,1)]
    factor_combinations = list(it.product(factorlist, repeat=number_parameters))

    ratios = []
    coefficients = []
    ratios_parameters = []

    nbr_iterations = len(factor_combinations)
    iteration = 0

    for factor_tuple in factor_combinations:
        new_parameters_values = copy.deepcopy(parameters_values)

        for n, parameter in enumerate(parameters_values_keys):
            new_parameters_values[parameter] = factor_tuple[n]*parameters_values[parameter]

        instance.get_qutip_hamiltonian(new_parameters_values)
        H_pauli = instance.get_pauli_hamiltonian()
        coefficient = H_pauli.coeff(term)
        coefficients.append(coefficient)

        other_coefficients = []
        for item in all_other_terms:
            item_coeff = abs(H_pauli.coeff(item))
            other_coefficients.append(item_coeff)
        other_coefficients_sum = sum(other_coefficients)
        ratio = coefficient/other_coefficients_sum
        ratios.append(ratio)
        ratios_parameters.append(new_parameters_values)

        iteration += 1
        print("Iteration: " + str(iteration) + "/" + str(nbr_iterations))


    if operation == "maximize":
        max_ratio = max(ratios)
        max_parameters = ratios_parameters[ratios.index(max_ratio)]
        coeff_value = coefficients[ratios.index(max_ratio)]

        return coeff_value, max_ratio, max_parameters

    elif operation == "minimize":
        min_ratio = min(ratios)
        min_parameters = ratios_parameters[ratios.index(min_ratio)]
        coeff_value = coefficients[ratios.index(min_ratio)]

        return coeff_value, min_ratio, min_parameters

    else:
        raise Exception("kwarg 'operation' has to be either 'maximize' or 'minimize'")


def extremize_coeff_stochastic(CCircuit_instance, terms: list, temperature = 1,
                               n_steps_extr = 10, operation = "maximize", debug=False):

    accepted_r = [[],[]]
    refused_r = [[],[]]
    accepted_H_pauli = []
    accepted_parameters = []
    accepted_coefficients_sum = []

    if not CCircuit_instance.qutip_hamiltonian_flag:
        raise Exception("get_qutip_hamiltonian() has to be run before"
                        + " calling this function.")

    instance = copy.deepcopy(CCircuit_instance)
    instance.get_pauli_hamiltonian(symbolic=True)

    if debug is True:
        # Save for debugging
        file_dir = '/Users/philipp/Dropbox (Personal)/CircuitDesign/data/' \
                   'MCCircuitSearch/extremize_coeff_debug.pickle'
        with open(file_dir, 'wb') as file:
            pickle.dump({'instance': instance}, file)

    operators = [1, Pauli(1), Pauli(2), Pauli(3)]
    all_terms_components = list(it.product(operators, repeat=len(instance.marks)))
    all_other_terms = [TensorProduct(*item) for item in all_terms_components]
    for term in terms:
        if term in all_other_terms:
             all_other_terms.remove(term)
        else:
            print("Term " + str(term) +
                  " not in all_other_terms. Dimension of circuit might be different!!")
    parameters_values = instance.parameters_values
    parameters_values_keys = list(instance.parameters_values.keys())
    parameters_loop = copy.deepcopy(parameters_values_keys)
    for item in parameters_values_keys:
        if (str(item) == 'KPhi_0' or str(item) == 'L_{char}'
            or str(item) == 'C_{char}'):
            parameters_loop.remove(item)
    parameters_values_keys = parameters_loop
    number_parameters = len(parameters_values_keys)
    old_parameters_values = copy.deepcopy(parameters_values)

    nbr_steps = n_steps_extr

    H_pauli = sp.N(instance.H_pauli_symbolic.subs(old_parameters_values))
    print("H_pauli = " + str(H_pauli))

    def get_ratio(H_pauli_f):

        coefficients_abs = []
        for term_l in terms:
            coefficient = H_pauli_f.coeff(term_l)
            coefficients_abs.append(abs(coefficient))
        coefficients_sum_f = float(sum(coefficients_abs))

        other_coefficients = []
        for item in all_other_terms:
            item_coeff = abs(H_pauli_f.coeff(item))
            other_coefficients.append(item_coeff)
        other_coefficients_sum_f = float(sum(other_coefficients))

        if other_coefficients_sum_f == 0:
            ratio_f = np.inf
        else:
            ratio_f = coefficients_sum_f / other_coefficients_sum_f

        return ratio_f, coefficients_sum_f, other_coefficients_sum_f

    ratio, coefficients_sum, other_coefficients_sum = get_ratio(H_pauli)
    old_ratio = copy.deepcopy(ratio)

    acceptance_counter = 0

    for n in range(0,nbr_steps):

        print("Acceptance: " + str(acceptance_counter) + " / " + str(n+1))

        new_parameters_values = copy.deepcopy(old_parameters_values)

        rdm_parameter = parameters_values_keys[random.randint(0,number_parameters-1)]
        # rdm_factor = random.uniform(10**-4,10**4)
        # new_parameters_values[rdm_parameter] = rdm_factor * parameters_values[rdm_parameter]
        change_range = abs(parameters_values[rdm_parameter])
        rdm_factor = random.uniform(-change_range, change_range)
        new_parameters_values[rdm_parameter] += rdm_factor

        print("Changed parameter: " + str(rdm_parameter))
        print("Old Value: " + str(parameters_values[rdm_parameter]))
        print("New Value: " + str( new_parameters_values[rdm_parameter]))

        if new_parameters_values[rdm_parameter] <= 0:
            new_parameters_values[rdm_parameter] -= rdm_factor

        H_pauli = sp.N(instance.H_pauli_symbolic.subs(new_parameters_values))
        ratio, coefficients_sum, other_coefficients_sum = get_ratio(H_pauli)
        new_ratio = copy.deepcopy(ratio)

        print("New ratio: " + str(new_ratio) + ", Old ratio:" + str(old_ratio))
        print("Step: " + str(n + 1) + "/" + str(nbr_steps))
        print("sum coefficients: " + str(coefficients_sum) )
        print("Sum other coefficients: " + str(other_coefficients_sum))

        if operation == "maximize":

            exp_arg = float((new_ratio-old_ratio)/temperature)
            p_accept = min([1, np.exp(exp_arg)])
            print("p_accept: " + str(p_accept))
            random_nbr = random.uniform(0,1)

            if random_nbr <= p_accept:
                old_parameters_values = new_parameters_values
                old_ratio = new_ratio
                acceptance_counter += 1
                accepted_r[0].append(n)
                accepted_r[1].append(new_ratio)
                accepted_H_pauli.append(H_pauli)
                accepted_parameters.append(new_parameters_values)
                accepted_coefficients_sum.append(coefficients_sum)
            else:
                refused_r[0].append(n)
                refused_r[1].append(new_ratio)

        elif operation == "minimize":

            exp_arg = float((old_ratio - new_ratio) / temperature)
            p_accept = min([1, np.exp(exp_arg)])
            random_nbr = random.uniform(0,1)

            if random_nbr <= p_accept:
                old_parameters_values = new_parameters_values
                old_ratio = new_ratio
                acceptance_counter += 1
                accepted_r[0].append(n)
                accepted_r[1].append(new_ratio)
                accepted_H_pauli.append(H_pauli)
                accepted_parameters.append(new_parameters_values)
                accepted_coefficients_sum.append(coefficients_sum)
            else:
                refused_r[0].append(n)
                refused_r[1].append(new_ratio)

        else:
            raise Exception("kwarg 'operation' has to be either 'maximize' or 'minimize'")

    if operation == "maximize":
        omptimized_ratio = max(accepted_r[1])
    elif operation == "minimize":
        omptimized_ratio = min(accepted_r[1])

    optimized_H_pauli = accepted_H_pauli[accepted_r[1].index(omptimized_ratio)]
    optimized_parameters = accepted_parameters[accepted_r[1].index(omptimized_ratio)]
    optimized_coefficients_sum = accepted_coefficients_sum[accepted_r[1].index(omptimized_ratio)]


    return optimized_H_pauli, optimized_coefficients_sum, omptimized_ratio, optimized_parameters, accepted_r, refused_r






