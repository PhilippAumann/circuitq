
import os, sys
os.chdir('/Users/philipp/Dropbox (Personal)/CircuitDesign')
sys.path.append('/Users/philipp/Dropbox (Personal)/CircuitDesign')
from functionsAndClasses.mc_circuit_search import *
from functionsAndClasses.functions_file import *
from sympy.physics.paulialgebra import Pauli
from sympy.physics.quantum import TensorProduct
import random
import matplotlib.pyplot as plt
import pickle

my_random = random.Random()
my_random.seed(10)

term = TensorProduct(Pauli(2), Pauli(2))
terms = [term]

final_instance, accepted_f_values, refused_f_values,\
    file_dir, figures_dir, data_path = mc_search(terms, my_random, temperature = 0.05, n_max = 20)
#
# print("\nTrying to maximize coeffients now:\n")
# final_H_pauli, coefficient, new_ratio, new_parameters_values,\
#     accepted_r_values, refused_r_values= extremize_coeff_stochastic(final_instance, terms, temperature = 0.008,
#                                                                     n_steps_extr = 1000, operation = "maximize")

plt.figure(1)

# plt.subplot(121)
plt.title("Circuit Manipulation")
plt.ylabel('f')
plt.xlabel('n')
plt.plot(accepted_f_values[0],accepted_f_values[1],'go',
         refused_f_values[0],refused_f_values[1],'ro')

# plt.subplot(122)
# plt.title("Parameter Manipulation")
# plt.ylabel('Ratio')
# plt.xlabel('n')
# plt.plot(accepted_r_values[0],accepted_r_values[1],'go',
#          refused_r_values[0],refused_r_values[1],'ro')

plt.savefig(figures_dir + 'f_values.pdf')
plt.tight_layout()
plt.show()

save_data = { "final_instance": final_instance, "accepted_f_values": accepted_f_values,
              "refused_f_values": refused_f_values}
             # "accepted_r_values": accepted_r_values,
             # "refused_r_values": refused_r_values,
             # "final_parameters_values": new_parameters_values}

with open(os.path.join(file_dir, 'mc_search_data.pickle'), 'wb') as file:
    pickle.dump(save_data, file)
