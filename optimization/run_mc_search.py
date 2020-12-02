import circuitq as cq
import numpy as np
import random
import matplotlib.pyplot as plt

my_random = random.Random()
my_random.seed(10)
nbr_circuit_variations = 20
nbr_parameter_variations = 20

winner_instance, plot_data, \
cost_contributions_list = cq.mc_search(300, nbr_circuit_variations,
                            nbr_parameter_variations, my_random,
                            "optimization", 10, max_edges=8, filter=1e30)

initial_cost, accepted_circuits, refused_circuits, \
accepted_parameters, refused_parameters, graph_list = plot_data

#%%
# # =============================================================================
# # Plot cost function variations and its contributions
# # =============================================================================
fig, axs = plt.subplots(1,2, figsize=(9,5))
axs[0].set_xlabel("Variation step")
axs[0].set_ylabel("cost")
axs[0].plot(initial_cost[0], initial_cost[1], 'D', markersize = 13)
axs[0].plot(accepted_circuits[0],accepted_circuits[1],'g*', markersize = 18)
axs[0].plot(refused_circuits[0],refused_circuits[1],'r*', markersize = 18)
axs[0].plot(accepted_parameters[0], accepted_parameters[1], 'go')
axs[0].plot(refused_parameters[0], refused_parameters[1], 'ro')
xticks = np.linspace(0, len(cost_contributions_list), 10)
axs[0].set_xticks([int(tick) for tick in xticks])
cost_contributions_list = np.array(cost_contributions_list)
axs[0].set_ylim((-200,200))
axs[1].set_xlabel("Variation step")
axs[1].set_ylabel("contributions")
labels = ["T1_quasiparticles", "T1_charge", "T1_flux", "T1_ges_scaled",
                          "anharmonicity_scaled", "nbr_edges_scaled"]
linestyles = ["--", "-.", "-.", ":", "--", "-." ]
for n,l in enumerate(labels):
    axs[1].plot(cost_contributions_list[:,n]/cost_contributions_list[0,n],
                label = l, linestyle = linestyles[n])
axs[1].set_ylim((0,20))
axs[1].legend()
axs[1].set_xticks([int(tick) for tick in xticks])
plt.savefig('/Users/philipp/Dropbox (Personal)/CircuitDesign/git/figures/'
            'mc_search_optimization_overview/cost_function.pdf')
plt.show()

#%%
# # =============================================================================
# # Plot circuits in one document
# # =============================================================================
import os
figure_path_list = []
directory, dirs, files = next(os.walk('/Users/philipp/Dropbox (Personal)/'
                                 'CircuitDesign/git/figures/mc_search_optimization'))
files = sorted(files)
for file_name in files:
    figure_path = os.path.join(directory, file_name)
    figure_path_list.append(figure_path)
    if file_name.startswith(str(winner_instance[-1])):
        winner_path = figure_path
nbr_figures = len(figure_path_list)
n_cols = 3
n_rows = int(np.ceil(nbr_figures/n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10,5*n_rows))
row, col = 0, 0
for ax in axs.ravel():
    ax.axis('off')
for n, figure_path in enumerate(figure_path_list):
    if col == n_cols:
        col = 0
        row += 1
    image = plt.imread(figure_path)
    axs[row,col].imshow(image, aspect='equal')
    axs[row, col].set_title(files[n])
    col += 1
plt.savefig('/Users/philipp/Dropbox (Personal)/CircuitDesign/git/figures/'
            'mc_search_optimization_overview/circuit_variation.pdf')
plt.show()


#%%
# # =============================================================================
# # Plot winner instance
# # =============================================================================
fig, axs = plt.subplots(1,2, figsize=(9,5))
axs[0].set_title("Winner circuit")
axs[0].axis('off')
winner_image = plt.imread(winner_path)
axs[0].imshow(winner_image, aspect='equal')
axs[1].set_title("Winner parameters")
winner_parameters = winner_instance[1]
winner_circuit = winner_instance[2]
default_circuit = cq.CircuitQ(winner_circuit)
default_circuit.get_numerical_hamiltonian(100, default_zero=False)
default_parameters = default_circuit.parameter_values
parameter_names = default_circuit.h_parameters
axs[1].plot(np.array(winner_parameters)/np.array(default_parameters),'go')
axs[1].set_xlabel("Parameters")
axs[1].set_ylabel("Ratio to default value")
axs[1].set_xticks(range(len(parameter_names)))
axs[1].set_xticklabels(parameter_names, rotation='vertical')
plt.tight_layout()
plt.savefig('/Users/philipp/Dropbox (Personal)/CircuitDesign/git/figures/'
            'mc_search_optimization_overview/winner_instance.pdf')
plt.show()

