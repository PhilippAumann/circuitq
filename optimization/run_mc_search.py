import circuitq as cq
import numpy as np
import random
import matplotlib.pyplot as plt

my_random = random.Random()
my_random.seed(10)
nbr_circuit_variations = 20
nbr_parameter_variations = 50

winner_instance, plot_data, \
cost_contributions_list = cq.mc_search(300, nbr_circuit_variations,
                            nbr_parameter_variations, my_random,
                            "optimization", 10, max_edges=8, filter=1e30)

initial_cost, accepted_circuits, refused_circuits, \
accepted_parameters, refused_parameters = plot_data

#%%
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

plt.show()