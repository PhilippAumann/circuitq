import circuitq as cq

import random


my_random = random.Random()
my_random.seed(10)

winner_instance = cq.mc_search(100, 40, 20, my_random, 'test', temperature=0.01)
