import sys
sys.path.insert(0, './')
from nn_cno import ode
import numpy as np
import itertools
import jax.numpy as jnp
import diffrax


# load the network 
c = ode.logicODE("./nn_cno/datasets/working_case_study/PKN-test.sif",
    "./nn_cno/datasets/working_case_study/MD-test.csv")

c.preprocessing(expansion=False)


import time
start_time = time.time()

c.simulate()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

for _ in range(1):
    c.simulate()
print("--- %s seconds ---" % (time.time() - start_time))

