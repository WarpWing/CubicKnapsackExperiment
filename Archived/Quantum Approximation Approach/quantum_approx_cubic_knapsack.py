import numpy as np
import logging
import qiskit_algorithms
from qiskit import *
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.utils import algorithm_globals
from qiskit.utils import QuantumInstance
from docplex.mp.model import Model

# I could run this on IBM Cloud or AWS Braket but I want to confirm feasibility with Qiskit 0.46. I need to fix dependency issues first.


logging.basicConfig(level=logging.INFO)

# Define the problem (max_value, max_weight, max_volume) and the knapsack constraints (knapsack_max_weight, knapsack_max_volume)
num_items = 10
max_value = 100 
max_weight = 40
max_volume = 20
knapsack_max_weight = 100
knapsack_max_volume = 60

# Global seed for reproducibility
algorithm_globals.random_seed = 123
np.random.seed(123)

# Generate items with random values, weights, and volumes (I had to massively scale down search space for demonstration)
items = [(np.random.randint(1, max_value + 1), np.random.randint(1, max_weight + 1), np.random.randint(1, max_volume + 1)) for _ in range(num_items)]

# Define the problem with Docplex and adds constraints and maximization objective (within the Knapsack)
mdl = Model("CubicKnapsack")
x = [mdl.binary_var(name=f"x{i}") for i in range(len(items))]
mdl.maximize(mdl.sum(x[i] * items[i][0] for i in range(len(items))))
mdl.add_constraint(mdl.sum(x[i] * items[i][1] for i in range(len(items))) <= knapsack_max_weight)
mdl.add_constraint(mdl.sum(x[i] * items[i][2] for i in range(len(items))) <= knapsack_max_volume)


# Convert the problem to a Quadratic Program and solve it using QAOA
qp = from_docplex_mp(mdl)
backend = Aer.get_backend('aer_simulator_statevector')
quantum_instance = QuantumInstance(backend=backend, shots=2048, seed_simulator=algorithm_globals.random_seed, seed_transpiler=algorithm_globals.random_seed)

# Setup and solve the problem using QAOA
optimizer = COBYLA() 
qaoa = QAOA(optimizer=optimizer, reps=1, quantum_instance=quantum_instance)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

# Result printing
logging.info(f"Selected Items: {result.x}")
logging.info(f"Total Value: {result.fval}")
