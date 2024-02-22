import numpy as np
import logging
from qiskit import Aer
from qiskit.utils import algorithm_globals
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization import QuadraticProgram
from qiskit.utils import QuantumInstance
from docplex.mp.model import Model
from qiskit.quantum_info import SparsePauliOp

# I could run this on IBM Cloud or AWS Braket but I want to confirm feasibility with Qiskit 0.46. I need to fix dependency issues first.

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the problem
num_items = 10
max_value = 100
max_weight = 40
max_volume = 20
knapsack_max_weight = 100
knapsack_max_volume = 60

# Seed for reproducibility
algorithm_globals.random_seed = 123
np.random.seed(123)

# Generate items
items = [(np.random.randint(1, max_value + 1), np.random.randint(1, max_weight + 1), np.random.randint(1, max_volume + 1)) for _ in range(num_items)]

# Define the problem with Docplex and adds constraints and maximization objective (within the Knapsack)
mdl = Model("CubicKnapsack")
x = [mdl.binary_var(name=f"x{i}") for i in range(len(items))]
mdl.maximize(mdl.sum(x[i] * items[i][0] for i in range(len(items))))
mdl.add_constraint(mdl.sum(x[i] * items[i][1] for i in range(len(items))) <= 100)
mdl.add_constraint(mdl.sum(x[i] * items[i][2] for i in range(len(items))) <= 60)

# Convert to Quadratic Program
qp = from_docplex_mp(mdl)

# Setup QuantumInstance and optimizer with a ASR Simulator Backend.
backend = Aer.get_backend('aer_simulator_statevector')
quantum_instance = QuantumInstance(backend=backend, shots=1024, seed_simulator=123, seed_transpiler=123)

# Setup and solve the problem using QAOA
optimizer = COBYLA() # I could use SPSA or other optimizers but COYBLA is a good start.
qaoa = QAOA(optimizer=optimizer, reps=1, quantum_instance=quantum_instance)

optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

# Print the results
logging.info(f"Selected Items: {result.x}")
logging.info(f"Total Value: {result.fval}")
