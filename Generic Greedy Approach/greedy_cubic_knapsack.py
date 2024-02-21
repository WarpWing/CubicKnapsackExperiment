import time
import numpy as np

def greedy_cubic_knapsack(values, weights, volumes, max_weight_capacity, max_volume_capacity):

    # Combine item attributes and sort them based on efficiency (value-to-weight-volume ratio). I would love to look into Dynamic Metrics that scale with current state.
    items_with_ratios = [(index, value, weight, volume, value / (weight * volume)) for index, (value, weight, volume) in enumerate(zip(values, weights, volumes))]
    items_with_ratios.sort(key=lambda item: item[4], reverse=True)  # Sort items by their efficiency

    total_value,current_weight,current_volume = 0,0,0
    selected_items = []

    # Select items based on their sorted efficiency until the knapsack's capacity is reached.
    for index, value, weight, volume, _ in items_with_ratios:
        if current_weight + weight <= max_weight_capacity and current_volume + volume <= max_volume_capacity:
            selected_items.append(index)
            total_value += value
            current_weight += weight
            current_volume += volume

    return total_value, selected_items

# Setup for a more complex test case. Hopefully I didn't overcomment this for the sake of clarity.

np.random.seed(500000) #This sets a random seed for reproducibility. I should abstract this to a unit test.
num_items = 10000
max_value = 2000 
max_weight = 500 
max_volume = 400 

# Generate item attributes
values_large = np.random.randint(1, max_value, num_items).tolist()
weights_large = np.random.randint(1, max_weight, num_items).tolist()
volumes_large = np.random.randint(1, max_volume, num_items).tolist()
max_weight_capacity_large = 100000
max_volume_capacity_large = 80000

# Measure execution time
start_time = time.perf_counter()
solution_large = greedy_cubic_knapsack(values_large, weights_large, volumes_large, max_weight_capacity_large, max_volume_capacity_large)
end_time = time.perf_counter()


# Print initial problem setup for clarity sake.
print("Number of Items:", num_items)
print("Maximum Weight Capacity of the Knapsack:", max_weight_capacity_large)
print("Maximum Volume Capacity of the Knapsack:", max_volume_capacity_large)
print()


# Print the details of the selected items sequentially.
print("Selected Items Details:")
sorted_selected_items = sorted(solution_large[1])
total_selected_weight = sum(weights_large[idx] for idx in sorted_selected_items)
total_selected_volume = sum(volumes_large[idx] for idx in sorted_selected_items)
for idx in sorted_selected_items:
    print(f"Item {idx}: Value = {values_large[idx]}, Weight = {weights_large[idx]}, Volume = {volumes_large[idx]}")

# Print summary.
print("\nSummary of Selected Items:")
print(f"Total Value Achieved: {solution_large[0]}")
print(f"Total Weight of Selected Items: {total_selected_weight} (out of {max_weight_capacity_large})")
print(f"Total Volume of Selected Items: {total_selected_volume} (out of {max_volume_capacity_large})")
print(f"Time Taken for Problem (seconds): {end_time - start_time:.10f}")


