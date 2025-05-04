import numpy as np
import os

# Simulate example landmark data for A, B, and C
np.random.seed(42)
example_data = {
    'A': [np.random.rand(63) for _ in range(5)],
    'B': [np.random.rand(63) for _ in range(5)],
    'C': [np.random.rand(63) for _ in range(5)]
}

# Save as .npy files
output_dir = "asl_example_dataset"
os.makedirs(output_dir, exist_ok=True)

for label, samples in example_data.items():
    for i, sample in enumerate(samples):
        filename = f"{label}_{i}.npy"
        path = os.path.join(output_dir, filename)
        np.save(path, sample)

output_dir
