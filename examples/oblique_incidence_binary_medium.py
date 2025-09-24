"""
Wavesim simulation: Oblique light beam through binary random refractive index pillars
===================================================================================
Simulates obliquely incident light passing through a 10×10×1 wavelength region
where each 1×1×1 wavelength pillar has a binary refractive index: either 1.0 or 2.2
(equivalent to a random mask where 0→n=1.0, 1→n=2.2).
"""

import torch
import numpy as np
from time import time
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess
from __init__ import plot

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
wavelength = 1.0  # wavelength in micrometers (um)
pixel_size = 0.25  # points per wavelength (4 pixels per wavelength)

# Grid size: 10×10×1 wavelengths = 40×40×4 pixels (since pixel_size = 0.25)
n_wavelengths = (10, 10, 1)  # size in wavelengths
n_size = tuple(int(w / pixel_size) for w in n_wavelengths)  # size in pixels
print(f"Grid size: {n_size} pixels ({n_wavelengths} wavelengths)")

# Generate binary random mask (10×10 pillars)
random_mask = np.random.randint(0, 2, size=(n_wavelengths[0], n_wavelengths[1], n_wavelengths[2]))
print(f"Random mask (10×10×1): \n{random_mask.squeeze()}")

# Convert mask to refractive indices: 0→n=1.0, 1→n=2.2
n_map = np.zeros(n_size, dtype=np.complex64)
pixels_per_wavelength = int(1 / pixel_size)  # 4 pixels per wavelength

for i in range(n_wavelengths[0]):
    for j in range(n_wavelengths[1]):
        for k in range(n_wavelengths[2]):
            # Binary refractive index: 1.0 or 2.2 based on mask
            n_value = 1.0 if random_mask[i, j, k] == 0 else 2.2

            # Fill the corresponding pixel region
            i_start, i_end = i * pixels_per_wavelength, (i + 1) * pixels_per_wavelength
            j_start, j_end = j * pixels_per_wavelength, (j + 1) * pixels_per_wavelength
            k_start, k_end = k * pixels_per_wavelength, (k + 1) * pixels_per_wavelength

            n_map[i_start:i_end, j_start:j_end, k_start:k_end] = n_value

print(f"Refractive indices: {np.unique(n_map.real)} (binary: 1.0 and 2.2)")
print(f"Number of n=1.0 pillars: {np.sum(random_mask == 0)}")
print(f"Number of n=2.2 pillars: {np.sum(random_mask == 1)}")

# Add boundaries and convert to permittivity
boundary_widths = 8  # width of the boundary in pixels
n_squared, boundary_array = preprocess(n_map**2, boundary_widths)  # permittivity is n²

# Create oblique incident wave source
# For oblique incidence, we create a plane wave at an angle
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = np.zeros(n_ext, dtype=np.complex64)

# Create oblique plane wave (e.g., 30 degree angle in x-z plane)
angle_deg = 30  # degrees
angle_rad = np.radians(angle_deg)
kx = (2 * np.pi / wavelength) * np.sin(angle_rad) * pixel_size
kz = (2 * np.pi / wavelength) * np.cos(angle_rad) * pixel_size

# Create plane wave across the input face (z=0 boundary)
for i in range(n_ext[0]):
    for j in range(n_ext[1]):
        x_coord = (i - boundary_array[0]) * pixel_size * wavelength
        y_coord = (j - boundary_array[1]) * pixel_size * wavelength
        # Plane wave with oblique incidence
        phase = kx * x_coord
        source[i, j, boundary_array[2]] = np.exp(1j * phase)

source = torch.tensor(source, dtype=torch.complex64)

print(f"Created oblique source at {angle_deg}° angle")
print(f"Source amplitude: {torch.abs(source).max().item():.3f}")

# Set up the domain
periodic = (False, False, False)  # non-periodic boundaries for oblique incidence
domain = HelmholtzDomain(
    permittivity=n_squared,
    periodic=periodic,
    pixel_size=pixel_size,
    wavelength=wavelength,
    n_boundary=8
)

print(f"Domain setup complete. Starting simulation...")

# Run the simulation
start = time()
u_computed, iterations, residual_norm = run_algorithm(
    domain,
    source,
    max_iterations=2000,
    threshold=1e-6
)
end = time() - start

print(f'\nSimulation completed!')
print(f'Time: {end:.2f} s')
print(f'Iterations: {iterations}')
print(f'Residual norm: {residual_norm:.3e}')

# Extract the field from the boundary region
field_result = u_computed.squeeze().cpu().numpy()
# Remove boundary regions to get the field in the ROI
roi_slices = tuple(slice(boundary_widths, -boundary_widths) for boundary_widths in boundary_array)
field_roi = field_result[roi_slices]

print(f'Output field shape: {field_roi.shape}')
print(f'Field amplitude range: {np.abs(field_roi).min():.3e} to {np.abs(field_roi).max():.3e}')

# The field_roi now contains the wave field after passing through the binary random medium
# You can analyze transmission, reflection, scattering patterns, etc.

# Optional: Plot results if plotting is available
try:
    plot(field_roi, None, None)
except:
    print("Plotting not available, but simulation completed successfully")