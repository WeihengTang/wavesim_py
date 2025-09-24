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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
pixel_size = 0.1  # points per wavelength (10 pixels per wavelength for higher resolution)

# Grid size: 10×10×1 wavelengths + 0.5λ air layer = 100×100×15 pixels (since pixel_size = 0.1)
air_layer_thickness = 0.5  # wavelengths of air before the medium
n_wavelengths_medium = (10, 10, 1)  # size of the random medium
n_wavelengths_total = (n_wavelengths_medium[0], n_wavelengths_medium[1], n_wavelengths_medium[2] + air_layer_thickness)
n_size = tuple(int(w / pixel_size) for w in n_wavelengths_total)  # total size in pixels
air_pixels = int(air_layer_thickness / pixel_size)  # pixels for air layer
medium_pixels = int(n_wavelengths_medium[2] / pixel_size)  # pixels for medium
print(f"Total grid size: {n_size} pixels ({n_wavelengths_total} wavelengths)")
print(f"Air layer: {air_pixels} pixels ({air_layer_thickness} wavelengths)")
print(f"Medium: {medium_pixels} pixels ({n_wavelengths_medium[2]} wavelengths)")

# Generate binary random mask (10×10 pillars)
random_mask = np.random.randint(0, 2, size=(n_wavelengths[0], n_wavelengths[1], n_wavelengths[2]))
print(f"Random mask (10×10×1): \n{random_mask.squeeze()}")

# Convert mask to refractive indices: 0→n=1.0, 1→n=2.2
n_map = np.ones(n_size, dtype=np.complex64)  # Start with air (n=1.0) everywhere
pixels_per_wavelength = int(1 / pixel_size)  # 10 pixels per wavelength

# Fill only the medium region (after the air layer)
for i in range(n_wavelengths_medium[0]):
    for j in range(n_wavelengths_medium[1]):
        for k in range(n_wavelengths_medium[2]):
            # Binary refractive index: 1.0 or 2.2 based on mask
            n_value = 1.0 if random_mask[i, j, k] == 0 else 2.2

            # Fill the corresponding pixel region (offset by air layer)
            i_start, i_end = i * pixels_per_wavelength, (i + 1) * pixels_per_wavelength
            j_start, j_end = j * pixels_per_wavelength, (j + 1) * pixels_per_wavelength
            k_start, k_end = air_pixels + k * pixels_per_wavelength, air_pixels + (k + 1) * pixels_per_wavelength

            n_map[i_start:i_end, j_start:j_end, k_start:k_end] = n_value

print(f"Refractive indices: {np.unique(n_map.real)} (binary: 1.0 and 2.2)")
print(f"Number of n=1.0 pillars: {np.sum(random_mask == 0)}")
print(f"Number of n=2.2 pillars: {np.sum(random_mask == 1)}")
print(f"Pixels per wavelength: {pixels_per_wavelength}")

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

# Create plane wave across the input face (z=0, at the beginning of air layer)
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

# Enhanced 2D Visualization using slices
def create_visualizations(field_3d, n_map_roi, random_mask):
    """Create multiple 2D visualizations of the simulation results"""

    # Create output directory
    import os
    output_dir = 'oblique_incidence_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Calculate field properties
    intensity = np.abs(field_3d)**2
    phase = np.angle(field_3d)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Random mask pattern (top-view)
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(random_mask.squeeze(), cmap='binary', origin='lower')
    ax1.set_title('Random Binary Mask\n(0=n1.0, 1=n2.2)')
    ax1.set_xlabel('Y (wavelengths)')
    ax1.set_ylabel('X (wavelengths)')
    plt.colorbar(im1, ax=ax1)

    # 2. Refractive index cross-section (X-Z plane, middle Y)
    y_mid = n_map_roi.shape[1] // 2
    ax2 = plt.subplot(3, 4, 2)
    n_xz = n_map_roi[:, y_mid, :].real
    im2 = ax2.imshow(n_xz.T, cmap='viridis', origin='lower', aspect='equal',
                     extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax2.set_title('Refractive Index (X-Z)')
    ax2.set_xlabel('X (wavelengths)')
    ax2.set_ylabel('Z (wavelengths)')
    # Add vertical line to show medium interface
    ax2.axhline(y=air_layer_thickness, color='red', linestyle='--', alpha=0.7, label='Medium interface')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Refractive Index')

    # 3. Field intensity - X-Z plane (middle Y)
    ax3 = plt.subplot(3, 4, 3)
    intensity_xz = intensity[:, y_mid, :]
    im3 = ax3.imshow(intensity_xz.T, cmap='hot', origin='lower', aspect='equal',
                     extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax3.set_title('Intensity |E|² (X-Z)')
    ax3.set_xlabel('X (wavelengths)')
    ax3.set_ylabel('Z (wavelengths)')
    # Add vertical line to show medium interface
    ax3.axhline(y=air_layer_thickness, color='cyan', linestyle='--', alpha=0.7, label='Medium interface')
    ax3.legend()
    plt.colorbar(im3, ax=ax3, label='Intensity')

    # 4. Field phase - X-Z plane (middle Y)
    ax4 = plt.subplot(3, 4, 4)
    phase_xz = phase[:, y_mid, :]
    im4 = ax4.imshow(phase_xz.T, cmap='hsv', origin='lower', aspect='equal',
                     extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]],
                     vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Phase arg(E) (X-Z)')
    ax4.set_xlabel('X (wavelengths)')
    ax4.set_ylabel('Z (wavelengths)')
    # Add vertical line to show medium interface
    ax4.axhline(y=air_layer_thickness, color='white', linestyle='--', alpha=0.7, label='Medium interface')
    ax4.legend()
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')

    # 5. Field intensity - X-Y plane (input face, z=0)
    ax5 = plt.subplot(3, 4, 5)
    intensity_xy_input = intensity[:, :, 0]
    im5 = ax5.imshow(intensity_xy_input.T, cmap='hot', origin='lower',
                     extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[1]])
    ax5.set_title('Incident Beam (X-Y, z=0)')
    ax5.set_xlabel('X (wavelengths)')
    ax5.set_ylabel('Y (wavelengths)')
    plt.colorbar(im5, ax=ax5, label='Intensity')

    # 6. Field intensity - X-Y plane (output face, z=end)
    ax6 = plt.subplot(3, 4, 6)
    intensity_xy_output = intensity[:, :, -1]
    im6 = ax6.imshow(intensity_xy_output.T, cmap='hot', origin='lower',
                     extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[1]])
    ax6.set_title(f'Output Intensity (X-Y, z={n_wavelengths_total[2]}λ)')
    ax6.set_xlabel('X (wavelengths)')
    ax6.set_ylabel('Y (wavelengths)')
    plt.colorbar(im6, ax=ax6, label='Intensity')

    # 7. Field intensity - Y-Z plane (middle X)
    x_mid = intensity.shape[0] // 2
    ax7 = plt.subplot(3, 4, 7)
    intensity_yz = intensity[x_mid, :, :]
    im7 = ax7.imshow(intensity_yz.T, cmap='hot', origin='lower', aspect='equal',
                     extent=[0, n_wavelengths_total[1], 0, n_wavelengths_total[2]])
    ax7.set_title('Intensity |E|² (Y-Z)')
    ax7.set_xlabel('Y (wavelengths)')
    ax7.set_ylabel('Z (wavelengths)')
    # Add vertical line to show medium interface
    ax7.axhline(y=air_layer_thickness, color='cyan', linestyle='--', alpha=0.7, label='Medium interface')
    ax7.legend()
    plt.colorbar(im7, ax=ax7, label='Intensity')

    # 8. Transmission analysis - intensity along Z
    ax8 = plt.subplot(3, 4, 8)
    z_coords = np.linspace(0, n_wavelengths_total[2], intensity.shape[2])
    avg_intensity_z = np.mean(intensity, axis=(0, 1))
    max_intensity_z = np.max(intensity, axis=(0, 1))
    min_intensity_z = np.min(intensity, axis=(0, 1))

    ax8.plot(z_coords, avg_intensity_z, 'b-', linewidth=2, label='Average')
    ax8.fill_between(z_coords, min_intensity_z, max_intensity_z, alpha=0.3, label='Min-Max Range')
    ax8.axvline(x=air_layer_thickness, color='red', linestyle='--', alpha=0.7, label='Medium interface')
    ax8.set_xlabel('Z (wavelengths)')
    ax8.set_ylabel('Intensity')
    ax8.set_title('Intensity vs Propagation Distance')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # 9. Scattering pattern - final intensity distribution
    ax9 = plt.subplot(3, 4, 9)
    final_intensity = intensity[:, :, -1]
    im9 = ax9.imshow(final_intensity.T, cmap='plasma', origin='lower',
                     extent=[0, n_wavelengths[0], 0, n_wavelengths[1]])
    ax9.set_title('Transmitted Beam Pattern')
    ax9.set_xlabel('X (wavelengths)')
    ax9.set_ylabel('Y (wavelengths)')
    plt.colorbar(im9, ax=ax9, label='Intensity')

    # 10. Real part of field - X-Z plane
    ax10 = plt.subplot(3, 4, 10)
    real_xz = field_3d[:, y_mid, :].real
    im10 = ax10.imshow(real_xz.T, cmap='RdBu', origin='lower', aspect='equal',
                       extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax10.set_title('Re(E) Wave Pattern (X-Z)')
    ax10.set_xlabel('X (wavelengths)')
    ax10.set_ylabel('Z (wavelengths)')
    # Add vertical line to show medium interface
    ax10.axhline(y=air_layer_thickness, color='black', linestyle='--', alpha=0.7, label='Medium interface')
    ax10.legend()
    plt.colorbar(im10, ax=ax10, label='Re(E)')

    # 11. Cross-sectional intensity profiles
    ax11 = plt.subplot(3, 4, 11)
    x_coords = np.linspace(0, n_wavelengths_total[0], intensity.shape[0])
    y_coords = np.linspace(0, n_wavelengths_total[1], intensity.shape[1])

    # Intensity profiles at different z positions
    air_interface_idx = int(air_layer_thickness * intensity.shape[2] / n_wavelengths_total[2])
    z_positions = [0, air_interface_idx//2, air_interface_idx, (air_interface_idx + intensity.shape[2])//2, intensity.shape[2]-1]
    z_values = [z_pos * n_wavelengths_total[2] / intensity.shape[2] for z_pos in z_positions]
    z_labels = [f'z={z_val:.2f}λ' for z_val in z_values]

    for i, (z_pos, z_label) in enumerate(zip(z_positions, z_labels)):
        profile = intensity[:, y_mid, z_pos]
        ax11.plot(x_coords, profile, linewidth=2, label=z_label)

    ax11.set_xlabel('X (wavelengths)')
    ax11.set_ylabel('Intensity')
    ax11.set_title('X-profiles at Different Z')
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    # 12. Phase evolution
    ax12 = plt.subplot(3, 4, 12)
    phase_evolution = np.mean(phase, axis=(0, 1))
    ax12.plot(z_coords, phase_evolution, 'r-', linewidth=2)
    ax12.axvline(x=air_layer_thickness, color='black', linestyle='--', alpha=0.7, label='Medium interface')
    ax12.set_xlabel('Z (wavelengths)')
    ax12.set_ylabel('Average Phase (rad)')
    ax12.set_title('Phase Evolution')
    ax12.grid(True, alpha=0.3)
    ax12.legend()

    plt.tight_layout()
    overview_path = os.path.join(output_dir, 'overview_all_plots.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Save individual plots
    print(f"Saving individual plots to {output_dir}/...")

    # 1. Random mask
    fig1, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(random_mask.squeeze(), cmap='binary', origin='lower')
    ax.set_title('Random Binary Mask (0=n1.0, 1=n2.2)', fontsize=14)
    ax.set_xlabel('Y (wavelengths)')
    ax.set_ylabel('X (wavelengths)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_random_mask.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Refractive index X-Z
    fig2, ax = plt.subplots(figsize=(12, 6))
    n_xz = n_map_roi[:, y_mid, :].real
    im = ax.imshow(n_xz.T, cmap='viridis', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax.set_title('Refractive Index Cross-section (X-Z plane)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Z (wavelengths)')
    ax.axhline(y=air_layer_thickness, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Air-Medium Interface')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Refractive Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_refractive_index_xz.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Intensity X-Z
    fig3, ax = plt.subplots(figsize=(12, 6))
    intensity_xz = intensity[:, y_mid, :]
    im = ax.imshow(intensity_xz.T, cmap='hot', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax.set_title('Field Intensity |E|² (X-Z plane)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Z (wavelengths)')
    ax.axhline(y=air_layer_thickness, color='cyan', linestyle='--', alpha=0.8, linewidth=2, label='Air-Medium Interface')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_intensity_xz.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Phase X-Z
    fig4, ax = plt.subplots(figsize=(12, 6))
    phase_xz = phase[:, y_mid, :]
    im = ax.imshow(phase_xz.T, cmap='hsv', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]],
                   vmin=-np.pi, vmax=np.pi)
    ax.set_title('Field Phase arg(E) (X-Z plane)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Z (wavelengths)')
    ax.axhline(y=air_layer_thickness, color='white', linestyle='--', alpha=0.8, linewidth=2, label='Air-Medium Interface')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_phase_xz.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Incident beam (in air, before medium)
    fig5, ax = plt.subplots(figsize=(8, 8))
    intensity_xy_input = intensity[:, :, 0]
    im = ax.imshow(intensity_xy_input.T, cmap='hot', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[1]])
    ax.set_title('Incident Beam in Air (X-Y, z=0)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Y (wavelengths)')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_incident_beam.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Output beam
    fig6, ax = plt.subplots(figsize=(8, 8))
    intensity_xy_output = intensity[:, :, -1]
    im = ax.imshow(intensity_xy_output.T, cmap='hot', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[1]])
    ax.set_title(f'Transmitted Beam (X-Y, z={n_wavelengths_total[2]}λ)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Y (wavelengths)')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_transmitted_beam.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Real field X-Z
    fig7, ax = plt.subplots(figsize=(12, 6))
    real_xz = field_3d[:, y_mid, :].real
    im = ax.imshow(real_xz.T, cmap='RdBu', origin='lower', aspect='equal',
                   extent=[0, n_wavelengths_total[0], 0, n_wavelengths_total[2]])
    ax.set_title('Real Field Re(E) Wave Pattern (X-Z)', fontsize=14)
    ax.set_xlabel('X (wavelengths)')
    ax.set_ylabel('Z (wavelengths)')
    ax.axhline(y=air_layer_thickness, color='black', linestyle='--', alpha=0.8, linewidth=2, label='Air-Medium Interface')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Re(E)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_real_field_xz.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Transmission analysis
    fig8, ax = plt.subplots(figsize=(12, 6))
    z_coords = np.linspace(0, n_wavelengths_total[2], intensity.shape[2])
    avg_intensity_z = np.mean(intensity, axis=(0, 1))
    max_intensity_z = np.max(intensity, axis=(0, 1))
    min_intensity_z = np.min(intensity, axis=(0, 1))
    ax.plot(z_coords, avg_intensity_z, 'b-', linewidth=2, label='Average Intensity')
    ax.fill_between(z_coords, min_intensity_z, max_intensity_z, alpha=0.3, label='Min-Max Range')
    ax.axvline(x=air_layer_thickness, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Air-Medium Interface')
    ax.set_xlabel('Z (wavelengths)')
    ax.set_ylabel('Intensity')
    ax.set_title('Intensity vs Propagation Distance', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Add text annotations
    ax.text(air_layer_thickness/2, ax.get_ylim()[1]*0.9, 'Air Layer\n(Incident Wave)',
            ha='center', va='top', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(air_layer_thickness + (n_wavelengths_total[2]-air_layer_thickness)/2, ax.get_ylim()[1]*0.9,
            'Random Medium\n(Scattering)', ha='center', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_transmission_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print some analysis results
    print(f"\n--- Scattering Analysis ---")
    incident_power = np.sum(intensity[:, :, 0])
    transmitted_power = np.sum(intensity[:, :, -1])
    transmission_ratio = transmitted_power / incident_power

    print(f"Incident power: {incident_power:.3f}")
    print(f"Transmitted power: {transmitted_power:.3f}")
    print(f"Transmission ratio: {transmission_ratio:.3f}")
    print(f"Max intensity enhancement: {np.max(intensity) / np.max(intensity[:, :, 0]):.3f}")

    # Beam spreading analysis
    def beam_width(intensity_2d):
        center_y = intensity_2d.shape[1] // 2
        x_profile = intensity_2d[:, center_y]
        total_power = np.sum(x_profile)
        if total_power > 0:
            x_coords = np.arange(len(x_profile))
            centroid = np.sum(x_coords * x_profile) / total_power
            variance = np.sum((x_coords - centroid)**2 * x_profile) / total_power
            return 2 * np.sqrt(variance) * pixel_size * wavelength  # FWHM approximation
        return 0

    input_width = beam_width(intensity[:, :, 0])
    output_width = beam_width(intensity[:, :, -1])
    print(f"Input beam width: {input_width:.3f} wavelengths")
    print(f"Output beam width: {output_width:.3f} wavelengths")
    print(f"Beam spreading factor: {output_width/input_width:.3f}")

# Remove boundaries to get ROI-only field and refractive index
field_roi_only = field_roi
n_map_roi = n_map  # Original n_map without boundaries

print("\nGenerating visualizations...")
create_visualizations(field_roi_only, n_map_roi, random_mask)
print("Visualizations saved in 'oblique_incidence_visualizations/' folder")