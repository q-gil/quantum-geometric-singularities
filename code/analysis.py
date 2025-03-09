"""
Analysis and Visualization Tools for Quantum Geometric Singularities

This module provides analysis and visualization tools for the research presented in
"Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks"
by Hamid Bahri (2025).

These tools are used to analyze the effects of measurements on emergent geometry,
particularly focusing on the formation of geometric tears at the critical point γ = -√(μ0μ1).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

from tensor_network import TensorNetwork, create_measurement_operator, critical_gamma
from measurement_operators import MeasurementOperator, scan_gamma_values, scan_phase_space
from geometric_measures import GeometricMeasures

def analyze_critical_point(mu0: float = 0.5, 
                          gamma_range: Tuple[float, float] = (-0.5, 0.5), 
                          num_points: int = 100,
                          save_results: bool = False,
                          output_dir: str = './data/critical_point_scans/') -> Dict[str, Any]:
    """
    Analyze the behavior of the system near the critical point.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        gamma_range: Range of gamma values to scan (min, max)
        num_points: Number of points in the scan
        save_results: Whether to save the results to files
        output_dir: Directory to save results in (if save_results is True)
        
    Returns:
        Dictionary containing the analysis results
    """
    # Calculate critical gamma value
    mu1 = 1.0 - mu0
    gamma_c = critical_gamma(mu0, mu1)
    
    # Create a list of gamma values to scan
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_points)
    
    # Arrays to store results
    mutual_info_values = np.zeros(num_points)
    distance_values = np.zeros(num_points)
    schmidt_gap_values = np.zeros(num_points)
    curvature_values = np.zeros(num_points)
    
    # Analyze each gamma value
    for i, gamma in enumerate(gamma_values):
        # Skip if gamma is beyond the allowed range
        if abs(gamma) > np.sqrt(mu0 * mu1) + 1e-10:
            mutual_info_values[i] = np.nan
            distance_values[i] = np.nan
            schmidt_gap_values[i] = np.nan
            curvature_values[i] = np.nan
            continue
        
        # Create measurement operator
        M = create_measurement_operator(mu0, mu1, gamma)
        
        # Create tensor network and apply measurement
        network = TensorNetwork()
        network.apply_measurement(M)
        
        # Calculate mutual information between adjacent qubits
        mutual_info = network.mutual_information([0], [1])
        mutual_info_values[i] = mutual_info
        
        # Calculate distance
        if mutual_info > 0:
            distance = -np.log2(mutual_info / (2 * np.log2(2)))
        else:
            distance = float('inf')
        distance_values[i] = distance
        
        # Calculate Schmidt gap
        schmidt_coeffs = network.entanglement_spectrum([0, 1], [2, 3])
        if len(schmidt_coeffs) >= 2:
            schmidt_gap = schmidt_coeffs[0] - schmidt_coeffs[1]
        else:
            schmidt_gap = 0
        schmidt_gap_values[i] = schmidt_gap
        
        # Calculate quantum curvature
        measures = GeometricMeasures(network)
        curvature = measures.quantum_curvature_vector()[0]
        curvature_values[i] = curvature
    
    # Compile results
    results = {
        'mu0': mu0,
        'mu1': mu1,
        'gamma_c': gamma_c,
        'gamma_values': gamma_values,
        'mutual_info_values': mutual_info_values,
        'distance_values': distance_values,
        'schmidt_gap_values': schmidt_gap_values,
        'curvature_values': curvature_values
    }
    
    # Save results if requested
    if save_results:
        import os
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to a pickle file
        output_file = os.path.join(output_dir, f'critical_scan_mu0_{mu0:.2f}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def analyze_entanglement_spectrum(mu0: float = 0.5, 
                                gamma_range: Tuple[float, float] = (-0.5, 0.5), 
                                num_points: int = 100,
                                save_results: bool = False,
                                output_dir: str = './data/entanglement_spectra/') -> Dict[str, Any]:
    """
    Analyze the entanglement spectrum near the critical point.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        gamma_range: Range of gamma values to scan (min, max)
        num_points: Number of points in the scan
        save_results: Whether to save the results to files
        output_dir: Directory to save results in (if save_results is True)
        
    Returns:
        Dictionary containing the analysis results
    """
    # Calculate critical gamma value
    mu1 = 1.0 - mu0
    gamma_c = critical_gamma(mu0, mu1)
    
    # Create a list of gamma values to scan
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_points)
    
    # Arrays to store results
    eigenvalues = []
    
    # Analyze each gamma value
    for i, gamma in enumerate(gamma_values):
        # Skip if gamma is beyond the allowed range
        if abs(gamma) > np.sqrt(mu0 * mu1) + 1e-10:
            eigenvalues.append(None)
            continue
        
        # Create measurement operator
        M = create_measurement_operator(mu0, mu1, gamma)
        
        # Create tensor network and apply measurement
        network = TensorNetwork()
        network.apply_measurement(M)
        
        # Calculate entanglement spectrum for a bipartition of the ring
        spectrum = network.entanglement_spectrum([0, 1], [2, 3])
        eigenvalues.append(spectrum)
    
    # Compile results
    results = {
        'mu0': mu0,
        'mu1': mu1,
        'gamma_c': gamma_c,
        'gamma_values': gamma_values,
        'eigenvalues': eigenvalues
    }
    
    # Save results if requested
    if save_results:
        import os
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to a pickle file
        output_file = os.path.join(output_dir, f'entanglement_spectrum_mu0_{mu0:.2f}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def analyze_perturbation(mu0: float = 0.5, 
                        epsilon_range: Tuple[float, float] = (0.001, 0.1), 
                        num_points: int = 10,
                        log_scale: bool = True,
                        save_results: bool = False,
                        output_dir: str = './data/perturbation_analysis/') -> Dict[str, Any]:
    """
    Analyze perturbations around the critical point.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        epsilon_range: Range of epsilon values to scan (min, max)
        num_points: Number of points in the scan
        log_scale: Whether to use logarithmic spacing for epsilon values
        save_results: Whether to save the results to files
        output_dir: Directory to save results in (if save_results is True)
        
    Returns:
        Dictionary containing the analysis results
    """
    # Calculate critical gamma value
    mu1 = 1.0 - mu0
    gamma_c = critical_gamma(mu0, mu1)
    
    # Create a list of epsilon values to scan
    if log_scale:
        epsilon_values = np.logspace(np.log10(epsilon_range[0]), np.log10(epsilon_range[1]), num_points)
    else:
        epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_points)
    
    # Arrays to store results
    mutual_info_values = np.zeros(num_points)
    log_epsilon_values = np.zeros(num_points)
    log_mutual_info_values = np.zeros(num_points)
    
    # Analyze each epsilon value
    for i, epsilon in enumerate(epsilon_values):
        # Calculate gamma value
        gamma = gamma_c + epsilon
        
        # Create measurement operator
        M = create_measurement_operator(mu0, mu1, gamma)
        
        # Create tensor network and apply measurement
        network = TensorNetwork()
        network.apply_measurement(M)
        
        # Calculate mutual information between adjacent qubits
        mutual_info = network.mutual_information([0], [1])
        mutual_info_values[i] = mutual_info
        
        # Calculate logarithms for log-log plot
        if epsilon > 0 and mutual_info > 0:
            log_epsilon_values[i] = np.log(epsilon)
            log_mutual_info_values[i] = np.log(mutual_info)
        else:
            log_epsilon_values[i] = np.nan
            log_mutual_info_values[i] = np.nan
    
    # Compile results
    results = {
        'mu0': mu0,
        'mu1': mu1,
        'gamma_c': gamma_c,
        'epsilon_values': epsilon_values,
        'mutual_info_values': mutual_info_values,
        'log_epsilon_values': log_epsilon_values,
        'log_mutual_info_values': log_mutual_info_values
    }
    
    # Save results if requested
    if save_results:
        import os
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to a pickle file
        output_file = os.path.join(output_dir, f'perturbation_analysis_mu0_{mu0:.2f}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def verify_conservation_law(mu0: float = 0.5, 
                          gamma_values: List[float] = None,
                          save_results: bool = False,
                          output_dir: str = './data/conservation_law/') -> Dict[str, Any]:
    """
    Verify the conservation law for the quantity Q = κ_quantum(i) × exp(S(P|X)/S_max).
    
    Args:
        mu0: Probability of projecting onto |0⟩
        gamma_values: List of gamma values to test
        save_results: Whether to save the results to files
        output_dir: Directory to save results in (if save_results is True)
        
    Returns:
        Dictionary containing the verification results
    """
    # Set default gamma values if not provided
    if gamma_values is None:
        mu1 = 1.0 - mu0
        max_gamma = np.sqrt(mu0 * mu1)
        gamma_values = [0, 0.5*max_gamma, -0.5*max_gamma, -max_gamma]
    
    # Lists to store results
    conserved_values = []
    is_conserved = []
    
    # Calculate initial value (before measurement)
    network_initial = TensorNetwork()
    measures_initial = GeometricMeasures(network_initial)
    Q_initial = measures_initial.conserved_quantity()
    
    # Test each gamma value
    for gamma in gamma_values:
        # Create measurement operator
        mu1 = 1.0 - mu0
        M = create_measurement_operator(mu0, mu1, gamma)
        
        # Create tensor network and apply measurement
        network = TensorNetwork()
        network.apply_measurement(M)
        
        # Calculate conserved quantity
        measures = GeometricMeasures(network)
        Q = measures.conserved_quantity()
        
        # Check if conserved
        conserved = abs(Q - Q_initial) < 1e-6
        
        # Store results
        conserved_values.append(Q)
        is_conserved.append(conserved)
    
    # Compile results
    results = {
        'mu0': mu0,
        'gamma_values': gamma_values,
        'Q_initial': Q_initial,
        'Q_values': conserved_values,
        'is_conserved': is_conserved
    }
    
    # Save results if requested
    if save_results:
        import os
        import pickle
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to a pickle file
        output_file = os.path.join(output_dir, f'conservation_law_mu0_{mu0:.2f}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def plot_phase_diagram(mu0_range: Tuple[float, float] = (0.1, 0.9), 
                      mu0_points: int = 10,
                      save_fig: bool = False,
                      output_file: str = './figures/fig2_phase_diagram.png') -> Tuple[Any, Any]:
    """
    Plot the phase diagram showing the critical condition for geometric tears.
    
    Args:
        mu0_range: Range of mu0 values to plot (min, max)
        mu0_points: Number of points to sample in the mu0 range
        save_fig: Whether to save the figure to a file
        output_file: File path to save the figure to (if save_fig is True)
        
    Returns:
        Figure and axes objects for further customization
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid of mu0 and gamma values
    mu0_values = np.linspace(mu0_range[0], mu0_range[1], mu0_points)
    
    # Create a 2D grid
    mu0_grid, gamma_grid = np.meshgrid(
        mu0_values,
        np.linspace(-1, 1, 100)
    )
    
    # Calculate mu1 values
    mu1_grid = 1 - mu0_grid
    
    # Calculate maximum possible gamma for each point
    max_gamma_grid = np.sqrt(mu0_grid * mu1_grid)
    
    # Create a mask for the allowed region (|gamma| <= sqrt(mu0 * mu1))
    mask_allowed = np.abs(gamma_grid) <= max_gamma_grid
    
    # Create a mask for the connected geometry region (gamma > -sqrt(mu0 * mu1))
    mask_connected = gamma_grid > -max_gamma_grid
    
    # Create a mask for the torn geometry region (gamma < -sqrt(mu0 * mu1))
    mask_torn = gamma_grid < -max_gamma_grid
    
    # Create masks for the final regions
    final_mask_connected = mask_allowed & mask_connected
    final_mask_torn = mask_allowed & mask_torn
    
    # Create a custom colormap for the plot
    colors = [(0.8, 0.2, 0.2), (1, 1, 1), (0.2, 0.4, 0.8)]  # Red, White, Blue
    cmap_name = 'torn_connected'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Plot the connected region
    connected_region = np.zeros_like(mu0_grid)
    connected_region[final_mask_connected] = 0.5
    ax.contourf(mu0_grid, gamma_grid, connected_region, levels=[0.4, 0.6], colors=['#A6C4E0'])
    
    # Plot the torn region
    torn_region = np.zeros_like(mu0_grid)
    torn_region[final_mask_torn] = 0.5
    ax.contourf(mu0_grid, gamma_grid, torn_region, levels=[0.4, 0.6], colors=['#E0A6A6'])
    
    # Plot the critical line
    critical_gamma_values = np.array([-np.sqrt(mu0 * (1-mu0)) for mu0 in mu0_values])
    ax.plot(mu0_values, critical_gamma_values, 'k-', lw=2, label='Critical Line')
    
    # Add labels and legend
    ax.set_xlabel(r'$\mu_0$', fontsize=14)
    ax.set_ylabel(r'$\gamma$', fontsize=14)
    ax.set_title('Phase Diagram: Geometric Tears', fontsize=16)
    
    # Add text labels for the regions
    ax.text(0.3, 0.5, 'Connected Geometry', fontsize=14, ha='center')
    ax.text(0.7, -0.5, 'Torn Geometry', fontsize=14, ha='center')
    
    # Add a note about the critical condition
    ax.text(0.95, -0.95, r'$\gamma = -\sqrt{\mu_0\mu_1}$', fontsize=14, ha='right')
    
    # Set axis limits
    ax.set_xlim(mu0_range)
    ax.set_ylim(-1, 1)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure if requested
    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_mutual_information(results: Dict[str, Any],
                          save_fig: bool = False,
                          output_file: str = './figures/fig3_mutual_information.png') -> Tuple[Any, Any]:
    """
    Plot mutual information between adjacent qubits as a function of gamma.
    
    Args:
        results: Dictionary of results from analyze_critical_point
        save_fig: Whether to save the figure to a file
        output_file: File path to save the figure to (if save_fig is True)
        
    Returns:
        Figure and axes objects for further customization
    """
    # Extract data from results
    gamma_values = results['gamma_values']
    mutual_info_values = results['mutual_info_values']
    gamma_c = results['gamma_c']
    mu0 = results['mu0']
    mu1 = results['mu1']
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mutual information
    ax.plot(gamma_values, mutual_info_values, 'b-', lw=2)
    
    # Add a vertical line at the critical point
    ax.axvline(x=gamma_c, color='r', linestyle='--', label=r'$\gamma_c = -\sqrt{\mu_0\mu_1}$')
    
    # Add labels and legend
    ax.set_xlabel(r'$\gamma$', fontsize=14)
    ax.set_ylabel('Mutual Information', fontsize=14)
    ax.set_title(f'Mutual Information vs Interference Parameter ($\\mu_0 = {mu0}$)', fontsize=16)
    ax.legend()
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure if requested
    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_geometric_measures(results: Dict[str, Any],
                          save_fig: bool = False,
                          output_file: str = './figures/fig4_geometric_measures.png') -> Tuple[Any, Any]:
    """
    Plot distance and curvature as functions of gamma.
    
    Args:
        results: Dictionary of results from analyze_critical_point
        save_fig: Whether to save the figure to a file
        output_file: File path to save the figure to (if save_fig is True)
        
    Returns:
        Figure and axes objects for further customization
    """
    # Extract data from results
    gamma_values = results['gamma_values']
    distance_values = results['distance_values']
    curvature_values = results['curvature_values']
    gamma_c = results['gamma_c']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot distance
    ax1.semilogy(gamma_values, distance_values, 'b-', lw=2)
    ax1.axvline(x=gamma_c, color='r', linestyle='--')
    ax1.set_xlabel(r'$\gamma$', fontsize=14)
    ax1.set_ylabel('Effective Distance', fontsize=14)
    ax1.set_title('Distance Between Qubits', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot curvature
    ax2.semilogy(gamma_values, np.abs(curvature_values), 'b-', lw=2)
    ax2.axvline(x=gamma_c, color='r', linestyle='--')
    ax2.set_xlabel(r'$\gamma$', fontsize=14)
    ax2.set_ylabel('Curvature', fontsize=14)
    ax2.set_title('Quantum Curvature', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def plot_entanglement_spectrum(results: Dict[str, Any],
                             save_fig: bool = False,
                             output_file: str = './figures/fig5_entanglement_spectrum.png') -> Tuple[Any, Any]:
    """
    Plot the entanglement spectrum as a function of gamma.
    
    Args:
        results: Dictionary of results from analyze_entanglement_spectrum
        save_fig: Whether to save the figure to a file
        output_file: File path to save the figure to (if save_fig is True)
        
    Returns:
        Figure and axes objects for further customization
    """
    # Extract data from results
    gamma_values = results['gamma_values']
    eigenvalues_list = results['eigenvalues']
    gamma_c = results['gamma_c']
    
    # Create arrays for the top two eigenvalues
    lambda1 = np.zeros_like(gamma_values)
    lambda2 = np.zeros_like(gamma_values)
    
    # Extract the top two eigenvalues at each gamma value
    for i, eigenvalues in enumerate(eigenvalues_list):
        if eigenvalues is not None and len(eigenvalues) >= 2:
            lambda1[i] = eigenvalues[0]
            lambda2[i] = eigenvalues[1]
        else:
            lambda1[i] = np.nan
            lambda2[i] = np.nan
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot eigenvalues
    ax.plot(gamma_values, lambda1, 'b-', lw=2, label=r'$\lambda_1$')
    ax.plot(gamma_values, lambda2, 'r-', lw=2, label=r'$\lambda_2$')
    
    # Add a vertical line at the critical point
    ax.axvline(x=gamma_c, color='k', linestyle='--')
    
    # Add labels and legend
    ax.set_xlabel(r'$\gamma$', fontsize=14)
    ax.set_ylabel('Eigenvalue', fontsize=14)
    ax.set_title('Entanglement Spectrum', fontsize=16)
    ax.legend()
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure if requested
    if save_fig:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig, ax


def run_simulation():
    """
    Run a complete simulation and generate all figures.
    """
    import os
    
    # Create output directories
    os.makedirs('./data/critical_point_scans/', exist_ok=True)
    os.makedirs('./data/entanglement_spectra/', exist_ok=True)
    os.makedirs('./data/perturbation_analysis/', exist_ok=True)
    os.makedirs('./data/conservation_law/', exist_ok=True)
    os.makedirs('./figures/', exist_ok=True)
    
    # Run analyses for different mu0 values
    mu0_values = [0.3, 0.5, 0.7]
    all_results = {}
    
    for mu0 in mu0_values:
        print(f"Analyzing mu0 = {mu0}...")
        
        # Analyze critical point
        critical_results = analyze_critical_point(mu0, save_results=True)
        all_results[f'critical_mu0_{mu0}'] = critical_results
        
        # Analyze entanglement spectrum
        spectrum_results = analyze_entanglement_spectrum(mu0, save_results=True)
        all_results[f'spectrum_mu0_{mu0}'] = spectrum_results
        
        # Analyze perturbation
        perturbation_results = analyze_perturbation(mu0, save_results=True)
        all_results[f'perturbation_mu0_{mu0}'] = perturbation_results
        
        # Verify conservation law
        conservation_results = verify_conservation_law(mu0, save_results=True)
        all_results[f'conservation_mu0_{mu0}'] = conservation_results
    
    # Generate figures
    print("Generating figures...")
    
    # Phase diagram
    plot_phase_diagram(save_fig=True)
    
    # Plots for mu0 = 0.5
    mu0 = 0.5
    critical_results = all_results[f'critical_mu0_{mu0}']
    spectrum_results = all_results[f'spectrum_mu0_{mu0}']
    
    plot_mutual_information(critical_results, save_fig=True)
    plot_geometric_measures(critical_results, save_fig=True)
    plot_entanglement_spectrum(spectrum_results, save_fig=True)
    
    print("Simulation complete!")


if __name__ == "__main__":
    run_simulation()