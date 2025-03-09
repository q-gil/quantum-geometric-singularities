"""
Quantum Measurement Implementations for Quantum Geometric Singularities

This module provides implementations of quantum measurement operators described in
"Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks"
by Hamid Bahri (2025).

The module focuses on measurement operators with different interference characteristics
and their effects on the tensor network geometry.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from tensor_network import TensorNetwork, create_measurement_operator, critical_gamma

class MeasurementOperator:
    """
    A class representing quantum measurement operators with specific interference characteristics.
    """
    
    def __init__(self, mu0: float = 0.5, mu1: float = None, gamma: float = None):
        """
        Initialize a measurement operator with specific parameters.
        
        Args:
            mu0: Probability of projecting onto |0⟩
            mu1: Probability of projecting onto |1⟩ (defaults to 1-mu0)
            gamma: Interference parameter (defaults to 0 for neutral interference)
        """
        self.mu0 = mu0
        self.mu1 = 1.0 - mu0 if mu1 is None else mu1
        
        # Default to neutral interference
        self.gamma = 0.0 if gamma is None else gamma
        
        # Check if gamma satisfies the Cauchy-Schwarz inequality
        max_gamma = np.sqrt(self.mu0 * self.mu1)
        if abs(self.gamma) > max_gamma + 1e-10:
            raise ValueError(f"Invalid gamma value. |gamma| must be <= sqrt(mu0*mu1) = {max_gamma}")
        
        # Create the measurement operator matrix
        self.matrix = self._create_matrix()
    
    def _create_matrix(self) -> np.ndarray:
        """
        Create the measurement operator matrix.
        
        Returns:
            2x2 measurement operator matrix
        """
        M = np.zeros((2, 2), dtype=complex)
        
        # Set matrix elements according to equation (8) in the paper
        M[0, 0] = np.sqrt(self.mu0)
        M[0, 1] = np.conj(self.gamma) / np.sqrt(self.mu1)
        M[1, 0] = self.gamma / np.sqrt(self.mu0)
        M[1, 1] = np.sqrt(self.mu1)
        
        return M
    
    def apply_to(self, network: TensorNetwork) -> TensorNetwork:
        """
        Apply the measurement operator to a tensor network.
        
        Args:
            network: The tensor network to apply the measurement to
            
        Returns:
            Modified tensor network after measurement
        """
        # Create a copy of the network
        # In a full implementation, this would create a deep copy
        result = network  # Simplified for demonstration
        
        # Apply the measurement
        result.apply_measurement(self.matrix)
        
        return result


def create_constructive_interference(mu0: float = 0.5, mu1: float = None, 
                                    strength: float = 0.5) -> MeasurementOperator:
    """
    Create a measurement operator with constructive interference.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        mu1: Probability of projecting onto |1⟩ (defaults to 1-mu0)
        strength: Relative strength of the interference (0 to 1)
        
    Returns:
        MeasurementOperator with constructive interference
    """
    mu1 = 1.0 - mu0 if mu1 is None else mu1
    
    # Calculate maximum allowed gamma
    max_gamma = np.sqrt(mu0 * mu1)
    
    # Set gamma to a positive value scaled by strength
    gamma = max_gamma * strength
    
    return MeasurementOperator(mu0, mu1, gamma)


def create_destructive_interference(mu0: float = 0.5, mu1: float = None, 
                                   strength: float = 0.5) -> MeasurementOperator:
    """
    Create a measurement operator with destructive interference.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        mu1: Probability of projecting onto |1⟩ (defaults to 1-mu0)
        strength: Relative strength of the interference (0 to 1)
        
    Returns:
        MeasurementOperator with destructive interference
    """
    mu1 = 1.0 - mu0 if mu1 is None else mu1
    
    # Calculate maximum allowed negative gamma
    max_gamma = np.sqrt(mu0 * mu1)
    
    # Set gamma to a negative value scaled by strength
    gamma = -max_gamma * strength
    
    return MeasurementOperator(mu0, mu1, gamma)


def create_critical_measurement(mu0: float = 0.5, mu1: float = None) -> MeasurementOperator:
    """
    Create a measurement operator at the critical point where geometric tears form.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        mu1: Probability of projecting onto |1⟩ (defaults to 1-mu0)
        
    Returns:
        MeasurementOperator at the critical point
    """
    mu1 = 1.0 - mu0 if mu1 is None else mu1
    
    # Calculate critical gamma value
    gamma = -np.sqrt(mu0 * mu1)
    
    return MeasurementOperator(mu0, mu1, gamma)


def scan_gamma_values(network: TensorNetwork, mu0: float = 0.5, 
                     gamma_range: Tuple[float, float] = (-0.5, 0.5), 
                     num_points: int = 100) -> Dict[str, np.ndarray]:
    """
    Scan through gamma values and measure the effect on the tensor network.
    
    Args:
        network: The tensor network to analyze
        mu0: Probability of projecting onto |0⟩
        gamma_range: Range of gamma values to scan (min, max)
        num_points: Number of points to sample in the range
        
    Returns:
        Dictionary containing arrays of gamma values and corresponding measurements
    """
    mu1 = 1.0 - mu0
    
    # Create array of gamma values
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_points)
    
    # Arrays to store results
    mutual_info_values = np.zeros(num_points)
    distance_values = np.zeros(num_points)
    schmidt_gap_values = np.zeros(num_points)
    
    # Critical value for reference
    gamma_c = critical_gamma(mu0, mu1)
    
    # Scan through gamma values
    for i, gamma in enumerate(gamma_values):
        # Skip if gamma is beyond the allowed range
        if abs(gamma) > np.sqrt(mu0 * mu1) + 1e-10:
            mutual_info_values[i] = np.nan
            distance_values[i] = np.nan
            schmidt_gap_values[i] = np.nan
            continue
        
        # Create measurement operator
        measurement = MeasurementOperator(mu0, mu1, gamma)
        
        # Apply measurement to a copy of the network
        # In a full implementation, this would create a deep copy
        measured_network = network  # Simplified
        measured_network.apply_measurement(measurement.matrix)
        
        # Calculate mutual information between adjacent qubits
        mutual_info = measured_network.mutual_information([0], [1])
        mutual_info_values[i] = mutual_info
        
        # Calculate distance (if mutual information is positive)
        if mutual_info > 0:
            distance = -np.log2(mutual_info / (2 * np.log2(2)))
        else:
            distance = float('inf')
        distance_values[i] = distance
        
        # Calculate Schmidt gap
        spectrum = measured_network.entanglement_spectrum([0, 1], [2, 3])
        if len(spectrum) >= 2:
            schmidt_gap = spectrum[0] - spectrum[1]
        else:
            schmidt_gap = 0
        schmidt_gap_values[i] = schmidt_gap
    
    # Return results
    return {
        'gamma_values': gamma_values,
        'mutual_info_values': mutual_info_values,
        'distance_values': distance_values,
        'schmidt_gap_values': schmidt_gap_values,
        'gamma_c': gamma_c
    }


def scan_phase_space(network: TensorNetwork, 
                    mu0_range: Tuple[float, float] = (0.1, 0.9),
                    gamma_range: Tuple[float, float] = (-0.5, 0.5),
                    num_points: Tuple[int, int] = (20, 20)) -> Dict[str, np.ndarray]:
    """
    Scan the phase space of mu0 and gamma values.
    
    Args:
        network: The tensor network to analyze
        mu0_range: Range of mu0 values to scan (min, max)
        gamma_range: Range of gamma values to scan (min, max)
        num_points: Number of points to sample in each dimension (mu0, gamma)
        
    Returns:
        Dictionary containing 2D arrays of measurements across the phase space
    """
    # Create arrays of parameter values
    mu0_values = np.linspace(mu0_range[0], mu0_range[1], num_points[0])
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_points[1])
    
    # Create 2D meshgrid for the phase space
    mu0_grid, gamma_grid = np.meshgrid(mu0_values, gamma_values)
    
    # Arrays to store results
    mutual_info_grid = np.zeros_like(mu0_grid)
    distance_grid = np.zeros_like(mu0_grid)
    schmidt_gap_grid = np.zeros_like(mu0_grid)
    
    # Calculate critical line
    mu1_grid = 1.0 - mu0_grid
    critical_gamma_grid = -np.sqrt(mu0_grid * mu1_grid)
    
    # Scan through the phase space
    for i in range(num_points[0]):
        for j in range(num_points[1]):
            mu0 = mu0_values[i]
            gamma = gamma_values[j]
            mu1 = 1.0 - mu0
            
            # Skip if gamma is beyond the allowed range
            if abs(gamma) > np.sqrt(mu0 * mu1) + 1e-10:
                mutual_info_grid[j, i] = np.nan
                distance_grid[j, i] = np.nan
                schmidt_gap_grid[j, i] = np.nan
                continue
            
            # Create measurement operator
            measurement = MeasurementOperator(mu0, mu1, gamma)
            
            # Apply measurement to a copy of the network
            # In a full implementation, this would create a deep copy
            measured_network = network  # Simplified
            measured_network.apply_measurement(measurement.matrix)
            
            # Calculate mutual information
            mutual_info = measured_network.mutual_information([0], [1])
            mutual_info_grid[j, i] = mutual_info
            
            # Calculate distance
            if mutual_info > 0:
                distance = -np.log2(mutual_info / (2 * np.log2(2)))
            else:
                distance = float('inf')
            distance_grid[j, i] = distance
            
            # Calculate Schmidt gap
            spectrum = measured_network.entanglement_spectrum([0, 1], [2, 3])
            if len(spectrum) >= 2:
                schmidt_gap = spectrum[0] - spectrum[1]
            else:
                schmidt_gap = 0
            schmidt_gap_grid[j, i] = schmidt_gap
    
    # Return results
    return {
        'mu0_grid': mu0_grid,
        'gamma_grid': gamma_grid,
        'critical_gamma_grid': critical_gamma_grid,
        'mutual_info_grid': mutual_info_grid,
        'distance_grid': distance_grid,
        'schmidt_gap_grid': schmidt_gap_grid
    }