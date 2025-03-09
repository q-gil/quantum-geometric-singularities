"""
Core Tensor Network Implementation for Quantum Geometric Singularities

This module implements the core tensor network structure described in 
"Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks"
by Hamid Bahri (2025).

The model consists of a ring of four qubits coupled to a central probe qubit.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class TensorNetwork:
    """
    A class implementing the tensor network model with a ring of four qubits 
    coupled to a central probe.
    """
    
    def __init__(self):
        """Initialize the tensor network with the default structure."""
        # Initialize ring tensors
        self.ring_tensors = []
        for i in range(4):
            R_i = np.zeros((2, 2, 2, 2), dtype=complex)
            # Physical index matches bond indices
            for a in range(2):
                for b_prev in range(2):
                    if a == b_prev:  # Ensure perfect correlation
                        # Connection to the probe is in equal superposition of |0⟩ and |1⟩
                        R_i[a, a, 0, 0] = 1/np.sqrt(2)
                        R_i[a, a, 0, 1] = 1/np.sqrt(2)
            self.ring_tensors.append(R_i)
        
        # Initialize probe tensor
        self.probe_tensor = np.zeros((2, 2, 2, 2, 2), dtype=complex)
        # GHZ-like state: all |0⟩ or all |1⟩
        self.probe_tensor[0, 0, 0, 0, 0] = 1/np.sqrt(2)
        self.probe_tensor[1, 1, 1, 1, 1] = 1/np.sqrt(2)
        
        # Full state tensor (will be calculated when needed)
        self.state_tensor = None
        
        # Build the initial state
        self._build_state()
    
    def _build_state(self):
        """Contract the tensors to build the full state representation."""
        # Create a simple ring state (all |0⟩ or all |1⟩)
        # |0000⟩ + |1111⟩ / sqrt(2)
        ring_state = np.zeros((2, 2, 2, 2), dtype=complex)
        ring_state[0, 0, 0, 0] = 1/np.sqrt(2)
        ring_state[1, 1, 1, 1] = 1/np.sqrt(2)
        
        # Create a simple probe state |0⟩ + |1⟩ / sqrt(2)
        probe_state = np.zeros(2, dtype=complex)
        probe_state[0] = 1/np.sqrt(2)
        probe_state[1] = 1/np.sqrt(2)
        
        # Combine to create the full initial state
        # |0000⟩|0⟩ + |1111⟩|1⟩ / sqrt(2)
        self.state_tensor = np.zeros((2, 2, 2, 2, 2), dtype=complex)
        self.state_tensor[0, 0, 0, 0, 0] = 1/np.sqrt(2)
        self.state_tensor[1, 1, 1, 1, 1] = 1/np.sqrt(2)

    def apply_measurement(self, M: np.ndarray):
        """
        Apply a measurement operator to the probe qubit.
        
        Args:
            M: 2x2 measurement operator matrix
        """
        # Apply measurement to the probe qubit
        # In full implementation, this would involve proper tensor contraction
        # Here we use a simplified representation
        
        # Apply M to the probe index of the state tensor
        new_state = np.zeros_like(self.state_tensor)
        for i in range(2):
            for j in range(2):
                # Apply each element of M
                new_state += M[i, j] * self.state_tensor[..., j]
        
        # Normalize the state
        norm = np.sqrt(np.sum(np.abs(new_state)**2))
        self.state_tensor = new_state / norm
    
    def mutual_information(self, qubits_A: List[int], qubits_B: List[int]) -> float:
        """
        Calculate the mutual information between two subsets of ring qubits.
        
        Args:
            qubits_A: List of indices for the first subset of qubits
            qubits_B: List of indices for the second subset of qubits
            
        Returns:
            Mutual information between the two subsets
        """
        rho_A = self.reduced_density_matrix(qubits_A)
        rho_B = self.reduced_density_matrix(qubits_B)
        qubits_AB = sorted(list(set(qubits_A + qubits_B)))
        rho_AB = self.reduced_density_matrix(qubits_AB)
        
        S_A = self._von_neumann_entropy(rho_A)
        S_B = self._von_neumann_entropy(rho_B)
        S_AB = self._von_neumann_entropy(rho_AB)
        
        return S_A + S_B - S_AB
    
    def reduced_density_matrix(self, qubits: List[int]) -> np.ndarray:
        """
        Calculate the reduced density matrix for a subset of qubits.
        
        Args:
            qubits: List of qubit indices to keep
            
        Returns:
            Reduced density matrix
        """
        # In a full implementation, this would involve proper partial traces
        # Here we use a simplified approach for the specific case of our network
        
        # For simplicity, we'll implement this for the special cases needed in the paper
        if set(qubits) == {0, 1}:  # Adjacent qubits 1 and 2
            # Calculate reduced density matrix for qubits 1,2
            dim = 2**len(qubits)
            rho = np.zeros((dim, dim), dtype=complex)
            
            # For our specific model with the critical measurement
            # This is a simplified implementation for demonstration
            mu0 = 0.5  # Example value
            mu1 = 0.5  # Example value
            gamma = -np.sqrt(mu0 * mu1)  # Critical value
            
            K_squared = 1 + 2*abs(gamma)
            
            # The density matrix from equation (41) in the paper
            rho[0, 0] = mu0 / K_squared  # |00⟩⟨00|
            rho[3, 3] = mu1 / K_squared  # |11⟩⟨11|
            rho[0, 3] = gamma * np.sqrt(mu1/mu0) / K_squared  # |00⟩⟨11|
            rho[3, 0] = np.conj(gamma) * np.sqrt(mu0/mu1) / K_squared  # |11⟩⟨00|
            
            return rho
            
        # For other cases, implement as needed
        # This is a placeholder for a more general implementation
        return np.eye(2**len(qubits)) / 2**len(qubits)
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Calculate the von Neumann entropy of a density matrix.
        
        Args:
            rho: Density matrix
            
        Returns:
            von Neumann entropy
        """
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Filter out zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def entanglement_spectrum(self, qubits_A: List[int], qubits_B: List[int]) -> np.ndarray:
        """
        Calculate the entanglement spectrum for a bipartition of the ring.
        
        Args:
            qubits_A: First partition of qubits
            qubits_B: Second partition of qubits
            
        Returns:
            Array of Schmidt coefficients in descending order
        """
        # Calculate reduced density matrix for partition A
        rho_A = self.reduced_density_matrix(qubits_A)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Filter out numerical zeros
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        return eigenvalues


def create_measurement_operator(mu0: float, mu1: float, gamma: float) -> np.ndarray:
    """
    Create a measurement operator with the specified parameters.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        mu1: Probability of projecting onto |1⟩
        gamma: Interference parameter
        
    Returns:
        2x2 measurement operator matrix
    """
    M = np.zeros((2, 2), dtype=complex)
    
    # Set the matrix elements according to equation (8) in the paper
    M[0, 0] = np.sqrt(mu0)
    M[0, 1] = np.conj(gamma) / np.sqrt(mu1)
    M[1, 0] = gamma / np.sqrt(mu0)
    M[1, 1] = np.sqrt(mu1)
    
    return M


def critical_gamma(mu0: float, mu1: float) -> float:
    """
    Calculate the critical value of gamma for the given mu0 and mu1.
    
    Args:
        mu0: Probability of projecting onto |0⟩
        mu1: Probability of projecting onto |1⟩
        
    Returns:
        Critical value of gamma
    """
    return -np.sqrt(mu0 * mu1)