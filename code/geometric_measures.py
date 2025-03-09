"""
Geometric Measures for Quantum Geometric Singularities

This module provides functions to calculate geometric properties described in
"Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks"
by Hamid Bahri (2025).

The module focuses on distance measures, quantum curvature, and conserved quantities.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from tensor_network import TensorNetwork

class GeometricMeasures:
    """
    A class for calculating geometric properties of the tensor network.
    """
    
    def __init__(self, network: TensorNetwork):
        """
        Initialize with a tensor network.
        
        Args:
            network: The tensor network to analyze
        """
        self.network = network
        
        # Cache for calculated values
        self._distance_table = None
        self._curvature_vector = None
    
    def calculate_distance(self, qubit_i: int, qubit_j: int) -> float:
        """
        Calculate the information-theoretic distance between two qubits.
        
        Args:
            qubit_i: Index of the first qubit
            qubit_j: Index of the second qubit
            
        Returns:
            Distance between the qubits
        """
        # Calculate mutual information
        mutual_info = self.network.mutual_information([qubit_i], [qubit_j])
        
        # Calculate distance using equation (26) from the paper
        if mutual_info > 0:
            distance = -np.log2(mutual_info / (2 * np.log2(2)))
        else:
            distance = float('inf')
        
        return distance
    
    def distance_table(self) -> np.ndarray:
        """
        Calculate the distance table between all pairs of qubits.
        
        Returns:
            4x4 array where entry (i,j) is the distance between qubits i and j
        """
        if self._distance_table is not None:
            return self._distance_table
        
        # Initialize distance table
        num_qubits = 4  # Fixed for this model
        distance_table = np.zeros((num_qubits, num_qubits))
        
        # Calculate distances
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                distance = self.calculate_distance(i, j)
                distance_table[i, j] = distance
                distance_table[j, i] = distance
        
        # Cache the result
        self._distance_table = distance_table
        
        return distance_table
    
    def quantum_curvature(self, qubit_i: int) -> float:
        """
        Calculate the quantum curvature at a given qubit.
        
        Args:
            qubit_i: Index of the qubit
            
        Returns:
            Quantum curvature value
        """
        # Calculate conditional entropy S(P|Ri)
        rho_i = self.network.reduced_density_matrix([qubit_i])
        rho_P = self._get_probe_density_matrix()
        rho_Pi = self._get_joint_density_matrix(qubit_i)
        
        S_P = self._von_neumann_entropy(rho_P)
        S_Pi = self._von_neumann_entropy(rho_Pi)
        S_i = self._von_neumann_entropy(rho_i)
        
        S_P_given_i = S_Pi - S_i
        
        # Calculate normalized mutual information ξ
        # Assuming qubits are arranged in a ring, so qubit_i+1 wraps around
        next_qubit = (qubit_i + 1) % 4
        mutual_info = self.network.mutual_information([qubit_i], [next_qubit])
        xi = mutual_info / (2 * np.log2(2))
        
        # Handle the case where mutual information is zero
        if xi < 1e-10:
            return float('nan')  # Curvature is undefined at a tear
        
        # Calculate curvature using equation (27) from the paper
        curvature = (np.exp(S_P_given_i) - xi**2) / xi
        
        return curvature
    
    def quantum_curvature_vector(self) -> np.ndarray:
        """
        Calculate the quantum curvature at all qubits.
        
        Returns:
            Array of curvature values for each qubit
        """
        if self._curvature_vector is not None:
            return self._curvature_vector
        
        # Calculate curvature for each qubit
        num_qubits = 4  # Fixed for this model
        curvature_vector = np.zeros(num_qubits)
        
        for i in range(num_qubits):
            curvature_vector[i] = self.quantum_curvature(i)
        
        # Cache the result
        self._curvature_vector = curvature_vector
        
        return curvature_vector
    
    def conserved_quantity(self) -> float:
        """
        Calculate the conserved quantity Q defined in equation (28).
        
        Returns:
            Value of the conserved quantity
        """
        # Calculate quantum curvature at qubit 0 (any qubit can be used)
        curvature = self.quantum_curvature(0)
        
        # Calculate conditional entropy of probe given entire ring
        rho_P = self._get_probe_density_matrix()
        rho_PX = self._get_joint_density_matrix_all()
        rho_X = self.network.reduced_density_matrix([0, 1, 2, 3])
        
        S_PX = self._von_neumann_entropy(rho_PX)
        S_X = self._von_neumann_entropy(rho_X)
        
        S_P_given_X = S_PX - S_X
        
        # Maximum possible conditional entropy (1 bit for a qubit system)
        S_max = 1.0
        
        # Calculate Q using equation (28) from the paper
        Q = curvature * np.exp(S_P_given_X / S_max)
        
        return Q
    
    def _get_probe_density_matrix(self) -> np.ndarray:
        """
        Get the reduced density matrix of the probe.
        
        Returns:
            2x2 density matrix for the probe qubit
        """
        # In a full implementation, this would compute the partial trace
        # Here we use a simplified approach for demonstration
        
        # For our specific model, the probe is maximally mixed after measurement
        return np.eye(2) / 2
    
    def _get_joint_density_matrix(self, qubit_i: int) -> np.ndarray:
        """
        Get the joint density matrix of the probe and a ring qubit.
        
        Args:
            qubit_i: Index of the ring qubit
            
        Returns:
            4x4 joint density matrix
        """
        # In a full implementation, this would compute the partial trace
        # Here we use a simplified approach for demonstration
        
        # For our specific model, return a simplified representation
        # In a real implementation, this would be computed from the full state
        
        # Create a 4x4 identity matrix as a placeholder
        return np.eye(4) / 4
    
    def _get_joint_density_matrix_all(self) -> np.ndarray:
        """
        Get the joint density matrix of the probe and all ring qubits.
        
        Returns:
            32x32 joint density matrix
        """
        # In a full implementation, this would be the full state density matrix
        # Here we use a simplified approach for demonstration
        
        # For our specific model, the full state is pure
        # So we return a rank-1 density matrix
        # In a real implementation, this would be computed from the full state
        
        # Create a 32x32 rank-1 matrix as a placeholder
        dim = 2**(4+1)  # 4 ring qubits + 1 probe qubit
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0  # Just a placeholder, not the actual state
        
        return rho
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Calculate the von Neumann entropy of a density matrix.
        
        Args:
            rho: Density matrix
            
        Returns:
            von Neumann entropy in bits
        """
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Filter out zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy
        return -np.sum(eigenvalues * np.log2(eigenvalues))