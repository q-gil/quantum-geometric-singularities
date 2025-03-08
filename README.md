# Quantum Geometric Singularities

This repository contains code and data supporting the paper "Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks" by Hamid Bahri (2025).

## Abstract

This research establishes a fundamental connection between quantum measurements and emergent spacetime geometry using tensor networks. Our minimal model—a ring of four qubits coupled to a central probe—demonstrates how quantum measurements with specific interference characteristics shape geometric properties. We introduce a phase-sensitive quantum curvature measure and prove mathematically that measurements with destructive interference (characterized precisely by γ = -√(μ₀μ₁)) create geometric "tears"—analogous to spacetime singularities—where mutual information between adjacent regions vanishes and distance becomes infinite.

## Repository Structure

- `/code/`: Implementation of tensor network simulations
  - `tensor_network.py`: Core tensor network implementation
  - `measurement_operators.py`: Quantum measurement implementations
  - `geometric_measures.py`: Functions to calculate geometric properties
  - `analysis.py`: Analysis and visualization tools

- `/data/`: Simulation results and analysis
  - `critical_point_scans/`: Data near the critical point γ = -√(μ₀μ₁)
  - `entanglement_spectra/`: Entanglement spectrum data
  - `mutual_information/`: Mutual information measurements

- `/figures/`: Generated figures from the paper
  - `fig1_tensor_network.png`: Tensor network structure
  - `fig2_phase_diagram.png`: Phase diagram of geometric tears
  - `fig3_mutual_information.png`: Mutual information plots
  - `fig4_geometric_measures.png`: Distance and curvature plots
  - `fig5_entanglement_spectrum.png`: Entanglement spectrum plots
  - `fig6_schematic.png`: Schematic of geometric tears

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- QuTiP (for quantum simulations)
- TensorNetwork (Google's tensor network library)

## Usage

1. Clone this repository:
```bash
git clone https://github.com/q-gil/quantum-geometric-singularities.git
cd quantum-geometric-singularities
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main simulation:
```bash
python code/run_simulation.py
```

4. Analyze results:
```bash
python code/analyze_results.py
```

## Citation

If you use this code or build upon our methods, please cite:
```
Bahri, H. (2025). Quantum Geometric Singularities: Measurement-Induced Topological Transitions in Tensor Networks.
```

## License

This code is released under the MIT License. See the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please contact:
- Email: research@q-gil.org
- Website: https://q-gil.org
- GitHub: https://github.com/q-gil
