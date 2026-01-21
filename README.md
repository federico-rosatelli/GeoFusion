# Stellarator Optimization Project

![University of Bonn](https://www.uni-bonn.de/de/universitaet/medien-universitaet/medien-presse-kommunikation/medien-corporate-design/uni_bonn_logo_standard_logo.jpg/images/image/large)

**University of Bonn**  
**Lab Geometry Processing**  
**Author:** Federico Rosatelli

---

## About the Project
GeoFusion is an interactive platform developed to assist in the geometric design and optimization of stellarators. Fusion energy relies on precise magnetic confinement, and the shape of the plasma boundary is critical to its success. This tool bridges the gap between theoretical physics and computational design, providing an intuitive interface to explore these complex geometries while balancing physical stability with engineering constraints.

## Key Features
*   **Dynamic 3D Visualization**: Inspect plasma boundaries in a fully interactive 3D environment, allowing for detailed analysis of surface curvature and topology.
*   **Geometric Optimization**: Leverage physics-based algorithms to find optimal shapes, prioritizing either:
    *   **Coil Simplicity**: Reducing the complexity of the external magnets to ensure build feasibility.
    *   **MHD Stability**: Optimizing the geometry for stable plasma confinement.
*   **Real-Time Feedback**: Adjust Fourier modes manually and instantly monitor key metrics—such as Aspect Ratio and Curvature—ensuring a tight feedback loop during the design process.

## Technologies Used
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%23ffffff)

## Getting Started
To run the application locally, ensure you have the necessary dependencies installed:

1.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch the Dashboard**
    ```bash
    streamlit run app/gui_app.py
    ```

## Future Work / To-Do
The current `GeoFusion` project requires the following updates to align with the state-of-the-art:

### Physics Engine
- [ ] **Interface with VMEC++ or DESC**: The current magnetic field proxy (`1/R`) is insufficient for accurate optimization. Integration with a full MHD equilibrium solver like VMEC++ or DESC is required.
    - *Current*: `src/physics/magnetic.py` uses analytical approximations.
    - *Goal*: Call VMEC++ or DESC to get real `B` field and equilibria.

### Objectives
- [ ] **Vacuum Magnetic Well ($W_{MHD}$)**: Replace the current mean curvature proxy ($H^2$) with the Vacuum Magnetic Well metric to properly optimize for MHD stability.
    - *Current*: `src/optimization/objectives.py` calculates `mean(H^2)`.
    - *Goal*: Implement $W_{MHD}$ calculation from VMEC/DESC outputs.

### Constraints
- [ ] **Max Elongation ($\epsilon_{max}$)**: Implement a constraint for the maximum elongation of the plasma cross-section.
- [ ] **Mirror Ratio ($\Delta_{edge}$)**: Integrate the existing `calculate_mirror_ratio` from `geometry.py` into the optimization loop as a constraint.
- [ ] **Rotational Transform ($\iota$)**: Add a constraint for the edge rotational transform per field period.
- [ ] **Quasi-Isodynamicity (QI) Residual**: Implement the QI residual metric as defined in the paper.
- [ ] **Turbulent Transport ($\chi_{\nabla r}$)**: Implement the proxy for turbulent transport ("flux-surface compression in regions of bad curvature").

### Benchmarks
- [ ] **Define Standard Benchmarks**: Implement the three specific optimization problems described in the paper:
    1.  **Geometric**: Minimize $\epsilon_{max}$ subject to fixed Aspect Ratio and $\iota$.
    2.  **Simple-to-build QI**: Minimize coil complexity (magnetic gradient scale length) subject to QI and mirror ratio constraints.
    3.  **MHD-stable QI**: Multi-objective optimization of coil complexity and compactness subject to MHD stability and transport constraints.
