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
