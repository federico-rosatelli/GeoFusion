# GeoFusion: Stellarator Geometric Optimization & Surrogate Modeling

![University of Bonn](https://www.uni-bonn.de/de/universitaet/medien-universitaet/medien-presse-kommunikation/medien-corporate-design/uni_bonn_logo_standard_logo.jpg/images/image/large)

**University of Bonn** **Lab:** Geometry Processing  
**Author:** Federico Rosatelli  

---

## About the Project

**GeoFusion** is an interactive platform and computational framework developed to assist in the geometric design and optimization of stellarators. Because fusion energy relies on extremely precise magnetic confinement, the exact shape of the plasma boundary is critical to the operational success of the device.

This tool bridges the gap between theoretical physics and computational design. Leveraging the [**ConStellaration**](https://huggingface.co/datasets/proxima-fusion/constellaration) dataset for quasi-isodynamic (QI) configurations, GeoFusion provides an intuitive interface to explore complex topologies and a Machine Learning infrastructure to rapidly approximate computationally expensive physics metrics. The primary objective is to balance physical plasma stability (e.g., MHD stability) with engineering feasibility (e.g., simplicity of external coils).

---

## Key Features

* **Dynamic 3D Visualization**: Inspect plasma boundaries in a fully interactive 3D environment, allowing for detailed analysis of surface curvature and topology in real-time.
* **Machine Learning Surrogate Models**: Fast approximation of complex stellarator metrics (e.g., Quasi-Isodynamicity, MHD Stability, rotational transform, and mirror ratio) using Neural Networks. The framework supports dynamically configurable architectures (MLP, CNN) to process both flat feature vectors and sequence-based inputs like Fourier coefficients.
* **Geometric Optimization**: Leverage physics-based optimization algorithms (e.g., SLSQP) to find optimal plasma shapes, prioritizing:
    * **Coil Simplicity**: Reducing the complexity of the external magnets to ensure build feasibility.
    * **MHD Stability & Confinement**: Optimizing the geometry for stable plasma confinement.
* **Real-Time Feedback**: Adjust Fourier modes manually and instantly monitor key metrics—such as Aspect Ratio and Curvature—ensuring a tight feedback loop during the design process.

---

## Technologies Used

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%23ffffff)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

---

## Core Code Structure

The project's source code is modular and organized within the `src/` directory:

* **`io/`**: Handles input management and data loading, including fetching initial stellarator configurations from the ConStellaration dataset.
* **`optimization/`**: Contains the core optimization logic. This includes the main SLSQP loop (`optimizer.py`), objective functions for coil complexity and MHD stability proxies (`objectives.py`), and boundary constraint validation (`constraints.py`).
* **`physics/`**: Implements physical and geometric foundations. Transforms Fourier coefficients into 3D coordinates, calculates surface curvatures (`geometry.py`), and handles current magnetic field approximations (`magnetic.py`).
* **`ml/`**: Modules dedicated to the data management, training, and inference of surrogate neural networks.
* **`visualization/`**: Tools for 3D rendering (via Matplotlib/Plotly) and evaluating model performance through loss and accuracy plots.

---

## Getting Started

To run the application and execute the scripts locally, ensure all necessary dependencies are installed in your Python environment.

### 1. Installation
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard (GUI)
To start the interactive visual interface:
```bash
streamlit run app/gui_app.py
```

### 3. Command-Line Interface (CLI) for Surrogate Models
The main.py file provides a command-line interface to manage the lifecycle of the deep learning models defined in configs/model_struct.json.

* **Train the models:**
    ```bash
    python main.py --train --config configs/conf.yaml --struct configs/model_struct.json
    ```

* **Force retraining (overrides existing models):**
    ```bash
    python main.py --train --force
    ```

* **Plot evaluation graphs (Loss/Accuracy history):**
    ```bash
    python main.py --plot
    ```

* **Run optimization tests:**
    ```bash
    python main.py --test
    ```


## Roadmap & Future Work
The long-term goal of GeoFusion is to fully align with state-of-the-art fusion research by transitioning from analytical proxies to rigorous physics validations.


### 1. Physics Engine Integration
* **Interface with VMEC++ or DESC:** Currently, magnetic.py uses analytical approximations ($|B| \approx 1/R$). Integration with a full MHD equilibrium solver is required to generate ground-truth data and validate the surrogate models.

## 2. Objectives Evolution
* **Vacuum Magnetic Well ($W_{MHD}$):** Replace the current geometric proxy based on mean curvature ($H^2$) with the actual magnetic well calculation derived from VMEC/DESC outputs.

### 3. New Optimization Constraints
* **Max Elongation ($\epsilon_{max}$):** Implement a constraint for the maximum elongation of the plasma cross-section.

* **Mirror Ratio ($\Delta_{edge}$):** Integrate the existing calculation function into the SLSQP optimization loop.

* **Rotational Transform ($\iota$):** Add a strict constraint for the edge rotational transform per field period.

* **Quasi-Isodynamicity (QI) Residual:** Implement the QI residual error metric as defined in current literature.

* **Turbulent Transport ($\chi_{\nabla r}$):** Implement a proxy for flux-surface compression in regions of "bad" curvature.

### 4. Standardized Benchmarks
Implementation of the three reference optimization problems based on the ConStellaration paper:

* **Geometric:** Minimize $\epsilon_{max}$ subject to fixed Aspect Ratio and $\iota$.

* **Simple-to-build QI:** Minimize coil complexity (magnetic gradient scale length) subject to QI and Mirror Ratio constraints.

* **MHD-stable QI:** Multi-objective optimization to balance coil complexity and compactness with MHD stability and transport constraints.