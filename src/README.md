# Source Code Documentation

Below is a detailed breakdown of the modules and functions, along with their relevance to the project requirements and future work.

## Modules

### `src.io`
Handles data loading and input management.

*   **`loader.py`**
    *   `load_constellaration_dataset(split, cache_dir)`: Loads the "constellaration" dataset from Hugging Face or local cache. This provides the initial stellarator configurations for training or optimization.

### `src.optimization`
Contains the variable optimization logic, objective functions, and constraints.

*   **`constraints.py`**
    *   `check_constraints(R_mn, Z_mn, config_constraints)`: Checks if a given boundary configuration satisfies physical and engineering constraints (e.g., Aspect Ratio).
        *   *To-Do*: Currently implements Aspect Ratio checks. Future work includes adding `Max Elongation`, `Mirror Ratio`, `Rotational Transform`, and `Quasi-Isodynamicity`.

*   **`objectives.py`**
    *   `calculate_coil_simplicity(R_mn, Z_mn, config_data_template)`: Computes the "Simple-to-build" objective using the normalized magnetic field gradient scale length ($e_{L_{\nabla B}}$).
    *   `calculate_mhd_stability(R_mn, Z_mn, config_data_template)`: Computes the proxy for MHD stability using the square of the mean curvature ($H^2$).
        *   *To-Do*: The MHD stability metric is currently a geometric proxy ($H^2$). Future work aims to replace this with the **Vacuum Magnetic Well ($W_{MHD}$)**.

*   **`optimizer.py`**
    *   `optimize_stellarator(initial_config, problem_type, max_iter)`: The main entry point for the optimization loop. Uses `scipy.optimize.minimize` (SLSQP) to adjust boundary coefficients ($R_{mn}, Z_{mn}$) to minimize the loss function defined by `objectives.py` subject to `constraints.py`.

### `src.physics`
Implements the physical and geometric calculations underlying the optimization capabilities.

*   **`geometry.py`**
    *   `get_surface_coordinates(config_data, theta_res, phi_res)`: Converts Fourier coefficients ($R_{mn}, Z_{mn}$) into real-space 3D coordinates ($X, Y, Z$) and calculates partial derivatives ($X_\theta, X_\phi$, etc.).
    *   `calculate_aspect_ratio(R_mn, Z_mn)`: Computes the plasma aspect ratio (Major Radius / Minor Radius).
    *   `calculate_mirror_ratio(B_field)`: Computes the magnetic mirror ratio.
        *   *To-Do*: This function exists but needs to be integrated into the optimization loop as a constraint.
    *   `calculate_curvature(surface_data)`: Calculates the Mean Curvature ($H$) and Gaussian Curvature ($K$) of the plasma surface. Used as a proxy for MHD stability.

*   **`magnetic.py`**
    *   `calculate_magnetic_field_proxy(surface_data)`: Approximates the magnetic field magnitude ($|B| \approx 1/R$).
        *   *To-Do*: This is a placeholder. Future work requires interfacing with full MHD equilibrium solvers like **VMEC++** or **DESC** to obtain accurate magnetic fields.
    *   `calculate_magnetic_field_gradient(B_field, surface_data)`: Computes the gradient of the magnetic field magnitude on the surface, used for the coil simplicity objective.

### `src.visualization`
Tools for inspecting results.

*   **`plotting.py`**
    *   `plot_stellarator_shape(X, Y, Z)`: Renders the 3D plasma boundary surface using Matplotlib.

## Summary of Current Capabilities vs. Requirements

| Feature | Current Implementation | Requirement / Future Work |
| :--- | :--- | :--- |
| **Magnetic Field** | Proxy ($1/R$) in `magnetic.py` | Interface with **VMEC++** or **DESC** |
| **MHD Stability** | Mean Curvature ($H^2$) in `objectives.py` | **Vacuum Magnetic Well ($W_{MHD}$)** |
| **Optimization** | Scipy SLSQP in `optimizer.py` | Expand to multi-objective benchmarks |
| **Constraints** | Aspect Ratio | Add Max Elongation, Mirror Ratio, $\iota$, QI Residual |
