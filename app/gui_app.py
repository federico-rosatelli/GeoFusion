
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.io.loader import load_constellaration_dataset
from src.optimization.optimizer import optimize_stellarator
from src.physics import geometry
from src.optimization import objectives


st.set_page_config(page_title="Stellarator Optimizer", layout="wide")

@st.cache_data
def load_data():
    dataset = load_constellaration_dataset()
    if not dataset:
        return None
    return dataset

def plot_surface_plotly(X, Y, Z):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.9)])
    fig.update_layout(title='Plasma Boundary', autosize=True,
                      scene=dict(
                          xaxis_title='X [m]',
                          yaxis_title='Y [m]',
                          zaxis_title='Z [m]',
                          aspectmode='data'
                      ),
                      margin=dict(l=0, r=0, b=0, t=30))
    return fig


def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            color: #FFFFFF;
            background-color: #FF4B4B;
            border-radius: 20px;
            height: 3em;
            width: 100%;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF3333;
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        }
        .metric-card {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        h1 {
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em !important;
            padding-bottom: 20px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #444;
            border-radius: 10px;
            background-color: #1E1E1E;
        }
        [data-testid="stMetricLabel"] {
            color: #E0E0E0 !important;
        }
        [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        header {
            background-color: #0E1117 !important;
        }
        .stHeader {
            background-color: #0E1117 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    apply_custom_css()
    st.title("Stellarator Design")
    
    dataset = load_data()
    if not dataset:
        st.error("Could not load dataset.")
        return

    with st.sidebar:
        st.header("Configuration")
        
        config_index = st.number_input("Select Config Index", min_value=0, max_value=len(dataset)-1, value=0, step=1)
        initial_config = dataset[config_index]
        
        if 'current_config' not in st.session_state or st.session_state.get('config_index_prev') != config_index:
            st.session_state.current_config = initial_config.copy()
            st.session_state.current_config['boundary.r_cos'] = [list(x) for x in initial_config['boundary.r_cos']]
            st.session_state.current_config['boundary.z_sin'] = [list(x) for x in initial_config['boundary.z_sin']]
            st.session_state.config_index_prev = config_index

        current_config = st.session_state.current_config

        st.divider()
        st.header("Parameter Tuning")
        
        with st.expander("Low-Order Modes", expanded=True):
            R_mn = np.array(current_config['boundary.r_cos'])
            Z_mn = np.array(current_config['boundary.z_sin'])
            n_cols = R_mn.shape[1]
            n_center = n_cols // 2

            def get_coeff_val(arr, m, n):
                idx = n_center + n
                if 0 <= m < arr.shape[0] and 0 <= idx < arr.shape[1]:
                    return float(arr[m, idx])
                return 0.0

            def update_coeff(arr_name, m, n, val):
                arr = np.array(st.session_state.current_config[arr_name])
                idx = n_center + n
                if 0 <= m < arr.shape[0] and 0 <= idx < arr.shape[1]:
                    arr[m, idx] = val
                    st.session_state.current_config[arr_name] = arr.tolist()


            def synced_parameter(label, arr_name, m, n, min_v, max_v, step):
                base_key = f"param_{arr_name}_{m}_{n}"
                s_key = f"slider_{base_key}"
                n_key = f"num_{base_key}"

                current_val = get_coeff_val(np.array(st.session_state.current_config[arr_name]), m, n)
                
                if s_key not in st.session_state:
                    st.session_state[s_key] = current_val
                if n_key not in st.session_state:
                    st.session_state[n_key] = current_val

                def update_from_num():
                    new_val = st.session_state[n_key]
                    st.session_state[s_key] = new_val 
                    update_coeff(arr_name, m, n, new_val) 

                def update_from_slider():
                    new_val = st.session_state[s_key]
                    st.session_state[n_key] = new_val 
                    update_coeff(arr_name, m, n, new_val) 

                cols = st.columns([2, 1])
                
                
                cols[0].slider(
                    label=label,
                    min_value=min_v, max_value=max_v,
                    step=step,
                    key=s_key,
                    on_change=update_from_slider
                )
                
                
                cols[1].number_input(
                    label=label, 
                    min_value=min_v, max_value=max_v,
                    step=step,
                    key=n_key,
                    label_visibility="collapsed",
                    on_change=update_from_num
                )


            
            synced_parameter("R(0,0) [Major]", 'boundary.r_cos', 0, 0, 0.1, 10.0, 0.1)
            synced_parameter("R(1,0) [Minor]", 'boundary.r_cos', 1, 0, -2.0, 2.0, 0.05)
            synced_parameter("Z(1,0) [Elong]", 'boundary.z_sin', 1, 0, -2.0, 2.0, 0.05)
            
            synced_parameter("R(0,1) [Ripple]", 'boundary.r_cos', 0, 1, -1.0, 1.0, 0.05)

        st.divider()
        st.header("Optimization")
        problem_type = st.selectbox("Objective", ["simple-to-build", "mhd-stable"])
        max_iter = st.number_input("Iterations", 1, 500, 20)
        
        if st.button("Start Optimization", type="primary"):
            with st.spinner("Optimizing geometry..."):
                optimized_res = optimize_stellarator(current_config, problem_type=problem_type, max_iter=max_iter)
                st.session_state.current_config = optimized_res
                st.success("Optimization Complete!")
                st.rerun()

    
    
    R_mn = np.array(st.session_state.current_config['boundary.r_cos'])
    Z_mn = np.array(st.session_state.current_config['boundary.z_sin'])
    
    initial_c = dataset[config_index]
    R_init = np.array(initial_c['boundary.r_cos'])
    Z_init = np.array(initial_c['boundary.z_sin'])
    
    ar = geometry.calculate_aspect_ratio(R_mn, Z_mn)
    ar0 = geometry.calculate_aspect_ratio(R_init, Z_init)
    
    comp = objectives.calculate_coil_simplicity(R_mn, Z_mn, st.session_state.current_config)
    comp0 = objectives.calculate_coil_simplicity(R_init, Z_init, initial_c)
    
    mhd = objectives.calculate_mhd_stability(R_mn, Z_mn, st.session_state.current_config)
    mhd0 = objectives.calculate_mhd_stability(R_init, Z_init, initial_c)

    st.subheader("Interactive 3D View")
    surface_data = geometry.get_surface_coordinates(st.session_state.current_config)
    
    fig = plot_surface_plotly(surface_data['X'], surface_data['Y'], surface_data['Z'])
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        scene=dict(
            xaxis=dict(backgroundcolor="#0E1117", gridcolor="#444", showbackground=True, zerolinecolor="#444"),
            yaxis=dict(backgroundcolor="#0E1117", gridcolor="#444", showbackground=True, zerolinecolor="#444"),
            zaxis=dict(backgroundcolor="#0E1117", gridcolor="#444", showbackground=True, zerolinecolor="#444"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Aspect Ratio", f"{ar:.2f}", f"{ar-ar0:.2f}")
    m2.metric("Coil Complexity", f"{comp:.4f}", f"{comp-comp0:.4f}", delta_color="inverse")
    m3.metric("MHD Stability (Curvature)", f"{mhd:.4f}", f"{mhd-mhd0:.4f}", delta_color="inverse")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        if 'optimization_history' in st.session_state.current_config:
            st.markdown("### Convergence History")
            st.area_chart(st.session_state.current_config['optimization_history'], color="#FF4B4B")
            
    with c2:
        st.markdown("### Export")
        import json
        st.download_button(
            "Download JSON Config",
            data=json.dumps(st.session_state.current_config, indent=2),
            file_name="stellarator_opt.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
