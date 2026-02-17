import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.io.loader import load_constellaration_dataset
from src.optimization.optimizer import optimize_stellarator
from src.ml.ensamble import StellaratorEnsemble
import torch
from src.physics import geometry
from src.optimization import objectives


st.set_page_config(page_title="Stellarator Optimizer", layout="wide")

@st.cache_data
def load_data():
    dataset = load_constellaration_dataset()
    if not dataset:
        return None
    return dataset

@st.cache_resource
def load_models():
    models = StellaratorEnsemble("configs/conf.yaml", "configs/model_struct.json")
    models.load_models()
    return models

def plot_surface_plotly(X, Y, Z):
    X = np.concatenate((X, X[0:1, :]), axis=0)
    Y = np.concatenate((Y, Y[0:1, :]), axis=0)
    Z = np.concatenate((Z, Z[0:1, :]), axis=0)

    X = np.concatenate((X, X[:, 0:1]), axis=1)
    Y = np.concatenate((Y, Y[:, 0:1]), axis=1)
    Z = np.concatenate((Z, Z[:, 0:1]), axis=1)

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
    
    models = load_models()

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
                    print(arr)


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


            
            st.markdown("**Global Size**")
            synced_parameter("R(0,0) [Major Radius]", 'boundary.r_cos', m=0, n=0, min_v=0.5, max_v=15.0, step=0.1)

            st.markdown("**Cross Section Shape**")
            synced_parameter("R(1,0) [Minor Radius]", 'boundary.r_cos', m=1, n=0, min_v=-3.0, max_v=3.0, step=0.05)
            # synced_parameter("Z(1,0) [Elongation]", 'boundary.z_sin', m=1, n=0, min_v=-3.0, max_v=3.0, step=0.05)

            st.markdown("**3D Helical Shaping (Twist)**")
            synced_parameter("R(1,1) [Helical R]", 'boundary.r_cos', m=1, n=1, min_v=-2.0, max_v=2.0, step=0.05)
            synced_parameter("Z(1,1) [Helical Z]", 'boundary.z_sin', m=1, n=1, min_v=-2.0, max_v=2.0, step=0.05)
            

        st.divider()
        with st.expander("Import JSON Config", expanded=False):
            tab_upload, tab_paste = st.tabs(["Upload File", "Paste Text"])
            
            with tab_upload:
                uploaded_file = st.file_uploader("Choose JSON File", type=["json"])
                if uploaded_file is not None:
                    if st.button("Load from File", key="btn_load_file"):
                        try:
                            import json
                            data = json.load(uploaded_file)
                            if 'boundary.r_cos' in data and 'boundary.z_sin' in data:
                                st.session_state.current_config = data
                                st.success("Config loaded!")
                                st.rerun()
                            else:
                                st.error("Invalid JSON: missing boundary coefficients")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab_paste:
                json_text = st.text_area("Paste JSON here", height=150)
                if st.button("Load from Text", key="btn_load_text"):
                    try:
                        import json
                        data = json.loads(json_text)
                        if 'boundary.r_cos' in data and 'boundary.z_sin' in data:
                            st.session_state.current_config = data
                            st.success("Config loaded!")
                            st.rerun()
                        else:
                            st.error("Invalid JSON")
                    except Exception as e:
                        st.error(f"Error parsing JSON: {e}")

        current_config = st.session_state.current_config

        st.divider()

        st.header("Optimization")
        problem_type = st.selectbox("Objective", ["simple-to-build", "mhd-stable", "GeoFusion-nn"], index=0)
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
    st.plotly_chart(fig, width='stretch')

    ar = geometry.calculate_aspect_ratio(R_mn, Z_mn)
    vol = geometry.calculate_volume(R_mn, Z_mn)
    mr = geometry.calculate_geometric_mirror_ratio(R_mn, Z_mn)
    mr = geometry.calculate_geometric_mirror_ratio(R_mn, Z_mn)
    
    # Model predictions
    input_vector = np.concatenate([R_mn.flatten(), Z_mn.flatten()])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(models.device)
    preds = models.predict(input_tensor)
    
    mhd = preds['w_mhd'].item()
    
    R_init = np.array(st.session_state.current_config['boundary.r_cos'])
    Z_init = np.array(st.session_state.current_config['boundary.z_sin'])
    
    ar0 = geometry.calculate_aspect_ratio(R_init, Z_init)
    vol0 = geometry.calculate_volume(R_init, Z_init)
    mr0 = geometry.calculate_geometric_mirror_ratio(R_init, Z_init)
    mr0 = geometry.calculate_geometric_mirror_ratio(R_init, Z_init)
    
    input_vector0 = np.concatenate([R_init.flatten(), Z_init.flatten()])
    input_tensor0 = torch.tensor(input_vector0, dtype=torch.float32).unsqueeze(0).to(models.device)
    preds0 = models.predict(input_tensor0)
    
    mhd0 = preds0['w_mhd'].item()

    st.markdown("### Metrics Dashboard")
    
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric(
        label="Aspect Ratio", 
        value=f"{ar:.2f}", 
        delta=f"{ar-ar0:.2f}",
        help="Aspect ratio of the stellarator"
    )
    
    m2.metric(
        label="Volume [m³]", 
        value=f"{vol:.3f}", 
        delta=f"{vol-vol0:.3f}",
        help="Volume of the stellarator"
    )
    
    m3.metric(
        label="Mirror Ratio", 
        value=f"{mr:.3f}", 
        delta=f"{mr-mr0:.3f}", 
        delta_color="inverse",
        help="Mirror ratio of the stellarator"
    )
    
    m4.metric(
        label="MHD Stability", 
        value=f"{mhd:.4f}", 
        delta=f"{mhd-mhd0:.4f}", 
        delta_color="inverse",
        help="MHD stability of the stellarator"
    )

    c1, c2 = st.columns([2, 1])
    
    with c1:
        if 'optimization_history' in st.session_state.current_config and st.session_state.current_config['optimization_history']:
            st.markdown("### Convergence History")
            #st.area_chart(st.session_state.current_config['optimization_history'], color="#FF4B4B")
            st.line_chart(st.session_state.current_config['optimization_history'], color="#FF4B4B")
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
