import streamlit as st
import tempfile
import os
from illuminance_sim import run_simulation

st.set_page_config(page_title="Illuminance Simulation", layout="wide")
st.title("💡 Illuminance Simulation from IES File")

# File uploader
uploaded_file = st.file_uploader("Upload IES File", type=["ies"])

# Room dimensions
col1, col2, col3 = st.columns(3)
with col1:
    room_width = st.number_input("Room Width (m)", min_value=1.0, value=5.0)
with col2:
    room_depth = st.number_input("Room Depth (m)", min_value=1.0, value=5.0)
with col3:
    room_height = st.number_input("Room Height (m)", min_value=2.0, value=3.0)

# Luminaire grid
col4, col5, col6 = st.columns(3)
with col4:
    grid_x = st.number_input("Grid X (luminaires)", min_value=1, value=2)
with col5:
    grid_y = st.number_input("Grid Y (luminaires)", min_value=1, value=2)
with col6:
    mount_height = st.number_input("Mounting Height (m)", min_value=0.1, value=2.8)

# Planes
st.subheader("Planes to Calculate")
planes_spec = []
if st.checkbox("Workplane", value=True):
    wp_height = st.number_input("Workplane Height (m)", min_value=0.0, value=0.8)
    planes_spec.append({"type": "workplane", "z": wp_height, "u_steps": 40, "v_steps": 40})
if st.checkbox("Walls"):
    planes_spec.extend([
        {"type": "wall", "which": "x0"},
        {"type": "wall", "which": "xw"},
        {"type": "wall", "which": "y0"},
        {"type": "wall", "which": "yd"},
    ])
if st.checkbox("Ceiling"):
    planes_spec.append({"type": "ceiling"})

# Grid resolution
col7, col8 = st.columns(2)
with col7:
    u_steps = st.number_input("U Steps", min_value=10, value=40)
with col8:
    v_steps = st.number_input("V Steps", min_value=10, value=40)

if uploaded_file and st.button("Run Simulation"):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ies_path = os.path.join(tmpdir, uploaded_file.name)
        with open(tmp_ies_path, "wb") as f:
            f.write(uploaded_file.read())

        # Build luminaire positions in a grid
        xs = [room_width/(grid_x+1)*(i+1) for i in range(grid_x)]
        ys = [room_depth/(grid_y+1)*(j+1) for j in range(grid_y)]
        lum_positions = [(x, y, mount_height) for x in xs for y in ys]

        # Run simulation
        results = run_simulation(
            ies_path=tmp_ies_path,
            room_size=(room_width, room_depth, room_height),
            lum_positions=lum_positions,
            planes_spec=planes_spec,
            out_dir=tmpdir
        )

        # Display results
        for plane_name, rec in results.items():
            st.subheader(plane_name)
            st.image(rec["png"])
            with open(rec["csv"], "rb") as f:
                st.download_button(
                    f"Download {plane_name} CSV",
                    f,
                    file_name=os.path.basename(rec["csv"])
                )
