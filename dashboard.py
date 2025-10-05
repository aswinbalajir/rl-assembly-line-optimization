# In dashboard.py

import streamlit as st
import numpy as np
import time
import pandas as pd
import statistics
import os
import random
import plotly.express as px

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from assembly_line_env import AssemblyLineEnv
from simulation_model import AssemblyLineSim, ORDER_BOOK_SIZE

# --- Configuration & Constants ---
EVALUATION_STEPS = 168 # 1 week
EVALUATION_SEED = 42

# --- Model & Environment Loading (Cached) ---
@st.cache_resource
def load_models():
    """Loads the trained agent and the baseline model."""
    model_path = "./best_model/best_model.zip"
    if not os.path.exists(model_path):
        st.error(f"FATAL ERROR: Trained model not found at '{model_path}'. Please ensure training is complete.")
        return None, None
    trained_model = PPO.load(model_path)

    class BaselineModel:
        def predict(self, obs, deterministic=True):
            return np.array([[0, 0, 0]]), None 
    baseline_model = BaselineModel()
    return trained_model, baseline_model

def create_env(scenario_params):
    """Creates a configured and wrapped evaluation environment."""
    stats_path = "vec_normalize_stats.pkl"
    if not os.path.exists(stats_path):
        st.error(f"FATAL ERROR: Normalization stats not found at '{stats_path}'. Please run training script.")
        return None
    env_creator = lambda: AssemblyLineEnv(randomize=False, **scenario_params)
    env = DummyVecEnv([env_creator])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    return env

# --- Main Simulation & Display Function ---
def run_and_display_simulation(policy_model, policy_name, placeholders, scenario_params):
    env = create_env(scenario_params)
    if env is None: return

    obs = env.reset()
    all_completed_parts = []
    animation_data = []
    
    # Initialize UI elements
    placeholders['hour'].metric("Simulated Hour", f"0/{EVALUATION_STEPS}")
    placeholders['throughput'].metric("Total Parts Completed", 0)
    
    for step in range(EVALUATION_STEPS):
        action, _ = policy_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info_dict = info[0]
        
        # --- Update Live Log ---
        st.session_state.event_log = info_dict.get('events', []) + st.session_state.event_log
        log_html = "<br>".join(f"<code>[Min {int(e[0]):>5}]: {e[1]}</code>" for e in st.session_state.event_log[:15])
        scrollable_div = f'<div style="height:400px;overflow-y:auto;border:1px solid #444;padding:10px;border-radius:5px;">{log_html}</div>'
        placeholders['log'].markdown(scrollable_div, unsafe_allow_html=True)
        
        # --- Update KPIs ---
        if info_dict.get('newly_completed_parts'):
            all_completed_parts.extend(info_dict['newly_completed_parts'])
        total_completed = len(all_completed_parts)
        placeholders['hour'].metric("Simulated Hour", f"{step + 1}/{EVALUATION_STEPS}")
        placeholders['throughput'].metric("Total Parts Completed", total_completed)

        # --- Collect data for the Plotly animation ---
        parts_in_stations = info_dict.get('parts_in_stations', {})
        sim_instance = env.get_attr('simulation')[0]
        
        # Combine parts from buffers and stations into one list
        all_active_parts = []
        for part in sim_instance.buffers['buffer12'].items:
            all_active_parts.append((part, "Buffer 1->2"))
        for part in sim_instance.buffers['buffer23'].items:
            all_active_parts.append((part, "Buffer 2->3"))
        for station_name, part in parts_in_stations.items():
            if part:
                all_active_parts.append((part, station_name.replace('_', ' ').title()))
        
        for part, location in all_active_parts:
            animation_data.append({"Hour": step + 1, "Part ID": f"Part-{part['id']}", "Location": location, "Priority": "HIGH" if part['priority'] == 1 else "LOW"})
        
        time.sleep(0.01)
        
    st.success(f"{policy_name} simulation finished!")
    st.balloons()

    # --- Generate Final Report ---
    placeholders['report'].markdown(f"##### Results for: {policy_name}")
    high_prio_parts = [p for p in all_completed_parts if p['priority'] == 1]
    high_prio_late = sum(1 for p in high_prio_parts if p['is_late'])
    avg_cycle_time_high = statistics.mean([p['cycle_time'] for p in high_prio_parts]) if high_prio_parts else 0
    report_col1, report_col2, report_col3 = placeholders['report'].columns(3)
    report_col1.metric("Total Throughput", f"{total_completed} units")
    report_col2.metric("HIGH Prio LATE", f"{high_prio_late} units")
    report_col3.metric("Avg HIGH Prio Cycle Time", f"{avg_cycle_time_high:.0f} mins")

    # --- Create and Display the Plotly Animation ---
    if not animation_data:
        placeholders['plotly_chart'].warning("No part movement was recorded to animate.")
        return
        
    animation_df = pd.DataFrame(animation_data)
    y_axis_order = ["Station 3", "Buffer 2->3", "Repair Station", "Station 2", "Buffer 1->2", "Station 1"]

    # --- FIX: Create an x-position for layout to prevent overlap ---
    # This gives each part at the same location/hour a unique horizontal position (0, 1, 2, ...)
    animation_df['x_pos'] = animation_df.groupby(['Hour', 'Location']).cumcount()

    fig = px.scatter(
        animation_df,
        x="x_pos",              # <-- Use the new position column for the x-axis
        y="Location",
        animation_frame="Hour",
        animation_group="Part ID",
        color="Priority",
        hover_name="Part ID",
        category_orders={"Location": y_axis_order},
        range_x=[-1, animation_df['x_pos'].max() + 1], # Set x-range based on max parts in one spot
        range_y=[-0.5, len(y_axis_order) - 0.5],
        title="Animated Part Tracker",
        color_discrete_map={"HIGH": "red", "LOW": "blue"}
    )

    # --- FIX: Clean up the x-axis since it's only for layout ---
    fig.update_xaxes(showticklabels=False, title=None, zeroline=False)
    fig.update_layout(height=500)
    placeholders['plotly_chart'].plotly_chart(fig, use_container_width=True)


# --- Page Setup & Main Logic ---
st.set_page_config(page_title="RL Digital Twin", layout="wide", initial_sidebar_state="expanded")

if 'event_log' not in st.session_state:
    st.session_state.event_log = []

trained_model, baseline_model = load_models()

st.sidebar.title("⚙️ Controls")
scenario_options = {
    "Original": {"fail_rate": 0.08, "priority_mix": {'HIGH': 0.2, 'LOW': 0.8}, "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}},
    "High Priority Rush": {"fail_rate": 0.08, "priority_mix": {'HIGH': 0.8, 'LOW': 0.2}, "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}},
    "High Failure Rate": {"fail_rate": 0.20, "priority_mix": {'HIGH': 0.2, 'LOW': 0.8}, "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}},
    "New Product Launch": {"fail_rate": 0.08, "priority_mix": {'HIGH': 0.2, 'LOW': 0.8}, "part_mix": {'Type_A': 0.2, 'Type_B': 0.7, 'Type_C': 0.1}},
}
selected_scenario_name = st.sidebar.selectbox("Select a Test Scenario:", scenario_options.keys())
scenario_params = scenario_options[selected_scenario_name]

run_baseline_button = st.sidebar.button("▶️ Run Baseline Policy")
run_agent_button = st.sidebar.button("🚀 Run Expert RL Agent")
st.sidebar.markdown("---")

st.title("🤖 Autonomous Assembly Line Optimization")
st.markdown(f"**Scenario:** {selected_scenario_name}")

kpi_col, report_col = st.columns(2)
with kpi_col:
    st.subheader("📊 Current State")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    hour_placeholder = kpi_col1.empty()
    wip_placeholder = kpi_col2.empty() # This is no longer used for a live chart but kept for layout
    throughput_placeholder = kpi_col3.empty()
with report_col:
    st.subheader("🏆 Final Report")
    report_placeholder = st.empty()
st.markdown("---")

tab1, tab2 = st.tabs(["Live Event Log", "Animated Part Tracker"])
with tab1:
    log_placeholder = st.empty()
with tab2:
    plotly_placeholder = st.empty()

placeholders = {
    "hour": hour_placeholder, "wip": wip_placeholder, "throughput": throughput_placeholder,
    "log": log_placeholder, "plotly_chart": plotly_placeholder, "report": report_placeholder,
}

if run_baseline_button:
    if baseline_model is not None:
        st.session_state.event_log = []
        run_and_display_simulation(baseline_model, "Baseline", placeholders, scenario_params)
if run_agent_button:
    if trained_model is not None:
        st.session_state.event_log = []
        run_and_display_simulation(trained_model, "RL Agent", placeholders, scenario_params)