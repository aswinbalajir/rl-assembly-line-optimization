# dashboard.py

import os
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Local imports
from assembly_line_env import AssemblyLineEnv
from simulation_model import (
    MINS_IN_DAY, 
    DAY_START_MINS, 
    LUNCH_START_MINS, 
    LUNCH_END_MINS, 
    NORMAL_DAY_END_MINS, 
    OVERTIME_DAY_END_MINS
)

# --- Configuration Constants ---
EVALUATION_STEPS = 1680  # Corresponds to 1 week (1680 steps * 6 mins = 10,080 mins)
EVALUATION_SEED = 42
MODEL_PATH = "./best_model/best_model.zip"
FALLBACK_MODEL_PATH = "ppo_assembly_line_model.zip"

# --- Styling Configuration ---
def inject_custom_css():
    """Injects custom CSS to style the dashboard components without emojis."""
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        
        /* Container for station status cards */
        .station-box {
            background-color: #262730; 
            padding: 10px; 
            border-radius: 8px; 
            border: 1px solid #444;
            text-align: center; 
            height: 140px; 
            display: flex; 
            flex-direction: column;
            justify-content: center; 
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .station-title { 
            font-size: 0.85em; 
            color: #aaa; 
            margin-bottom: 8px; 
            text-transform: uppercase; 
            letter-spacing: 1px;
        }
        
        /* High Priority Part Styling (Red Border) */
        .part-high { 
            color: #ff4b4b; 
            font-weight: 900; 
            font-size: 1.4em; 
            border: 2px solid #ff4b4b; 
            padding: 5px 10px; 
            border-radius: 6px;
            background-color: rgba(255, 75, 75, 0.1);
        }
        
        /* Low Priority Part Styling (Blue Border) */
        .part-low { 
            color: #4b9eff; 
            font-weight: 700; 
            font-size: 1.1em; 
            border: 1px solid #4b9eff;
            padding: 4px 8px; 
            border-radius: 6px;
        }
        
        .status-idle { color: #555; font-style: italic; font-size: 0.9em; }
        .status-busy { color: #00cc96; font-weight: bold; font-size: 0.8em; margin-top: 8px; }
        
        /* Repair Status Styling (Orange) */
        .status-repair { 
            color: #ffa500; 
            font-weight: bold; 
            font-size: 0.8em; 
            margin-top: 8px;
            border: 1px solid #ffa500; 
            padding: 2px 5px; 
            border-radius: 4px;
            background-color: rgba(255, 165, 0, 0.1);
        }
        
        /* Scrollable Log Container */
        .terminal-log {
            background-color: #000000; 
            border: 1px solid #333; 
            border-radius: 6px;
            padding: 12px; 
            font-family: 'Courier New', monospace; 
            font-size: 0.85em;
            height: 280px; 
            overflow-y: auto;
            color: #ccc;
        }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #222; padding-bottom: 2px; }
        .log-timestamp { color: #00cc96; margin-right: 8px; font-weight: bold; }
        .log-urgent { color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)

# --- Model Loading ---

@st.cache_resource
def load_ppo_model() -> Optional[PPO]:
    """
    Loads and caches the trained PPO model.
    
    This function is cached because loading the model from disk is an expensive operation.
    It does NOT load the Baseline agent, ensuring the Baseline is fresh for every run.
    
    Returns:
        PPO: The loaded Stable Baselines3 model, or None if not found.
    """
    if os.path.exists(MODEL_PATH):
        return PPO.load(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return PPO.load(FALLBACK_MODEL_PATH)
    return None

class BaselineFIFO:
    """
    A heuristic baseline agent implementing a First-In-First-Out (FIFO) strategy.
    
    Policy:
    1. Always select the first (oldest) order in the book (Index 0).
    2. Never halt production voluntarily.
    3. Never authorize overtime.
    """
    def predict(self, obs, deterministic=True):
        # Action format: [Part_Index, Halt_Flag, Overtime_Flag]
        return np.array([[0, 0, 0]]), None

# --- Simulation Setup Helpers ---

def create_env(scenario_params: Dict[str, Any]) -> DummyVecEnv:
    """Creates a vectorized environment with the specified scenario parameters."""
    env_creator = lambda: AssemblyLineEnv(randomize=False, **scenario_params)
    return DummyVecEnv([env_creator])

def calculate_wip(sim_instance) -> int:
    """
    Calculates the total Work-In-Progress (WIP) in the system.
    
    Includes parts currently processing in stations, parts waiting in buffers,
    and parts in station input queues.
    """
    active_count = len([p for p in sim_instance.parts_in_stations.values() if p is not None])
    buffer_count = sum(len(b.items) for b in sim_instance.buffers.values())
    queue_count = sum(len(s.queue) for s in sim_instance.stations.values())
    return active_count + buffer_count + queue_count

def format_time(total_minutes: float) -> str:
    """Converts absolute simulation minutes into a Day Hour:Minute string."""
    day = int(total_minutes // MINS_IN_DAY)
    remaining = int(total_minutes % MINS_IN_DAY)
    hour = remaining // 60
    minute = remaining % 60
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return f"{day_names[day % 7]} {hour:02d}:{minute:02d}"

# --- UI Rendering Functions ---

def render_station_card(container, title: str, part: Optional[Dict], is_repair: bool = False):
    """Renders a status card for a single workstation."""
    content = "<div class='status-idle'>IDLE</div>"
    if part:
        # Determine CSS class based on priority
        p_class = "part-high" if part['priority'] == 1 else "part-low"
        p_label = "URGENT" if part['priority'] == 1 else "STD"
        
        status_html = "<div class='status-busy'>PROCESSING</div>"
        if is_repair:
            status_html = "<div class='status-repair'>REPAIRING</div>"
            
        content = f"<div class='{p_class}'>{part['type']}<br><span style='font-size:0.6em'>{p_label}</span></div>{status_html}"

    html = f"<div class='station-box'><div class='station-title'>{title}</div>{content}</div>"
    container.markdown(html, unsafe_allow_html=True)

def render_transcript(container, logs: List[str]):
    """Renders the scrolling log of completed parts."""
    # Display only the last 15 log entries
    log_html = "".join([f"<div class='log-entry'>{entry}</div>" for entry in logs[-15:]])
    container.markdown(f"<div class='terminal-log'>{log_html}</div>", unsafe_allow_html=True)

def update_factory_ui(ui_map: Dict, sim_instance, logs: List[str], is_overtime_action: bool = False):
    """Updates all UI elements for a single factory instance (Stations, Buffers, Status)."""
    
    # 1. Update Stations
    render_station_card(ui_map['s1'], "Station 1", sim_instance.parts_in_stations.get('station1'))
    
    # Logic for Station 2 which shares display space with Repair Station
    active_part_s2 = sim_instance.parts_in_stations.get('repair_station') or sim_instance.parts_in_stations.get('station2')
    is_repair = sim_instance.parts_in_stations.get('repair_station') is not None
    render_station_card(ui_map['s2'], "Station 2", active_part_s2, is_repair)
    
    render_station_card(ui_map['s3'], "Station 3", sim_instance.parts_in_stations.get('station3'))

    # 2. Update Buffers
    # Buffer 1->2
    b1_count = len(sim_instance.buffers['buffer12'].items) + len(sim_instance.stations['station2'].queue)
    ui_map['b1_prog'].progress(min(b1_count / 15, 1.0))
    ui_map['b1_text'].markdown(f"<div style='text-align:center; font-size:0.7em;'>Queue: {b1_count}</div>", unsafe_allow_html=True)
    
    # Buffer 2->3
    b2_count = len(sim_instance.buffers['buffer23'].items) + len(sim_instance.stations['station3'].queue)
    ui_map['b2_prog'].progress(min(b2_count / 15, 1.0))
    ui_map['b2_text'].markdown(f"<div style='text-align:center; font-size:0.7em;'>Queue: {b2_count}</div>", unsafe_allow_html=True)

    # 3. Update Logs
    render_transcript(ui_map['log'], logs)

    # 4. Update Operational Status Indicator
    now = sim_instance.env.now
    day_idx = int((now // MINS_IN_DAY) % 7)
    time_of_day = int(now % MINS_IN_DAY)
    end_of_day = OVERTIME_DAY_END_MINS if sim_instance.overtime_active_today else NORMAL_DAY_END_MINS
    
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    day_name = days[day_idx]
    
    if day_idx == 6:
        status_html = f"<b>{day_name}</b> — <span style='color:#ff4b4b;'>WEEKEND CLOSED</span>"
    elif time_of_day < DAY_START_MINS or time_of_day >= end_of_day:
        status_html = f"<b>{day_name}</b> — <span style='color:#ff4b4b;'>OFF SHIFT</span>"
    elif LUNCH_START_MINS <= time_of_day < LUNCH_END_MINS:
        status_html = f"<b>{day_name}</b> — <span style='color:#ffa500;'>LUNCH BREAK</span>"
    else: 
        if is_overtime_action:
            ot_html = " <span style='color:#ffa500; font-weight:bold; animation: blink 1s infinite;'>(OVERTIME ACTIVE)</span>"
        else:
            ot_html = ""
        status_html = f"<b>{day_name}</b> — <span style='color:#00cc96;'>OPERATIONAL</span>{ot_html}"
        
    ui_map['status'].markdown(f"<div style='text-align:center; font-size:0.9em; padding:5px; border:1px solid #333; border-radius:4px;'>{status_html}</div>", unsafe_allow_html=True)

# --- Main Simulation Loop ---

def run_simulation_loop(trained_model, rl_ui, base_ui, metrics_ui, chart_ui, params, speed):
    """
    Executes the head-to-head simulation.
    
    Args:
        trained_model: The cached PPO agent.
        rl_ui (dict): Streamlit placeholders for the RL column.
        base_ui (dict): Streamlit placeholders for the Baseline column.
        metrics_ui: Placeholder for top-level metrics.
        chart_ui: Placeholder for the WIP chart.
        params (dict): Simulation configuration parameters.
        speed (float): Sleep time between updates.
    """
    
    # 1. Initialize Environments
    env_base = create_env(params)
    env_rl = create_env(params)
    env_base.seed(EVALUATION_SEED)
    env_rl.seed(EVALUATION_SEED)
    # Access inner SimPy environment instances
    real_env_base = env_base.envs[0]
    real_env_rl = env_rl.envs[0]
    
    # Reset Environments
    obs_base = env_base.reset()
    obs_rl = env_rl.reset()
    
    # 2. Instantiate Baseline Agent Here (Fix for Caching Bug)
    # This ensures a fresh instance with reset counters for every run.
    baseline_model = BaselineFIFO()
    
    # 3. Initialize Tracking Variables
    data = {
        'rl_wip': [], 'base_wip': [],
        'rl_completed': 0, 'base_completed': 0,
        'rl_late': 0, 'base_late': 0
    }
    
    # Daily production counts [Mon, Tue, ..., Sun]
    daily_stats = {
        'rl': [0] * 7,
        'base': [0] * 7
    }
    # Priority breakdown
    prio_stats = {
        'rl': {'HIGH': 0, 'LOW': 0},
        'base': {'HIGH': 0, 'LOW': 0}
    }
    
    logs_rl = ["<span class='log-timestamp'>00:00</span> Simulation Started"]
    logs_base = ["<span class='log-timestamp'>00:00</span> Simulation Started"]

    # Configuration for UI smoothness
    sim_step_mins = real_env_rl.step_duration
    
    progress_bar = st.progress(0)
    
    # 4. Main Step Loop
    for step in range(EVALUATION_STEPS):
        
        # A. Predict Actions
        action_base, _ = baseline_model.predict(obs_base, deterministic=True)
        action_rl, _ = trained_model.predict(obs_rl, deterministic=True)
        rl_is_overtime = action_rl.flatten()[2] > 0
        # B. Apply Actions (Wrapper function)
        def apply_action_to_sim(env_instance, action_vec):
            # Unpack: [Part_Index, Halt_Flag, Overtime_Flag]
            a = action_vec.flatten()
            part_idx, halt, ot = int(a[0]), int(a[1]), int(a[2])    
            
            sim = env_instance.simulation
            sim.set_source_status(bool(halt))
            sim.set_overtime_status(bool(ot))
            
            # Release part only if not halted
            if not bool(halt):
                try: 
                    sim.release_part(part_idx)
                except Exception: 
                    pass

        apply_action_to_sim(real_env_base, action_base)
        apply_action_to_sim(real_env_rl, action_rl)
        
        # C. Advance Simulation Time
        real_env_base.simulation.run(sim_step_mins)
        real_env_rl.simulation.run(sim_step_mins)
        
        # D. Collect Results
        _, res_base = real_env_base.simulation.get_kpis_and_state()
        _, res_rl = real_env_rl.simulation.get_kpis_and_state()
        def process_completions(res, log_list, key_prefix, stats_key):
            """
            Processes completed parts to update global counters, daily statistics, 
            priority breakdown, and the event log.
            """
            for p in res.get('newly_completed_parts', []):
                # --- 1. Update Global Counts ---
                data[f'{key_prefix}_completed'] += 1
                if p['priority'] == 1 and p['is_late']:
                    data[f'{key_prefix}_late'] += 1
                
                # --- 2. Update Daily & Priority Stats (CRITICAL for Bottom Table) ---
                # Calculate which day of the week (0=Mon, 6=Sun) the part finished
                day_idx = int((p['finish_time'] // MINS_IN_DAY) % 7)
                daily_stats[stats_key][day_idx] += 1
                
                # Update High/Low priority counts
                p_type_stats = 'HIGH' if p['priority'] == 1 else 'LOW'
                prio_stats[stats_key][p_type_stats] += 1
                
                # --- 3. Update Event Log ---
                t_str = format_time(p['finish_time'])
                p_cls = "log-urgent" if p['priority'] == 1 else ""
                status_text = "LATE" if p['is_late'] else "OK"
                
                # Retrieve Part Type (e.g., 'Type_A') or fallback to ID if missing
                part_label = p.get('type', f"Part-{p['type']}")
                
                # Log format: "Mon 14:00  Type_A Done (OK)"
                log_list.append(f"<span class='log-timestamp'>{t_str}</span> <span class='{p_cls}'>{part_label} Done ({status_text})</span>")

        # Keep these call signatures exactly as they were for the first function
        process_completions(res_base, logs_base, 'base', 'base')
        process_completions(res_rl, logs_rl, 'rl', 'rl')

        # Helper to process completions and update logs
        # def process_completions(res, log_list, key_completed, key_late):
        #     for p in res.get('newly_completed_parts', []):
        #         data[key_completed] += 1
        #         is_high = p['priority'] == 1
        #         is_late = p['is_late']
                
        #         if is_high and is_late:
        #             data[key_late] += 1
                
        #         # Format log entry
        #         t_str = format_time(p['finish_time'])
        #         p_cls = "log-urgent" if is_high else ""
        #         status_text = "LATE" if is_late else "OK"
        #         log_list.append(f"<span class='log-timestamp'>{t_str}</span> <span class='{p_cls}'>Part-{p['id']} Done ({status_text})</span>")
        
        # process_completions(res_base, logs_base, 'base_completed', 'base_late')
        # process_completions(res_rl, logs_rl, 'rl_completed', 'rl_late')
        
        # E. Update UI (throttled to every 3 steps for performance)
        if step % 3 == 0:
            # Update WIP Charts
            # data['rl_wip'].append(calculate_wip(real_env_rl.simulation))
            # data['base_wip'].append(calculate_wip(real_env_base.simulation))
            
            # chart_df = pd.DataFrame({
            #     "RL Agent (Smart)": data['rl_wip'],
            #     "Baseline (FIFO)": data['base_wip']
            # })
            # chart_ui.line_chart(chart_df.tail(50))
            
            # Update Metrics
            with metrics_ui.container():
                c1, c2, c3, c4 = st.columns(4)
                curr_time = real_env_rl.simulation.env.now
                c1.metric("Simulation Time", format_time(curr_time))
                c2.metric("Production Volume", f"{data['rl_completed']}", delta=data['rl_completed'] - data['base_completed'])
                
                fails_saved = data['base_late'] - data['rl_late']
                c3.metric("Critical Failures Prevented", f"{fails_saved}", delta=fails_saved)
                
                c4.metric("RL Late Orders", f"{data['rl_late']}", delta_color="inverse")

            # Update Factory Visuals
            
           
            update_factory_ui(rl_ui, real_env_rl.simulation, logs_rl, is_overtime_action=rl_is_overtime)
            update_factory_ui(base_ui, real_env_base.simulation, logs_base, is_overtime_action=False)
          
            
            def render_summary_table(container, d_stats, p_stats, total):
                days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                # Create a mini HTML table row for days
                day_row = "".join([f"<td style='padding:4px; border:1px solid #444;'>{d}<br><b>{c}</b></td>" for d,c in zip(days, d_stats)])
                
                html = f"""
                <table style='width:100%; font-size:0.8em; text-align:center; border-collapse:collapse;'>
                    <tr style='background-color:#222; color:#aaa;'>{day_row}</tr>
                </table>
                <div style='margin-top:5px; font-size:0.85em; display:flex; justify-content:space-between; padding:0 10px;'>
                    <span><b>Total:</b> {total}</span>
                    <span style='color:#ff4b4b;'><b>High:</b> {p_stats['HIGH']}</span>
                    <span style='color:#4b9eff;'><b>Low:</b> {p_stats['LOW']}</span>
                </div>
                """
                container.markdown(html, unsafe_allow_html=True)

            render_summary_table(rl_ui['daily_stats'], daily_stats['rl'], prio_stats['rl'], data['rl_completed'])
            render_summary_table(base_ui['daily_stats'], daily_stats['base'], prio_stats['base'], data['base_completed'])
            
            # Progress Bar and Sleep
            progress_bar.progress(min(step / EVALUATION_STEPS, 1.0))
            time.sleep(speed)
            
        # F. Update Observations for Next Step
        obs_base = real_env_base._get_obs()
        obs_rl = real_env_rl._get_obs()

    st.success("Simulation Run Complete.")

# --- Main Page Layout and Logic ---

st.set_page_config(page_title="Factory Digital Twin", layout="wide")
inject_custom_css()

st.sidebar.header("Simulation Controls")

# Scenario Definitions
scenario_map = {
    "Original": {
        "fail_rate": 0.08, 
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8}, 
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}
    },
    "High Priority Rush": {
        "fail_rate": 0.08, 
        "priority_mix": {'HIGH': 0.8, 'LOW': 0.2}, 
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}
    },
    "High Failure Rate": {
        "fail_rate": 0.20, 
        "priority_mix": {'HIGH': 0.2, 'LOW': 0.8}, 
        "part_mix": {'Type_A': 0.6, 'Type_B': 0.25, 'Type_C': 0.15}
    },
}

sel_scenario = st.sidebar.selectbox("Test Scenario", list(scenario_map.keys()))
run_speed = st.sidebar.slider("Simulation Speed", 0.01, 0.5, 0.05)

st.title("Factory Digital Twin: RL vs Baseline")
st.markdown("Real-time comparison of the Reinforcement Learning agent against standard FIFO rules.")

# UI Placeholder Setup
metrics_container = st.empty()
chart_placeholder = st.empty()

col1, col2 = st.columns(2)

def build_factory_column(col, title):
    """Helper to construct the UI layout for a single factory column."""
    phs = {}
    with col:
        st.subheader(title)
        phs['status'] = st.empty()
        
        # ... (Station and Buffer columns remain the same) ...
        r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns([1, 0.2, 1, 0.2, 1])
        phs['s1'] = r1_c1.empty()
        phs['s2'] = r1_c3.empty()
        phs['s3'] = r1_c5.empty()
        
        with r1_c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            phs['b1_prog'] = st.empty()
            phs['b1_text'] = st.empty()
        with r1_c4:
            st.markdown("<br>", unsafe_allow_html=True)
            phs['b2_prog'] = st.empty()
            phs['b2_text'] = st.empty()
            
        st.markdown("###### Live Event Log")
        phs['log'] = st.empty()
        
        # --- Daily Summary Table Placeholder ---
        st.markdown("---")
        phs['daily_stats'] = st.empty()
        
    return phs

rl_placeholders = build_factory_column(col1, "RL Agent")
base_placeholders = build_factory_column(col2, "Baseline (FIFO)")

# Start Button Logic
if st.sidebar.button("Start Simulation", type="primary"):
    # Load the RL model (cached)
    trained_agent = load_ppo_model()
    
    if trained_agent:
        # Pass the trained agent to the loop. 
        # Note: Baseline is NOT passed here, it is created inside the loop.
        run_simulation_loop(
            trained_agent, 
            rl_placeholders, 
            base_placeholders, 
            metrics_container, 
            chart_placeholder, 
            scenario_map[sel_scenario], 
            run_speed
        )
    else:
        st.error(f"Model not found at {MODEL_PATH}. Please train the agent first.")