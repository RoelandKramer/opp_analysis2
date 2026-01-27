import streamlit as st
import json
import os
import opponent_analysis as oa

# --- CONFIGURATION ---
st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

# Define your local paths here
DATA_PATH = "data/corner_events_all_matches.json"

# Define local image paths (Make sure these match your filenames in the images folder)
IMG_PATHS = {
    "def_L": "images/no_names_left.png",
    "def_R": "images/no_names_right.png",
    "att_L": "images/left_side_corner.png",
    "att_R": "images/right_side_corner.png"
}

# --- HELPER: AUTO ALIAS LOGIC ---
def get_smart_aliases(team_name):
    aliases = {team_name}
    if team_name.startswith("FC "): aliases.add(team_name[3:].strip())
    if team_name.endswith(" FC"): aliases.add(team_name[:-3].strip())
    return list(aliases)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_local_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_team_list(json_data):
    return oa.extract_all_teams(json_data)

@st.cache_data
def get_analysis_results(json_data, aliases):
    return oa.process_corner_data(json_data, aliases)

def get_img_path(key):
    """Returns the path if file exists, else None (white bg)"""
    path = IMG_PATHS.get(key)
    if path and os.path.exists(path):
        return path
    return None

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# 1. Load Data Automatically
json_data = load_local_data(DATA_PATH)

if not json_data:
    st.error(f"❌ Could not find data file at: `{DATA_PATH}`")
    st.info("Please make sure you have created the 'data' folder and placed your JSON file inside it.")
    st.stop() # Stop execution here if no data

# 2. Team Selector
all_teams = get_team_list(json_data)
selected_team = st.sidebar.selectbox("Select team", all_teams)

# 3. Process Selection
selected_team_aliases = []
if selected_team:
    selected_team_aliases = get_smart_aliases(selected_team)

# --- MAIN APP LAYOUT ---
if json_data and selected_team_aliases:
    
    # Run Analysis
    with st.spinner(f"Analyzing {selected_team_aliases[0]}..."):
        results = get_analysis_results(json_data, selected_team_aliases)
        viz_config = oa.get_visualization_coords()

    # Header
    st.title("Opponent analysis - Set Pieces")
    st.markdown(f"**Matches Analyzed:** {results['used_matches']} | **Team:** {selected_team_aliases[0]}")

    # --- ROW 1: ATTACKING PLOTS ---
    col_att_L, col_att_R = st.columns(2)
    
    with col_att_L:
        st.subheader("Att. Corners Left")
        fig = oa.plot_percent_attacking(
            get_img_path("att_L"), 
            viz_config["att_L"], 
            viz_config["att_centers_L"], 
            results["attacking"]["left_pct"], 
            f"Left Side Attacking ({results['own_left_count']} corners)"
        )
        st.pyplot(fig)

    with col_att_R:
        st.subheader("Att. Corners Right")
        fig = oa.plot_percent_attacking(
            get_img_path("att_R"), 
            viz_config["att_R"], 
            viz_config["att_centers_R"], 
            results["attacking"]["right_pct"], 
            f"Right Side Attacking ({results['own_right_count']} corners)"
        )
        st.pyplot(fig)

    # --- ROW 2: TABLES ---
    st.divider()
    
    st.markdown("##### 📋 Corner Takers (Left)")
    st.dataframe(results["tables"]["left"], use_container_width=True, hide_index=True)
    
    st.markdown("##### 📋 Corner Takers (Right)")
    st.dataframe(results["tables"]["right"], use_container_width=True, hide_index=True)

    # --- ROW 3: DEFENDING PLOTS ---
    st.divider()
    col_def_L, col_def_R = st.columns(2)

    with col_def_L:
        st.subheader("Def. Corners Left")
        tot, ids, pcts = results["defensive"]["left"]
        fig = oa.plot_shots_defensive(
            get_img_path("def_L"), 
            viz_config["def_L"], 
            pcts, tot, ids, 
            f"Opp. Corners FROM LEFT"
        )
        st.pyplot(fig)

    with col_def_R:
        st.subheader("Def. Corners Right")
        tot, ids, pcts = results["defensive"]["right"]
        fig = oa.plot_shots_defensive(
            get_img_path("def_R"), 
            viz_config["def_R"], 
            pcts, tot, ids, 
            f"Opp. Corners FROM RIGHT"
        )
        st.pyplot(fig)
