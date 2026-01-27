# app.py
import streamlit as st
import json
import os
import opponent_analysis as oa

import streamlit as st

# --- 1. PASSWORD CHECK FUNCTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "4444":  # <--- SET PASSWORD HERE
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the pass was verified earlier in the session
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Please enter the password to access the app", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
        
    return False

# --- 2. MAIN APP LOGIC ---
if check_password():
    # --- CONFIGURATION ---
    st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")
    
    DATA_PATH = "data/corner_events_all_matches.json"
    
    IMG_PATHS = {
        "def_L": "images/no_names_left.png",
        "def_R": "images/no_names_right.png",
        "att_L": "images/left_side_corner.png",
        "att_R": "images/right_side_corner.png"
    }
    
    # --- CACHED LOADING ---
    @st.cache_data
    def load_local_data(path):
        if not os.path.exists(path): return None
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    
    @st.cache_data
    def get_team_list(json_data):
        return oa.extract_all_teams(json_data)
    
    @st.cache_data
    def get_analysis_results(json_data, team_name):
        # Pass just the canonical name
        return oa.process_corner_data(json_data, team_name)
    
    def get_img_path(key):
        path = IMG_PATHS.get(key)
        return path if path and os.path.exists(path) else None
    
    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    json_data = load_local_data(DATA_PATH)
    
    if not json_data:
        st.error(f"❌ Data file not found at: `{DATA_PATH}`")
        st.stop()
    
    all_teams = get_team_list(json_data)
    selected_team = st.sidebar.selectbox("Select team", all_teams)
    
    # --- MAIN APP ---
    if json_data and selected_team:
        
        with st.spinner(f"Analyzing {selected_team}..."):
            results = get_analysis_results(json_data, selected_team)
            viz_config = oa.get_visualization_coords()
    
        st.title("Opponent analysis - Set Pieces")
        st.markdown(f"**Matches Analyzed:** {results['used_matches']} | **Team:** {selected_team}")
    
        # --- ROW 1: ATTACKING PLOTS ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Att. Corners Left ({results['own_left_count']} corners)")
            st.pyplot(oa.plot_percent_attacking(get_img_path("att_L"), viz_config["att_L"], viz_config["att_centers_L"], results["attacking"]["left_pct"], ""))
        with col2:
            st.subheader(f"Att. Corners Right ({results['own_right_count']} corners)")
            st.pyplot(oa.plot_percent_attacking(get_img_path("att_R"), viz_config["att_R"], viz_config["att_centers_R"], results["attacking"]["right_pct"], ""))
    
        # --- ROW 2: TABLES ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📋 Corner Takers (Left)")
            st.dataframe(results["tables"]["left"], use_container_width=True, hide_index=True)
        with col2:
            st.markdown("##### 📋 Corner Takers (Right)")
            st.dataframe(results["tables"]["right"], use_container_width=True, hide_index=True)
    
        # --- ROW 3: DEFENDING PLOTS ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            # Removed corner count, added explanation
            st.subheader("Def. Corners Left: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["left"]
            st.pyplot(oa.plot_shots_defensive(get_img_path("def_L"), viz_config["def_L"], pcts, tot, ids, ""))
        
        with col2:
            # Removed corner count, added explanation
            st.subheader("Def. Corners Right: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["right"]
            st.pyplot(oa.plot_shots_defensive(get_img_path("def_R"), viz_config["def_R"], pcts, tot, ids, ""))
