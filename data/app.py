# app.py
import streamlit as st
import json
import opponent_analysis as oa

# --- PAGE CONFIG ---
st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

# --- HELPER: AUTO ALIAS LOGIC ---
def get_smart_aliases(team_name):
    aliases = {team_name}
    if team_name.startswith("FC "): aliases.add(team_name[3:].strip())
    if team_name.endswith(" FC"): aliases.add(team_name[:-3].strip())
    return list(aliases)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_json_data(file):
    return json.load(file)

@st.cache_data
def get_team_list(json_data):
    return oa.extract_all_teams(json_data)

@st.cache_data
def get_analysis_results(json_data, aliases):
    return oa.process_corner_data(json_data, aliases)

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload Corner Events JSON", type="json")
selected_team_aliases = []
json_data = None

if uploaded_file:
    try:
        json_data = load_json_data(uploaded_file)
        all_teams = get_team_list(json_data)
        selected_team = st.sidebar.selectbox("Select team", all_teams)
        if selected_team:
            selected_team_aliases = get_smart_aliases(selected_team)
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Pitch Images (Optional)")
img_def_L = st.sidebar.file_uploader("Defensive Left Bg", type=["png", "jpg"], key="dl")
img_def_R = st.sidebar.file_uploader("Defensive Right Bg", type=["png", "jpg"], key="dr")
img_att_L = st.sidebar.file_uploader("Attacking Left Bg", type=["png", "jpg"], key="al")
img_att_R = st.sidebar.file_uploader("Attacking Right Bg", type=["png", "jpg"], key="ar")

# --- MAIN APP LAYOUT ---
if not uploaded_file:
    st.title("Opponent analysis - Set Pieces")
    st.info("Please upload the corner events JSON file in the sidebar to begin.")
elif json_data and selected_team_aliases:
    
    with st.spinner(f"Analyzing {selected_team_aliases[0]}..."):
        results = get_analysis_results(json_data, selected_team_aliases)
        # Load visualization polygons from the helper file
        viz_config = oa.get_visualization_coords()

    st.title("Opponent analysis - Set Pieces")
    st.markdown(f"**Matches Analyzed:** {results['used_matches']} | **Team:** {selected_team_aliases[0]}")

    # --- ROW 1: ATTACKING PLOTS ---
    col_att_L, col_att_R = st.columns(2)
    
    with col_att_L:
        st.subheader("Att. Corners Left")
        fig = oa.plot_percent_attacking(
            img_att_L, 
            viz_config["att_L"], 
            viz_config["att_centers_L"], 
            results["attacking"]["left_pct"], 
            f"Left Side Attacking ({results['own_left_count']} corners)"
        )
        st.pyplot(fig)

    with col_att_R:
        st.subheader("Att. Corners Right")
        fig = oa.plot_percent_attacking(
            img_att_R, 
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
            img_def_L, 
            viz_config["def_L"], 
            pcts, tot, ids, 
            f"Opp. Corners FROM LEFT"
        )
        st.pyplot(fig)

    with col_def_R:
        st.subheader("Def. Corners Right")
        tot, ids, pcts = results["defensive"]["right"]
        fig = oa.plot_shots_defensive(
            img_def_R, 
            viz_config["def_R"], 
            pcts, tot, ids, 
            f"Opp. Corners FROM RIGHT"
        )
        st.pyplot(fig)
