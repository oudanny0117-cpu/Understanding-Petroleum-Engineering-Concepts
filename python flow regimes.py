import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ===================================================================
# USER SETTINGS
# ===================================================================
fracture_positions = np.array([0, 300, 350, 500, 800, 950, 1010, 1050, 1400, 1675, 1950])
half_length = 250.0
W = 2 * half_length
num_points_per_interval = 600
num_terms = 120

# Derived
intervals = np.diff(fracture_positions)
total_length = fracture_positions[-1]
num_intervals = len(intervals)  # 10
n_vals = np.arange(1, 2 * num_terms + 1, 2)
L_avg = np.mean(intervals)

def pressure_in_interval(x_D_local, t_D_local):
    n = n_vals[:, np.newaxis]
    sin_term = np.sin(n * np.pi * x_D_local)
    exp_term = np.exp(-(n * np.pi)**2 * t_D_local)
    sum_term = np.sum((4 / (n * np.pi)) * sin_term * exp_term, axis=0)
    return 1 - sum_term

def generate_plot(t_D_global):
    # INCREASED FIGURE SIZE: much wider and taller for better visibility
    fig, ax = plt.subplots(figsize=(20, 10))  # Was (16, 6) → now bigger and taller

    # ... [rest of your plotting code unchanged until the end] ...

    # (Keep all your existing code inside the function: x_global, pcolormesh, colorbar, lines, legend, etc.)

    ax.set_ylabel('Fracture Half Length, xf (arbitrary units)', fontsize=14)
    ax.set_xlabel('Lateral Length, Lw (arbitrary units)', fontsize=14)
    
    # Fixed flow regime logic (corrected below)
    interference_count = 10
    for mid_pos, p_mid in zip(midpoint_positions, midpoint_pressures):
        if p_mid < 0.1:
            ax.axvline(mid_pos, color='cyan', linestyle='--', linewidth=3, alpha=0.9)
            interference_count -= 1
        else:
            ax.axvline(mid_pos, color='k', linestyle=':', linewidth=1.5, alpha=0.75)

    if interference_count == 10:
        flow_regime = "Infinite Acting"
    elif interference_count == 0:
        flow_regime = "Boundary Dominated Flow"
    else:
        flow_regime = "Transitional Flow"

    ax.set_title(
        'Multi-Stage Fractured Horizontal Well — Top View\n'
        f'Time Producing = {t_D_global:.5f} (arbitrary units) │ '
        f'Reservoir Boundaries Reached = {10 - interference_count}/{num_intervals} │ '
        f'Flow Regime = {flow_regime}',
        fontsize=18, pad=20  # Slightly larger title font
    )

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_yticks([-250, -125, 0, 125, 250])
    ax.set_yticklabels(['250', '125', '0 (Well)', '125', '250'])
    ax.grid(True, axis='y', alpha=0.3, color='white', linestyle='--')
    ax.set_xlim(0, total_length)
    ax.set_ylim(-half_length, half_length)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Hydraulic Fracture'),
        Line2D([0], [0], color='whitesmoke', lw=6, label='Horizontal Wellbore'),
        Line2D([0], [0], color='cyan', linestyle='--', lw=3, label='Significant Interference (P << Pi)'),
        Line2D([0], [0], color='k', linestyle=':', lw=1.5, alpha=0.75, label='No Significant Interference'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.85)

    plt.tight_layout()
    return fig

# Streamlit interface — IMPROVED LAYOUT
st.set_page_config(page_title="Fracture Simulation", layout="wide")  # Makes app full-width

st.title("Multi-Stage Fractured Horizontal Well — Pressure Depletion Visualization")

# Use columns to center the slider nicely
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    t_D_global = st.slider(
        "Time Producing (dimensionless)",
        min_value=0.0001,
        max_value=0.25,
        value=0.001,
        step=0.001,
        format="%.4f",
        help="Higher values = longer production time → more pressure depletion"
    )

# Display the large plot — use full width
fig = generate_plot(t_D_global)
st.pyplot(fig, use_container_width=True)  # This makes it stretch to fill the page width
