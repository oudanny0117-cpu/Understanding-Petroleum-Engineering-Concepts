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
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x_global = np.linspace(0, total_length, num_intervals * num_points_per_interval)
    p_D_global = np.zeros_like(x_global)
    
    midpoint_positions = []
    midpoint_pressures = []
    
    for i in range(num_intervals):
        L_i = intervals[i]
        left = fracture_positions[i]
        right = fracture_positions[i+1]
        mid_x = (left + right) / 2.0
        
        mask = (x_global >= left) & (x_global < right)
        x_local = x_global[mask]
        x_D_local = (x_local - left) / L_i
        
        t_D_local = t_D_global * (L_avg / L_i)**2
        p_interval = pressure_in_interval(x_D_local, t_D_local)
        p_D_global[mask] = p_interval
        
        p_mid = pressure_in_interval(np.array([0.5]), t_D_local)[0]
        midpoint_positions.append(mid_x)
        midpoint_pressures.append(p_mid)
    
    y_plot = np.linspace(-half_length, half_length, 160)
    X_plot, Y_plot = np.meshgrid(x_global, y_plot)
    P_2d = np.tile(p_D_global, (len(y_plot), 1))
    
    im = ax.pcolormesh(X_plot, Y_plot, P_2d, cmap='viridis', vmin=0, vmax=1.0, shading='auto')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Pressure', fontsize=14)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['P = Pi', 'P < Pi', 'P << Pi'])
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.invert_yaxis()

    # Fractures
    for pos in fracture_positions:
        ax.axvline(pos, ymin=0, ymax=1, color='red', linewidth=3.5)

    # Horizontal wellbore
    ax.axhline(0, color='whitesmoke', linewidth=6)

    # Interference indicators — exact original logic
    interference_count = 10  # starts at total number of intervals
    for mid_pos, p_mid in zip(midpoint_positions, midpoint_pressures):
        if p_mid < 0.1:
            ax.axvline(mid_pos, color='cyan', linestyle='--', linewidth=3, alpha=0.9)
            interference_count -= 1
        else:
            ax.axvline(mid_pos, color='k', linestyle=':', linewidth=1.5, alpha=0.75)

    # Flow regime based on remaining count
    if interference_count == num_intervals:
        flow_regime = "Infinite Acting"  # should never happen with this logic
    elif interference_count == 0:
        flow_regime = "Boundary Dominated Flow"  # wait — this is backwards from intent!
    else:
        flow_regime = "Transitional Flow"

    # Actually, looking at original intent: when interference_count == 0 → all reached → Boundary Dominated
    # When interference_count == 10 → none reached → Infinite Acting
    if interference_count == 0:
        flow_regime = "Boundary Dominated Flow"
    elif interference_count == 10:
        flow_regime = "Infinite Acting"
    else:
        flow_regime = "Transitional Flow"

    # Styling
    ax.set_ylabel('Fracture Half Length, xf (arbitrary units)', fontsize=14)
    ax.set_xlabel('Lateral Length, Lw (arbitrary units)', fontsize=14)
    ax.set_title(
        'Multi-Stage Fractured Horizontal Well — Top View\n'
        f'Time Producing = {t_D_global:.5f} (arbitrary units)  │  '
        f'Reservoir Boundaries Reached = {10 - interference_count}/{num_intervals}  │  '  # show reached, not remaining
        f'Flow Regime = {flow_regime}',
        fontsize=16, pad=20
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
        Line2D([0], [0], color='cyan', linestyle='--', lw=3, label='P = P_i'),
        Line2D([0], [0], color='k', linestyle=':', lw=1.5, alpha=0.85, label='P < P_i'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    return fig

# Streamlit interface
st.header("Multi-Stage Fractured Horizontal Well — Pressure Propagation Visualization")

t_D_global = st.slider(
    "Time Producing (arbitrary units)",
    min_value=0.0001,
    max_value=0.25,
    value=0.001,
    step=0.001,
    format="%.4f",
    help="Higher values = longer production time → more pressure depletion"
)

fig = generate_plot(t_D_global)
st.pyplot(fig)
