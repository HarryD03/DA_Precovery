
import numpy as np
from daceypy import ADS
import pandas as pd
import matplotlib as plt
import seaborn as sns

def state_to_orbital_frame(state_vector, mu, eps=1e-12):
    """
    Rotate an inertial 6D state [x,y,z,vx,vy,vz] into the orbital (perifocal-like) frame:
      - x̂ points to periapsis (eccentricity direction, if defined; else radial r̂)
      - ŷ completes the right-handed in-plane basis
      - ẑ is the orbit normal ĥ = (r×v)/|r×v|

    Returns:
      rotated_state: 6-vector in orbital frame
      R: 3×3 rotation matrix (inertial -> orbital)
    """
    r = np.asarray(state_vector[:3], dtype=float)
    v = np.asarray(state_vector[3:6], dtype=float)

    r_norm = np.linalg.norm(r)
    if r_norm < eps:
        raise ValueError("Position norm is ~0; cannot define orbital plane.")

    # Specific angular momentum (orbit normal)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if h_norm < eps:
        raise ValueError("h ≈ 0 (radial or undefined trajectory); cannot define orbital plane.")

    k_hat = h / h_norm  # ẑ along orbit normal

    # Eccentricity vector (periapsis direction)
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e_norm = np.linalg.norm(e_vec)

    if e_norm > 1e-10:
        i_hat = e_vec / e_norm          # x̂ to periapsis
    else:
        i_hat = r / r_norm              # circular: use instantaneous radial direction

    j_hat = np.cross(k_hat, i_hat)      # ŷ completes right-handed triad
    j_hat /= np.linalg.norm(j_hat)

    # Re-orthogonalize i_hat just in case (numerical hygiene)
    i_hat = np.cross(j_hat, k_hat)

    # Rotation from inertial to orbital: rows are new-basis unit vectors in the old basis
    R = np.vstack((i_hat, j_hat, k_hat))

    r_orb = R @ r
    v_orb = R @ v
    return np.concatenate([r_orb, v_orb]), R

def plot_ArcLength_vs_Propagation_vs_Nimages(Nimages, propagation_times, arc_lengths, method, isoArea = False, outfile=None,):
    """
        Plot a Heat Map: Grid is (Arc length, Propagation time) colour = Nimages.
        Toggle constant (RAxDEC) area contours

        params: NImages: Number of images for given Propagation time and Arc Length Structure (len(propagation_times) x len(arc_lengths))
        params: propagation_times: List of propagation times Structure: 1D vector
        params: arc_lengths: List of arc lengths Structure: 1D vector
        params: isoArea: Toggle for constant (RAxDEC) area contours. Structure: (len(propagation_times) x len(arc_lengths))
        params: method: method to filter
                        ("DAIOD+ADS", "IOD+PW", "AR")
        params: outfile: Save figure option
        params: title: Title for plot
    """

    #check size is the same
    assert propagation_times == len(Nimages[0,:])
    assert arc_lengths == len(Nimages[:,0])

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(Nimages, origin='lower', aspect='auto',
                   extent=[arc_lengths.min(), arc_lengths.max(), propagation_times.min(), propagation_times.max()],
                   cmap="viridis"
    )
    
    if isoArea:
        assert isoArea.shape == Nimages.shape

        X, Y = np.meshgrid(arc_lengths, propagation_times)

        contours = ax.contour(X, Y, isoArea, colors='white', levels=5, linewidths=1, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    plt.colorbar(im, ax=ax, label='Number of Images')
    ax.set_xlabel('Arc length [days]', fontsize=12)
    ax.set_ylabel('Propagation time [days]', fontsize=12)
    ax.set_title(f" ({method})", fontsize=13)

    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()

def plot_ArcLength_vs_Propagation_vs_WallClockTime(WallClockTime, propagation_times, arc_lengths, method, outfile=None, Nimages=None, images_per_sec=False, use_log_scale=False):
    """
        Plot Heat Map: Grid is (Arc length, Propagation time) colour = Wall Clock Time.
        Toggle: NImages per second and constant (RAxDEC) area contours

        params: WallClockTime: Wall clock time for given Propagation time and Arc Length Structure (len(propagation_times) x len(arc_lengths))
        params: propagation_times: List of propagation times Structure: 1D vector
        params: arc_lengths: List of arc lengths Structure: 1D vector
        params: method: method to filter ("DAIOD+ADS", "IOD+PW", "AR")
        params: isoArea: Toggle for constant (RAxDEC) area contours
        params: area_data: 2D array of area values with same shape as WallClockTime for contour lines
        params: outfile: Save figure option
        params: use_log_scale: Apply log10 scaling to wall clock time values
        params: images_per_sec: If True and Nimages provided, plot images per second instead
        params: Nimages: Number of images array (required if images_per_sec=True)
    """

    # Check size is the same
    assert len(propagation_times) == WallClockTime.shape[1]
    assert len(arc_lengths) == WallClockTime.shape[0]

    # Handle images per second calculation
    Z = WallClockTime.copy()
    zlabel = "Wall clock time [s]"
    
    if images_per_sec:
        if Nimages is None:
            raise ValueError("Nimages array required when images_per_sec=True")
        assert Nimages.shape == WallClockTime.shape, "Nimages must have same shape as WallClockTime"
        
        # Avoid divide-by-zero
        safe_time = np.where(WallClockTime > 0, WallClockTime, np.nan)
        Z = Nimages / safe_time
        zlabel = "Images per second"

    # Apply log scaling if requested
    if use_log_scale:
        mask = (Z <= 0) | ~np.isfinite(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.log10(Z)
        Z[mask] = np.nan
        zlabel = f"log10({zlabel})"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, origin='lower', aspect='auto',
                   extent=[arc_lengths.min(), arc_lengths.max(), propagation_times.min(), propagation_times.max()],
                   cmap="viridis"
    )
    
    # Add iso-area contour lines if requested
    if images_per_sec:
        assert images_per_sec.shape == WallClockTime.shape, "Area data must have same shape as WallClockTime"
        
        X, Y = np.meshgrid(arc_lengths, propagation_times)

        contours = ax.contour(X, Y, images_per_sec, colors='white', levels=5, linewidths=1, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    plt.colorbar(im, ax=ax, label=zlabel)
    ax.set_xlabel('Arc length [days]', fontsize=12)
    ax.set_ylabel('Propagation time [days]', fontsize=12)
    
    title_type = "Images per Second" if images_per_sec else "Wall Clock Time"
    ax.set_title(f"{title_type} ({method})", fontsize=13)

    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()

def convert_wide_to_long(wide_data):
    """
    Convert wide format data to long format for plotting
        
    Expected wide format:
    Type | Orbit_Determination | Propagation | Evaluation
    """
    long_data = wide_data.melt(
        id_vars=['Type'], 
        var_name='Segment', 
        value_name='WallclockTime'
    )
    
    return long_data


def plot_Type_vs_WallclockTime(data):
    """
    Plots Segmented Bar charts for DAIOD+ADS, IOD+PW, AR+PW where each stage is timed.
    y axis = Wall clock time
    x axis = Technique method (DAIOD+ADS, IOD+PW, AR+PW)
    colours = (Orbit determination, Propagation, evaluation)

    :params data: DataFrame containing (Type x segment x wallclock time)
    :return: 2D bar chart
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Pivot data for stacked bar chart
    pivot_data = data.pivot_table(
        index='Type', 
        columns='Segment', 
        values='WallclockTime', 
        aggfunc='sum'
    )
    
    # Ensure proper column order
    segment_order = ['Orbit_Determination', 'Propagation', 'Evaluation']
    pivot_data = pivot_data.reindex(columns=segment_order)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bar chart
    pivot_data.plot(
        kind='bar', 
        stacked=True, 
        ax=ax,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],  # Blue, Orange, Green
        edgecolor='black',
        linewidth=0.5
    )
    
    # Customize the plot
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Wall Clock Time [s]', fontsize=12)
    ax.set_title('Computation Time Breakdown by Method', fontsize=14)
    ax.legend(title='Processing Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_pareto(wall_clock_time, Nimages, methods, outfile=None):
    """
    Pareto Plot:
        x-axis: Wall clock time
        y-axis: Number of pictures
        Colour: Method
        Points: (Arc length, Propagation time) combinations

    params: wall_clock_time: Numpy array Structure: (len(propagation_times) x len(Arc_lengths)) x (len(methods))
    params: Nimages: Numpy array Structure: (len(propagation_times) x len(Arc_lengths)) x (len(methods))
    params: outfile: Save figure option
    """

    df = pd.DataFrame({
        "wall_time": wall_clock_time.flatten(),
        "pictures": Nimages.flatten(),
        "method": np.repeat(methods, wall_clock_time.shape[0] * wall_clock_time.shape[1])
    })

    fig, ax = plt.subplots(figsize=(7,5))

    sns.scatterplot(
        data=df, x="wall_time", y="pictures",
        hue="method", style="method",
        s=80, edgecolor="k", ax=ax
    )

    ax.set_xlabel("Wall Clock Time [s]", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Pareto Frontier: Computation vs Coverage", fontsize=13)

    # Optional annotation of arc_length & prop_time for clarity
    for _, row in df.iterrows():
        ax.annotate(
            f"({row['arc_length']}, {row['prop_time']})",
            (row["wall_time"], row["pictures"]),
            textcoords="offset points", xytext=(5,3), fontsize=7
        )

    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()

def Nsplit_vs_SimulationTime(Ts, tgrid, final_lists):
    """
        Plot 2D Graph to show ADS cost. -> assess ADS cost through time
        y axis: Number of ADS splits
        x axis: Simulation Time
        colour: Different Arc lengths
        Utilises Wittigs Figure5 function
    """

    nsplit = np.zeros((Ts))
    nsplit[0]= 1

    for i in range(Ts-1):
        nsplit[i+1]=len(final_lists[i])     #Number of split in each domain is the number of subdomains

    _ , ax = plt.subplots()
    ax.plot(tgrid, nsplit)
    ax.grid('minor',  linestyle=':')
    ax.set_xlabel('propagation time (-)')
    ax.set_ylabel('number of domains (-)')

    return ax


def plot_RA_vs_DEC(perimeter_points, alphashape_points, DAIOD_nom, apophis,  time_idx):
    """
        2D Plot to show the 3-sigma perimeter in the RA-DEC plane. @ time_idx
        
        :param perimeter: Converted perimeter points in the ECI  RA-DEC plane
                        Expected Structure: (N_perimeter_points, 3, N_subdomains)
        :param perimeter_norm: 
        :param DAIOD_nom: Nominal solution of the perimeter cloud
        :param apophis: RA and DEC coordinates of apophis @ time_idx
        :param time_idx: Time idx of interest.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(perimeter_points[:, 0], perimeter_points[:, 1], c='r', s=1, label='3-sigma Perimeter')
    plt.scatter(DAIOD_nom[0], DAIOD_nom[1], c='b', s=10, label='DAIOD Nominal')
    plt.scatter(apophis[0], apophis[1], c='g', s=10, label='Apophis')

    plt.plot(*alphashape_points.exterior.xy, 'r--', label='Alpha Shape')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Right Ascension [Rad]")
    plt.ylabel("Declination [Rad]")
    plt.title(f"3-Sigma Perimeter in RA-DEC Plane Equatorial @ Time Index {time_idx}")
    plt.legend()
    plt.grid()
    plt.show()

def plot_precovery_area_vs_time(recovery_areas, time_steps, methods, outfile=None):
    """
    Plot the recovery area over time.

    params: recovery_areas: Numpy array of shape (len(time_steps), len(methods))
    params: time_steps: Numpy array of shape (len(time_steps),)
    params: outfile: Save figure option
    """

    df = pd.DataFrame({
        "recovery_area": recovery_areas.flatten(),
        "time": np.repeat(time_steps, recovery_areas.shape[1]),
        "method": np.tile(methods, recovery_areas.shape[0])
    })

    fig, ax = plt.subplots(figsize=(7,5))

    sns.lineplot(
        data=df, x="time", y="Precovery_area",
        hue="method", ax=ax
    )

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Precovery Area [m^2]", fontsize=12)
    ax.set_title("Precovery Area Over Time", fontsize=13)

    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()

# Add this code after the ADS propagation sections and before the alphashape creation

def plot_split_history(final_lists_ADS, final_lists_ADS_DAIOD, tgrid, dt_value, script_dir):
    """
    Plot the number of ADS domains vs propagation time to show splitting history
    """
    
    # Extract number of domains at each time step
    n_domains_ADS_OD = []
    n_domains_ADS_PROP = []
    
    for time_idx, domain_list in enumerate(final_lists_ADS):
        n_domains_ADS_OD.append(len(domain_list))
    
    for time_idx, domain_list in enumerate(final_lists_ADS_DAIOD):
        n_domains_ADS_PROP.append(len(domain_list))
    
    # Convert time grid to days for better readability
    tgrid_days = (tgrid - tgrid[0]) / (24 * 3600)  # Convert seconds to days
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot number of domains
    plt.subplot(2, 1, 1)
    plt.plot(tgrid_days, n_domains_ADS_OD, 'o-', label='DAIOD+ADS (ADS OD)', 
             linewidth=2, markersize=6, color='blue')
    plt.plot(tgrid_days, n_domains_ADS_PROP, 's-', label='DAIOD+ADS (ADS Prop)', 
             linewidth=2, markersize=6, color='red')
    
    plt.xlabel('Propagation Time (days)')
    plt.ylabel('Number of ADS Domains')
    plt.title(f'ADS Domain Splitting History (Arc Length = {dt_value:.3f} days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale since domains can grow exponentially
    
    # Plot splitting rate (domains added per time step)
    plt.subplot(2, 1, 2)
    
    # Calculate splitting rates
    split_rate_ADS_OD = np.diff(n_domains_ADS_OD)
    split_rate_ADS_PROP = np.diff(n_domains_ADS_PROP)
    
    plt.plot(tgrid_days[1:], split_rate_ADS_OD, 'o-', label='DAIOD+ADS (ADS OD)', 
             linewidth=2, markersize=4, color='blue')
    plt.plot(tgrid_days[1:], split_rate_ADS_PROP, 's-', label='DAIOD+ADS (ADS Prop)', 
             linewidth=2, markersize=4, color='red')
    
    plt.xlabel('Propagation Time (days)')
    plt.ylabel('Domains Added per Time Step')
    plt.title('ADS Domain Splitting Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    split_plots_dir = script_dir / "ADS_split_plots"
    split_plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_filename = f"ads_split_history_arc_{dt_value:.3f}d.png"
    plot_path = split_plots_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ADS split history plot saved to: {plot_path}")
    
    # Return statistics for summary
    return {
        'max_domains_ADS_OD': max(n_domains_ADS_OD),
        'max_domains_ADS_PROP': max(n_domains_ADS_PROP),
        'final_domains_ADS_OD': n_domains_ADS_OD[-1],
        'final_domains_ADS_PROP': n_domains_ADS_PROP[-1],
        'total_splits_ADS_OD': sum(split_rate_ADS_OD),
        'total_splits_ADS_PROP': sum(split_rate_ADS_PROP)
    }

def plot_precovery_area_vs_Nimages(Nimages, precovery_areas, methods, outfile=None):
    """
    Plot the precovery area against the number of images.

    params: Nimages: Numpy array of shape (len(propagation_times) x len(Arc_lengths)) x (len(methods))
    params: precovery_areas: Numpy array of shape (len(propagation_times) x len(Arc_lengths)) x (len(methods)) -> Extract from alphashapes
    params: outfile: Save figure option
    """

    df = pd.DataFrame({
        "Nimages": Nimages.flatten(),
        "Precovery_area": precovery_areas.flatten(),
        "method": np.repeat(methods, Nimages.shape[0] * Nimages.shape[1])
    })

    fig, ax = plt.subplots(figsize=(7,5))

    sns.lineplot(
        data=df, x="Nimages", y="Precovery_area",
        hue="method", ax=ax
    )

    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_ylabel("Precovery Area [m^2]", fontsize=12)
    ax.set_title("Precovery Area vs Number of Images", fontsize=13)

    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()