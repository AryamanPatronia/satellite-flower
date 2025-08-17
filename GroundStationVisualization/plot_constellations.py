import os
import numpy as np
import matplotlib.pyplot as plt

# Visible Settings
SNAPSHOT_HOUR = 60                  # round 15 (t=60)...
SHOW_SNAPSHOT_LINKS = True          # Links to the visible satellites...
SHOW_INVISIBLE = True               # Satellites to the far side...
GROUND_STATION_LAT_DEG = 55.0       # GS Placement...
GROUND_STATION_LON_DEG = 0.0

# Visual scales...
EARTH_R   = 5  # Much larger Earth for visual focus...
ORBIT_R   = 6   # Orbits scale up accordingly...
AX_LIM    = 4.0   # Tighter axis limit for less empty space...
EARTH_ALPHA = 0.30

MANUAL_VISIBLE = {
    "walker_star":      {5, 1, 2, 3},
    "retrograde_polar": {5},                 # ["satellite_5","satellite_5"] -> {5}
    "polar_sso":        {1, 3, 5},
    "inclined_sparse":  {2, 3},
    "equatorial":       {5, 1},
}

SAT_COLORS = {
    1: "#0072B2",  # blue
    2: "#D55E00",  # vermillion
    3: "#009E73",  # green
    4: "#F0E442",  # yellow
    5: "#CC79A7",  # purple
}
SAT_OFFSET = {1:-10.0, 2:-5.0, 3:0.0, 4:+5.0, 5:+10.0}  # degrees

# Geometry & Rotations...
def rot_z(a): 
    c, s = np.cos(a), np.sin(a); 
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rot_x(a): 
    c, s = np.cos(a), np.sin(a); 
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def orbit_circle(incl_deg, raan_deg, radius=ORBIT_R, npts=700):
    t = np.linspace(0, 2*np.pi, npts)
    base = np.vstack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)])
    R = rot_z(np.deg2rad(raan_deg)) @ rot_x(np.deg2rad(incl_deg))
    pts = R @ base
    return pts[0], pts[1], pts[2]

def point_on_orbit(incl_deg, raan_deg, anomaly_deg, radius=ORBIT_R):
    R = rot_z(np.deg2rad(raan_deg)) @ rot_x(np.deg2rad(incl_deg))
    p = np.array([radius*np.cos(np.deg2rad(anomaly_deg)),
                  radius*np.sin(np.deg2rad(anomaly_deg)), 0.0])
    return R @ p

def latlon_to_xyz(lat_deg, lon_deg, R):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.array([R*np.cos(lat)*np.cos(lon), R*np.cos(lat)*np.sin(lon), R*np.sin(lat)])

# Contact / Anomaly Logic...
def best_anomaly_for_ground_contact(incl_deg, raan_deg, g_hat):
    """
    Orbit basis vectors u,v define p(nu)=cos(nu)*u + sin(nu)*v.
    Max dot(p, g_hat) occurs at nu* = atan2(<v,g>, <u,g>).
    Points opposite (nu*+180°) are far side.
    """
    R = rot_z(np.deg2rad(raan_deg)) @ rot_x(np.deg2rad(incl_deg))
    u = R @ np.array([1.0, 0.0, 0.0])
    v = R @ np.array([0.0, 1.0, 0.0])
    a = float(np.dot(u, g_hat)); b = float(np.dot(v, g_hat))
    nu_near = np.degrees(np.arctan2(b, a)) % 360.0
    nu_far  = (nu_near + 180.0) % 360.0
    return nu_near, nu_far

from collections import defaultdict
def distribute_anomalies_by_plane(sats_meta, visible_ids, g_hat, near_spread=14.0, far_spread=24.0):
    """
    For each plane (incl, raan), compute near/far 'centers' and spread satellites on that side
    so they don't clump. Returns dict {sid: anomaly_deg}.
    """
    groups = defaultdict(list)  # (incl, raan) -> [sid,...]
    for sid, incl, raan in sats_meta:
        groups[(float(incl), float(raan))].append(sid)

    sid_to_anom = {}
    for (incl, raan), sids in groups.items():
        near_sids = [s for s in sids if s in visible_ids]
        far_sids  = [s for s in sids if s not in visible_ids]
        nu_near, nu_far = best_anomaly_for_ground_contact(incl, raan, g_hat)

        def spaced(base, n, spread):
            if n <= 1: return [base]
            offsets = [(i - (n-1)/2.0) * spread for i in range(n)]
            return [(base + off) % 360.0 for off in offsets]

        near_angles = spaced(nu_near, len(near_sids), near_spread)
        far_angles  = spaced(nu_far,  len(far_sids),  far_spread)

        for sid, ang in zip(near_sids, near_angles):
            sid_to_anom[sid] = (ang + SAT_OFFSET.get(sid, 0.0)) % 360.0
        for sid, ang in zip(far_sids, far_angles):
            sid_to_anom[sid] = (ang + SAT_OFFSET.get(sid, 0.0)) % 360.0

    return sid_to_anom

# Drawing...
def draw_earth(ax, R=EARTH_R, alpha=EARTH_ALPHA):
    u, v = np.mgrid[0:2*np.pi:70j, 0:np.pi:35j]
    x = R*np.cos(u)*np.sin(v)
    y = R*np.sin(u)*np.sin(v)
    z = R*np.cos(v)
    # Grid...
    ax.plot_surface(x, y, z, linewidth=0.1, edgecolor='darkgrey', antialiased=True, alpha=alpha, zorder=1)
    # Draw small polar poles...
    pole_len = R * 0.18
    # North pole (top)...
    ax.plot([0, 0], [0, 0], [R, R+pole_len], color='black', linewidth=2.5, zorder=20)
    ax.text(0, 0, R+pole_len*1.08, 'N', fontsize=13, fontweight='bold', fontname='Times New Roman', ha='center', va='bottom', zorder=21)
    # South pole (bottom)...
    ax.plot([0, 0], [0, 0], [-R-pole_len, -R], color='black', linewidth=2.5, zorder=20)
    ax.text(0, 0, -R-pole_len*1.08, 'S', fontsize=13, fontweight='bold', fontname='Times New Roman', ha='center', va='top', zorder=21)
    # Draw the equator...
    t = np.linspace(0, 2*np.pi, 200)
    xe = R * np.cos(t)
    ye = R * np.sin(t)
    ze = np.zeros_like(t)
    ax.plot(xe, ye, ze, color='black', linewidth=1.5, linestyle='--', zorder=19)
    # Label the equator...
    eq_offset = 0.7
    ax.text(R + eq_offset, eq_offset, eq_offset, 'Equator', fontsize=13, fontweight='bold', fontname='Times New Roman', ha='center', va='center', zorder=21, rotation=0)

def style_axes(ax, title, view="frontal", lim=AX_LIM):
    ax.set_title(title, fontsize=16, fontweight="bold", fontname="Times New Roman")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1,1,1])
    # Grid Settings...
    ax.grid(False)
    ax.set_axis_off()  
    if view == "frontal":
        ax.view_init(elev=10, azim=-60)
    elif view == "top":
        ax.view_init(elev=90, azim=-90)

# Constellation Definitions...
def sats_walker_star():
    incl = 80.0
    raans = np.linspace(0, 360, 5, endpoint=False)  # 0,72,144,216,288
    sats_meta = [(i+1, incl, float(raans[i])) for i in range(5)]
    orbits = [(incl, float(r)) for r in raans]
    return sats_meta, orbits

def sats_polar_sso():
    incl = 98.0
    raans = np.linspace(0, 360, 3, endpoint=False)
    p0, p1, p2 = float(raans[0]), float(raans[1]), float(raans[2])
    sats_meta = [(1, incl, p0), (2, incl, p0), (3, incl, p1), (4, incl, p2), (5, incl, p2)]
    orbits = [(incl, p0), (incl, p1), (incl, p2)]
    return sats_meta, orbits

def sats_equatorial():
    incl, raan = 0.0, 0.0
    sats_meta = [(i, incl, raan) for i in range(1,6)]
    orbits = [(incl, raan)]
    return sats_meta, orbits

def sats_inclined_sparse():
    incl = 60.0
    p0, p1 = 0.0, 90.0
    sats_meta = [(1, incl, p0), (2, incl, p0), (3, incl, p1), (4, incl, p1), (5, incl, p1)]
    orbits = [(incl, p0), (incl, p1)]
    return sats_meta, orbits

def sats_retrograde_polar():
    incl, raan = 110.0, 0.0
    sats_meta = [(i, incl, raan) for i in range(1,6)]
    orbits = [(incl, raan)]
    return sats_meta, orbits

# Plotting...
def plot_constellation(name, sats_meta, orbits, visible_ids, outdir="figs"):
    os.makedirs(outdir, exist_ok=True)

    # Ground station vector...
    g = latlon_to_xyz(GROUND_STATION_LAT_DEG, GROUND_STATION_LON_DEG, EARTH_R)
    g_hat = g / (np.linalg.norm(g) + 1e-9)

    # Create figure and 3D axes...
    fig = plt.figure(figsize=(14,7))
    axs = [fig.add_subplot(1,2,i+1, projection='3d') for i in range(2)]
    # Remove all whitespace between axes for a seamless look...
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    views = ["frontal", "top"]
    for ax, view in zip(axs, views):
        # Earth + GS marker...
        draw_earth(ax)
        ax.scatter([g[0]],[g[1]],[g[2]], s=120, marker='^', edgecolor='black', linewidths=1.2, c=['#111'], zorder=10, label="GS (Parameter Server)")
        ax.text(g[0]*1.06, g[1]*1.06, g[2]*1.06, "GS (Parameter Server)", fontsize=13, fontweight="bold", fontname="Times New Roman")

        # Satellite
        for incl, raan in orbits:
            X, Y, Z = orbit_circle(incl, raan)
            ax.plot(X, Y, Z, linewidth=2.0, color="#555555", alpha=0.7, zorder=2)

        # Compute declumped anomalies per plane (near/far placement)...
        sid_to_anom = distribute_anomalies_by_plane(sats_meta, visible_ids, g_hat,
                                                    near_spread=14.0, far_spread=24.0)

        # Satellites...
        handles = []
        labels = []
        for sid, incl, raan in sats_meta:
            want_vis = (sid in visible_ids)
            nu = sid_to_anom[sid]
            p  = point_on_orbit(incl, raan, nu)

            if want_vis or SHOW_INVISIBLE:
                size = 130 if want_vis else 90
                lw   = 1.8 if want_vis else 0.7
                clr  = SAT_COLORS[sid] if want_vis else "#BBBBBB"
                alp  = 1.0 if want_vis else 0.40
                sc = ax.scatter([p[0]],[p[1]],[p[2]], s=size, edgecolor="black", linewidths=lw,
                               c=[clr], alpha=alp, zorder=6 if want_vis else 2, label=f"s{sid}" if want_vis else None)
                if want_vis:
                    handles.append(sc)
                    labels.append(f"s{sid}")
                ax.text(p[0]*1.04, p[1]*1.04, p[2]*1.04, f"s{sid}", fontsize=13, fontweight="bold", fontname="Times New Roman", alpha=alp)

                if SHOW_SNAPSHOT_LINKS and want_vis:
                    ax.plot([g[0], p[0]], [g[1], p[1]], [g[2], p[2]],
                            linestyle='--', linewidth=2.0, color="#222222", zorder=5)

        style_axes(ax, f"{name.replace('_',' ').title()} — {view.title()} view", view=view, lim=AX_LIM)
        if view == "frontal":
            ax.text2D(0.02, 0.02, "Representation of T=60h (round 15).", transform=ax.transAxes, fontsize=15, fontname="Times New Roman", fontweight="bold")

        # Legend for visible satellites...
        if handles and view == "frontal":
            ax.legend(handles, labels, loc='upper left', fontsize=12, frameon=False, prop={"family": "Times New Roman"}, markerscale=1.7)

    out_path = os.path.join(outdir, f"{name}_sidebyside.png")
    fig.savefig(out_path, dpi=420, bbox_inches='tight')
    plt.close(fig)

# Run...
if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)

    constellations = [
        ("walker_star",      sats_walker_star),
        ("polar_sso",        sats_polar_sso),
        ("equatorial",       sats_equatorial),
        ("inclined_sparse",  sats_inclined_sparse),
        ("retrograde_polar", sats_retrograde_polar),
    ]

    for name, sat_fn in constellations:
        sats_meta, orbits = sat_fn()                 # [(sid, incl, raan), ...], [(incl, raan), ...]
        visible = MANUAL_VISIBLE[name]               # set of int IDs visible at t=60h
        plot_constellation(name, sats_meta, orbits, visible_ids=visible, outdir="figs")

    print("Done. Check ./figs for saved PNGs.")
