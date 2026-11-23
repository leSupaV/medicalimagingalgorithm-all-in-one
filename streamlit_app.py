import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. PHANTOM & SINOGRAM GENERATOR (Kak & Slaney, Table 3.1 & Fig. 3.5)
# ==============================================================================
SHEPP_LOGAN_TABLE_3_1 = [
    (0.0,   0.0,     0.92,   0.69,    90.0,  2.00),
    (0.0,  -0.0184,  0.874,  0.6624,  90.0, -0.98),
    (0.22,  0.0,     0.31,   0.11,    72.0, -0.02),
    (-0.22, 0.0,     0.41,   0.16,   108.0, -0.02),
    (0.0,   0.35,    0.25,   0.21,    90.0,  0.01),
    (0.0,   0.1,     0.046,  0.046,    0.0,  0.01),
    (0.0,  -0.1,     0.046,  0.046,    0.0,  0.01),
    (-0.08,-0.605,   0.046,  0.023,    0.0,  0.01),
    (0.0,  -0.605,   0.023,  0.023,    0.0,  0.01),
    (0.06, -0.605,   0.046,  0.023,   90.0,  0.01),
]

def generate_phantom_image(grid_size=128, ellipses=SHEPP_LOGAN_TABLE_3_1):
    coords = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords)
    img = np.zeros((grid_size, grid_size), dtype=np.float64)
    for x0, y0, A, B, alpha_deg, rho in ellipses:
        alpha = np.deg2rad(alpha_deg)
        dx, dy = X - x0, Y - y0
        x_rot = dx * np.cos(alpha) + dy * np.sin(alpha)
        y_rot = -dx * np.sin(alpha) + dy * np.cos(alpha)
        img[(x_rot**2 / A**2) + (y_rot**2 / B**2) <= 1.0] += rho
    return img

def analytical_sinogram(angles_deg, num_rays=127, t_max=1.0, ellipses=SHEPP_LOGAN_TABLE_3_1):
    t_vals = np.linspace(-t_max, t_max, num_rays)
    angles_rad = np.deg2rad(angles_deg)
    sinogram = np.zeros((len(angles_rad), num_rays), dtype=np.float64)
    for i, theta in enumerate(angles_rad):
        proj_angle = np.zeros(num_rays, dtype=np.float64)
        for x0, y0, A, B, alpha_deg, rho in ellipses:
            alpha = np.deg2rad(alpha_deg)
            theta_eff = theta - alpha
            t_shift = t_vals - (x0 * np.cos(theta) + y0 * np.sin(theta))
            a_sq = (A * np.cos(theta_eff))**2 + (B * np.sin(theta_eff))**2
            inside = t_shift**2 < a_sq
            proj_angle[inside] += (2.0 * rho * A * B / a_sq) * np.sqrt(a_sq - t_shift[inside]**2)
        sinogram[i, :] = proj_angle
    return sinogram, t_vals

# ==============================================================================
# 2. RECONSTRUCTION ALGORITHMS (Ch. 3 FBP & Ch. 7 SART)
# ==============================================================================
def reconstruct_fbp(sinogram, angles_deg, t_vals, output_size=128, zero_pad=True, window="hamming"):
    num_angles, num_rays = sinogram.shape
    dt = t_vals[1] - t_vals[0]
    padded_len = max(64, int(2 ** np.ceil(np.log2(2 * num_rays)))) if zero_pad else num_rays
    
    padded_sino = np.zeros((num_angles, padded_len), dtype=np.float64)
    padded_sino[:, :num_rays] = sinogram
    
    freqs = np.fft.fftfreq(padded_len, d=dt)
    ramp_filter = np.abs(freqs)
    norm_freqs = np.fft.fftfreq(padded_len)
    
    if window == "hamming":
        win = 0.54 + 0.46 * np.cos(2.0 * np.pi * norm_freqs)
    elif window == "cosine":
        win = np.cos(np.pi * norm_freqs)
    else:
        win = np.ones(padded_len)
        
    sino_fft = np.fft.fft(padded_sino, axis=1)
    filtered_sino = np.real(np.fft.ifft(sino_fft * (ramp_filter * win), axis=1))[:, :num_rays]
    
    recon = np.zeros((output_size, output_size), dtype=np.float64)
    coords = np.linspace(-1.0, 1.0, output_size)
    X, Y = np.meshgrid(coords, coords)
    d_theta = np.pi / num_angles
    
    for i, theta in enumerate(np.deg2rad(angles_deg)):
        t_grid = X * np.cos(theta) + Y * np.sin(theta)
        recon += np.interp(t_grid, t_vals, filtered_sino[i, :], left=0.0, right=0.0)
        
    return recon * d_theta

def reconstruct_sart(sinogram, angles_deg, t_vals, output_size=128, iterations=1, 
                     relaxation=1.0, use_hamming=True, non_negative=True):
    num_angles, num_rays = sinogram.shape
    coords = np.linspace(-1.0, 1.0, output_size)
    delta = coords[1] - coords[0]
    ds = 0.5 * delta
    
    X, Y = np.meshgrid(coords, coords)
    recon_circle = (X**2 + Y**2) <= 1.0
    g = np.zeros((output_size, output_size), dtype=np.float64)
    
    step = max(1, int(round(num_angles * (73.8 / 180.0))))
    while np.gcd(step, num_angles) != 1 and step < num_angles:
        step += 1
    angle_order = [(k * step) % num_angles for k in range(num_angles)]
    
    num_samples = int(np.ceil(2.0 / ds)) + 1
    s_vals = np.linspace(-1.0, 1.0, num_samples)
    T_grid, S_grid = np.meshgrid(t_vals, s_vals, indexing='ij')
    inside_circle = (T_grid**2 + S_grid**2) <= 1.0
    L_i = np.sum(inside_circle, axis=1) * ds
    
    ray_weights = np.zeros_like(T_grid)
    for r in range(num_rays):
        valid_idx = np.where(inside_circle[r])[0]
        M_i = len(valid_idx)
        if M_i > 1:
            ray_weights[r, valid_idx] = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(M_i) / (M_i - 1))) if use_hamming else 1.0
        elif M_i == 1:
            ray_weights[r, valid_idx] = 1.0

    angles_rad = np.deg2rad(angles_deg)
    for _ in range(iterations):
        for idx in angle_order:
            theta = angles_rad[idx]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            col_f = ((T_grid * cos_t - S_grid * sin_t) + 1.0) / delta
            row_f = ((T_grid * sin_t + S_grid * cos_t) + 1.0) / delta
            
            c0, r0 = np.floor(col_f).astype(np.int32), np.floor(row_f).astype(np.int32)
            c1, r1 = c0 + 1, r0 + 1
            valid = inside_circle & (c0 >= 0) & (c1 < output_size) & (r0 >= 0) & (r1 < output_size)
            
            dc, dr = col_f - c0, row_f - r0
            w00 = np.where(valid, (1.0 - dc) * (1.0 - dr), 0.0)
            w10 = np.where(valid, dc * (1.0 - dr), 0.0)
            w01 = np.where(valid, (1.0 - dc) * dr, 0.0)
            w11 = np.where(valid, dc * dr, 0.0)
            
            c0_s, c1_s = np.clip(c0, 0, output_size - 1), np.clip(c1, 0, output_size - 1)
            r0_s, r1_s = np.clip(r0, 0, output_size - 1), np.clip(r1, 0, output_size - 1)
            
            f_interp = w00 * g[r0_s, c0_s] + w10 * g[r0_s, c1_s] + w01 * g[r1_s, c0_s] + w11 * g[r1_s, c1_s]
            q_i = np.sum(f_interp, axis=1) * ds
            
            residual = np.zeros(num_rays, dtype=np.float64)
            active = L_i > 1e-6
            residual[active] = (sinogram[idx, active] - q_i[active]) / L_i[active]
            
            weighted_corr = residual[:, None] * ray_weights * ds
            weight_term = ray_weights * ds
            
            num_acc, den_acc = np.zeros_like(g), np.zeros_like(g)
            for w_c, r_i, c_i in [(w00, r0_s, c0_s), (w10, r0_s, c1_s), (w01, r1_s, c0_s), (w11, r1_s, c1_s)]:
                np.add.at(num_acc, (r_i, c_i), weighted_corr * w_c)
                np.add.at(den_acc, (r_i, c_i), weight_term * w_c)
                
            upd = (den_acc > 1e-8) & recon_circle
            g[upd] += relaxation * (num_acc[upd] / den_acc[upd])
            if non_negative:
                g = np.maximum(g, 0.0)
    return g

# ==============================================================================
# 3. STREAMLIT USER INTERFACE
# ==============================================================================
st.set_page_config(page_title="CT Reconstruction Workstation", layout="wide")
st.title("Computed Tomography (CT) Reconstruction Workstation")

with st.sidebar:
    st.header("1. Scanner Geometry")
    grid_size = st.selectbox("Reconstruction Grid (N × N)", [64, 128, 256], index=1)
    num_angles = st.slider("Projection Angles (Views)", 15, 180, 100, step=5)
    num_rays = st.slider("Rays per Projection", 31, 255, 127, step=2)

    st.divider()
    st.header("2. Artifact & Noise Injection")
    noise_level = st.slider("Detector Noise (σ)", 0.0, 0.05, 0.0, step=0.005)
    ring_artifact = st.checkbox("Simulate Bad Detector (Ring Artifact)", value=False)
    bad_det = num_rays // 2 + 15 if num_rays > 40 else num_rays // 2

    st.divider()
    st.header("3. Reconstruction Method")
    algo = st.radio("Algorithm", ["Filtered Backprojection (FBP)", "SART (Iterative)", "Side-by-Side: FBP vs. SART"])

    if "FBP" in algo:
        zero_pad = st.checkbox("Zero-Pad Projections (Prevent Dishing)", value=True)
        fbp_window = st.selectbox("Filter Window", ["hamming", "cosine", "ram-lak"])

    if "SART" in algo:
        sart_iters = st.slider("SART Iterations", 1, 5, 1)
        sart_hamming = st.checkbox("Longitudinal Ray Hamming Window", value=True)
        sart_nonneg = st.checkbox("Enforce Non-Negativity (g ≥ 0)", value=True)

    st.divider()
    st.header("4. Contrast Window")
    preset = st.selectbox("Display Range", ["Soft Tissue [0.95 – 1.05]", "Full Range [0.0 – 2.0]"])
    vmin, vmax = (0.95, 1.05) if "Soft" in preset else (0.0, 2.0)

@st.cache_data
def get_clean_data(n, n_ang, n_ray):
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = generate_phantom_image(n, SHEPP_LOGAN_TABLE_3_1)
    sino, t = analytical_sinogram(ang, num_rays=n_ray, t_max=1.0)
    return gt, sino, ang, t

ground_truth, clean_sino, angles, t_vals = get_clean_data(grid_size, num_angles, num_rays)
sinogram = clean_sino.copy()

if noise_level > 0:
    sinogram += np.random.default_rng(42).normal(0.0, noise_level, size=sinogram.shape)
if ring_artifact:
    sinogram[:, bad_det] *= 0.985

results = {}
if "FBP" in algo:
    t0 = time.perf_counter()
    img = reconstruct_fbp(sinogram, angles, t_vals, output_size=grid_size, zero_pad=zero_pad, window=fbp_window)
    results["FBP"] = {"img": img, "ms": (time.perf_counter() - t0) * 1000, "rmse": np.sqrt(np.mean((ground_truth - img)**2))}

if "SART" in algo:
    with st.spinner("Running SART iterations..."):
        t0 = time.perf_counter()
        img = reconstruct_sart(sinogram, angles, t_vals, output_size=grid_size, iterations=sart_iters, use_hamming=sart_hamming, non_negative=sart_nonneg)
        results["SART"] = {"img": img, "ms": (time.perf_counter() - t0) * 1000, "rmse": np.sqrt(np.mean((ground_truth - img)**2))}

cols = st.columns(len(results) * 2 + 1)
cols[0].metric("Views × Rays", f"{num_angles} × {num_rays}")
idx = 1
for name, d in results.items():
    cols[idx].metric(f"{name} Time", f"{d['ms']:.1f} ms")
    cols[idx + 1].metric(f"{name} RMSE", f"{d['rmse']:.4f}")
    idx += 2

st.subheader("Tomographic Image Views")
img_cols = st.columns(2 + len(results))

for col, title, data, kw in [
    (img_cols[0], "1. Sinogram", sinogram, {"aspect": "auto", "extent": [-1, 1, 180, 0]}),
    (img_cols[1], "2. Ground Truth", ground_truth, {"origin": "lower", "extent": [-1, 1, -1, 1], "vmin": vmin, "vmax": vmax}),
]:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(data, cmap="gray", **kw)
    if "Truth" in title:
        ax.axhline(-0.605, color="cyan", linestyle="--", linewidth=1)
    ax.set_title(title)
    col.pyplot(fig)
    plt.close(fig)

for i, (name, d) in enumerate(results.items()):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(d["img"], cmap="gray", origin="lower", extent=[-1, 1, -1, 1], vmin=vmin, vmax=vmax)
    ax.axhline(-0.605, color="cyan", linestyle="--", linewidth=1)
    ax.set_title(f"{i+3}. {name} Reconstruction")
    img_cols[i + 2].pyplot(fig)
    plt.close(fig)

st.subheader("Line Profile Comparison (y = -0.605 Across Small Tumors)")
coords = np.linspace(-1.0, 1.0, grid_size)
row = np.argmin(np.abs(coords - (-0.605)))
fig_p, ax_p = plt.subplots(figsize=(10, 3.2))
ax_p.plot(coords, ground_truth[row, :], "k--", label="Ground Truth")
for name, d in results.items():
    ax_p.plot(coords, d["img"][row, :], label=f"{name}")
ax_p.set_ylim(0.85, 1.10)
ax_p.legend()
ax_p.grid(True, alpha=0.3)
st.pyplot(fig_p)
plt.close(fig_p)
