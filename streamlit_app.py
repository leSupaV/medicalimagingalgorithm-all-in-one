import io
import time
import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# 1. PHANTOM & SINOGRAM GENERATOR (Kak & Slaney, Table 3.1 & Eq. 7.25)
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

def forward_project_image(image, angles_deg, num_rays=127, t_max=1.0):
    """Discrete Radon transform via bilinear ray integration (Kak & Slaney Eq. 7.25)."""
    grid_size = image.shape[0]
    coords = np.linspace(-1.0, 1.0, grid_size)
    delta = coords[1] - coords[0]
    ds = 0.5 * delta

    t_vals = np.linspace(-t_max, t_max, num_rays)
    num_samples = int(np.ceil(2.0 * np.sqrt(2.0) / ds)) + 1
    s_vals = np.linspace(-np.sqrt(2.0), np.sqrt(2.0), num_samples)
    T_grid, S_grid = np.meshgrid(t_vals, s_vals, indexing='ij')

    sinogram = np.zeros((len(angles_deg), num_rays), dtype=np.float64)
    for i, theta in enumerate(np.deg2rad(angles_deg)):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        col_f = ((T_grid * cos_t - S_grid * sin_t) + 1.0) / delta
        row_f = ((T_grid * sin_t + S_grid * cos_t) + 1.0) / delta

        c0, r0 = np.floor(col_f).astype(np.int32), np.floor(row_f).astype(np.int32)
        c1, r1 = c0 + 1, r0 + 1
        valid = (c0 >= 0) & (c1 < grid_size) & (r0 >= 0) & (r1 < grid_size)

        dc, dr = col_f - c0, row_f - r0
        w00 = np.where(valid, (1.0 - dc) * (1.0 - dr), 0.0)
        w10 = np.where(valid, dc * (1.0 - dr), 0.0)
        w01 = np.where(valid, (1.0 - dc) * dr, 0.0)
        w11 = np.where(valid, dc * dr, 0.0)

        c0_s, c1_s = np.clip(c0, 0, grid_size - 1), np.clip(c1, 0, grid_size - 1)
        r0_s, r1_s = np.clip(r0, 0, grid_size - 1), np.clip(r1, 0, grid_size - 1)

        f_interp = (
            w00 * image[r0_s, c0_s]
            + w10 * image[r0_s, c1_s]
            + w01 * image[r1_s, c0_s]
            + w11 * image[r1_s, c1_s]
        )
        sinogram[i, :] = np.sum(f_interp, axis=1) * ds

    return sinogram, t_vals

def load_custom_image(file_bytes, file_name, grid_size):
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".npy"):
        arr = np.load(buf).astype(np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        img_pil = Image.fromarray(arr).resize((grid_size, grid_size), Image.Resampling.BILINEAR)
        img = np.array(img_pil, dtype=np.float64)
    else:
        img_pil = Image.open(buf).convert("L").resize((grid_size, grid_size), Image.Resampling.BILINEAR)
        img = np.array(img_pil, dtype=np.float64) / 255.0

    img = np.flipud(img)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    coords = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(coords, coords)
    img[(X**2 + Y**2) > 1.0] = 0.0
    return img

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

def render_with_scanline(img, vmin, vmax, flip_vertical=True, scanline_row=None):
    """Converts matrix to RGB uint8 and draws a dashed cyan scanline tying the image to the line profile."""
    if flip_vertical:
        img = np.flipud(img)
        if scanline_row is not None:
            scanline_row = (img.shape[0] - 1) - scanline_row

    clipped = np.clip((img - vmin) / max(1e-8, (vmax - vmin)), 0.0, 1.0)
    gray = (clipped * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    if scanline_row is not None and 0 <= scanline_row < rgb.shape[0]:
        # Dashed cyan indicator line across the sampled row
        for col in range(0, rgb.shape[1], 4):
            rgb[scanline_row, col:col+2] = [0, 230, 255]
    return rgb

def render_metric_card(title, value, subtitle, bg_color, border_color, text_color):
    """Renders a bold, color-coded KPI card."""
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-left: 5px solid {border_color};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
        ">
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {text_color}; opacity: 0.85;">
                {title}
            </div>
            <div style="font-size: 1.55rem; font-weight: 900; color: {text_color}; line-height: 1.2; margin: 2px 0;">
                {value}
            </div>
            <div style="font-size: 0.75rem; color: {text_color}; opacity: 0.75;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 3. STREAMLIT USER INTERFACE
# ==============================================================================
st.set_page_config(page_title="CT Reconstruction Workstation", layout="wide")

# Requirement 3: Centered Title Header
st.markdown(
    """
    <div style="text-align: center; padding: 0.25rem 0 1rem 0;">
        <h1 style="margin-bottom: 0.2rem; font-weight: 800;">Computed Tomography (CT) Reconstruction Workstation</h1>
        <p style="color: #9ca3af; font-size: 0.95rem; margin: 0;">
            Analytical Filtered Backprojection (Ch. 3) vs. Iterative SART (Ch. 7) &bull; Kak &amp; Slaney
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Requirement 1: Sidebar with Phantom Source Toggle & Custom Image Upload
with st.sidebar:
    st.header("1. Phantom / Input Source")
    use_custom = st.toggle("Use Custom Uploaded Image", value=False)
    uploaded_file = None
    if use_custom:
        uploaded_file = st.file_uploader("Upload Image (PNG, JPG, NPY)", type=["png", "jpg", "jpeg", "npy"])
    else:
        st.caption("Active: **Shepp-Logan Head Phantom (Table 3.1)**")

    st.divider()
    st.header("2. Scanner Geometry")
    grid_size = st.selectbox("Reconstruction Grid (N × N)", [64, 128, 256], index=1)
    num_angles = st.slider("Projection Angles (Views)", 15, 180, 100, step=5)
    num_rays = st.slider("Rays per Projection", 31, 255, 127, step=2)

    st.divider()
    st.header("3. Artifact & Noise Injection")
    noise_level = st.slider("Detector Noise (σ)", 0.0, 0.05, 0.0, step=0.005)
    ring_artifact = st.checkbox("Simulate Bad Detector (Ring Artifact)", value=False)
    bad_det = num_rays // 2 + 15 if num_rays > 40 else num_rays // 2

    st.divider()
    st.header("4. Reconstruction Method")
    algo = st.radio("Algorithm", ["Filtered Backprojection (FBP)", "SART (Iterative)", "Side-by-Side: FBP vs. SART"], index=2)

    if "FBP" in algo or "Side-by-Side" in algo:
        zero_pad = st.checkbox("Zero-Pad Projections (Prevent Dishing)", value=True)
        fbp_window = st.selectbox("Filter Window", ["hamming", "cosine", "ram-lak"])

    if "SART" in algo or "Side-by-Side" in algo:
        sart_iters = st.slider("SART Iterations", 1, 5, 1)
        sart_hamming = st.checkbox("Longitudinal Ray Hamming Window", value=True)
        sart_nonneg = st.checkbox("Enforce Non-Negativity (g ≥ 0)", value=True)

    st.divider()
    st.header("5. Contrast Window")
    if not use_custom or uploaded_file is None:
        preset = st.selectbox("Display Range", ["Soft Tissue [0.95 – 1.05]", "Full Range [0.0 – 2.0]"])
        vmin, vmax = (0.95, 1.05) if "Soft" in preset else (0.0, 2.0)
    else:
        vmin, vmax = st.slider("Custom Contrast Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

@st.cache_data
def get_shepp_logan_data(n, n_ang, n_ray):
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = generate_phantom_image(n, SHEPP_LOGAN_TABLE_3_1)
    sino, t = analytical_sinogram(ang, num_rays=n_ray, t_max=1.0)
    return gt, sino, ang, t

@st.cache_data
def get_custom_data(file_bytes, file_name, n, n_ang, n_ray):
    ang = np.linspace(0, 180, n_ang, endpoint=False)
    gt = load_custom_image(file_bytes, file_name, n)
    sino, t = forward_project_image(gt, ang, num_rays=n_ray, t_max=1.0)
    return gt, sino, ang, t

if use_custom and uploaded_file is not None:
    ground_truth, clean_sino, angles, t_vals = get_custom_data(
        uploaded_file.getvalue(), uploaded_file.name, grid_size, num_angles, num_rays
    )
    active_custom = True
else:
    if use_custom:
        st.info("Upload a PNG, JPG, or NPY file in the left sidebar (displaying Shepp-Logan phantom meanwhile).")
    ground_truth, clean_sino, angles, t_vals = get_shepp_logan_data(grid_size, num_angles, num_rays)
    active_custom = False

sinogram = clean_sino.copy()
if noise_level > 0:
    sinogram += np.random.default_rng(42).normal(0.0, noise_level, size=sinogram.shape)
if ring_artifact:
    sinogram[:, bad_det] *= 0.985

results = {}
if "FBP" in algo or "Side-by-Side" in algo:
    t0 = time.perf_counter()
    img = reconstruct_fbp(sinogram, angles, t_vals, output_size=grid_size, zero_pad=zero_pad, window=fbp_window)
    results["FBP"] = {"img": img, "ms": (time.perf_counter() - t0) * 1000, "rmse": np.sqrt(np.mean((ground_truth - img)**2))}

if "SART" in algo or "Side-by-Side" in algo:
    with st.spinner("Running SART iterations..."):
        t0 = time.perf_counter()
        img = reconstruct_sart(sinogram, angles, t_vals, output_size=grid_size, iterations=sart_iters, use_hamming=sart_hamming, non_negative=sart_nonneg)
        results["SART"] = {"img": img, "ms": (time.perf_counter() - t0) * 1000, "rmse": np.sqrt(np.mean((ground_truth - img)**2))}

# Requirement 2: Bold, Color-Coded KPI Metric Boxes
metric_cols = st.columns(len(results) * 2 + 1)
with metric_cols[0]:
    render_metric_card(
        "VIEWS × RAYS",
        f"{num_angles} × {num_rays}",
        f"Grid: {grid_size}×{grid_size} px",
        bg_color="rgba(59, 130, 246, 0.14)",
        border_color="#3b82f6",
        text_color="#93c5fd",
    )

col_idx = 1
palette = {
    "FBP": ("rgba(16, 185, 129, 0.14)", "#10b981", "#6ee7b7"),
    "SART": ("rgba(168, 85, 247, 0.14)", "#a855f7", "#d8b4fe"),
}
for name, d in results.items():
    bg, border, txt = palette[name]
    with metric_cols[col_idx]:
        render_metric_card(
            f"{name} COMPUTE TIME",
            f"{d['ms']:.1f} ms",
            "Execution latency",
            bg_color=bg,
            border_color=border,
            text_color=txt,
        )
    with metric_cols[col_idx + 1]:
        render_metric_card(
            f"{name} RMSE ERROR",
            f"{d['rmse']:.4f}",
            "Lower = closer to truth",
            bg_color="rgba(245, 158, 11, 0.14)",
            border_color="#f59e0b",
            text_color="#fcd34d",
        )
    col_idx += 2

# Requirement 5: Determine scanline coordinate & row index so images and profile graph are directly linked
coords = np.linspace(-1.0, 1.0, grid_size)
if not active_custom:
    profile_y = -0.605
    profile_label = "y = -0.605 (Cyan Scanline Across 3 Small Tumors)"
else:
    profile_y = 0.0
    profile_label = "y = 0.000 (Cyan Center Scanline)"
scan_row = int(np.argmin(np.abs(coords - profile_y)))

# Requirement 4: Neat Bordered Boxes for Each Tomographic Image View
st.subheader("Tomographic Image Views")
img_cols = st.columns(2 + len(results))

with img_cols[0]:
    with st.container(border=True):
        st.markdown("**1. Input Sinogram $P(t, \\theta)$**")
        st.image(
            render_with_scanline(sinogram, sinogram.min(), sinogram.max(), flip_vertical=False),
            use_container_width=True,
        )
        st.caption(f"{num_angles} angles × {num_rays} rays")

with img_cols[1]:
    with st.container(border=True):
        st.markdown("**2. Ground Truth Phantom**")
        st.image(
            render_with_scanline(ground_truth, vmin, vmax, flip_vertical=True, scanline_row=scan_row),
            use_container_width=True,
        )
        st.caption("Cyan dashed line = profile slice")

for i, (name, d) in enumerate(results.items()):
    with img_cols[i + 2]:
        with st.container(border=True):
            st.markdown(f"**{i+3}. {name} Reconstruction**")
            st.image(
                render_with_scanline(d["img"], vmin, vmax, flip_vertical=True, scanline_row=scan_row),
                use_container_width=True,
            )
            st.caption(f"RMSE: {d['rmse']:.4f} | {d['ms']:.1f} ms")

# Requirement 5: Line Profile Tied Directly to the Cyan Scanline in a Compact Box
with st.container(border=True):
    st.markdown(f"**Cross-Sectional Intensity Profile Along `{profile_label}`**")
    chart_data = {"Ground Truth": ground_truth[scan_row, :]}
    for name, d in results.items():
        chart_data[f"{name} Reconstruction"] = d["img"][scan_row, :]
    st.line_chart(chart_data, height=220)
