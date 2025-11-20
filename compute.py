"""
Computation functions for the Magnification Calibration Tool.

This module contains all the mathematical and image processing calculations
used by the main application.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import mrcfile
from pathlib import Path
from scipy.optimize import least_squares
import math
from skimage.transform import resize_local_mean
import plotly.graph_objects as go
import plotly.express as px
try:
    from finufft import nufft2d2
    FINUFFT_AVAILABLE = True
except ImportError:
    FINUFFT_AVAILABLE = False
from scipy.stats import median_abs_deviation
from shiny import ui
from shiny.express import expressify


@expressify
def google_analytics(id):
    if id is None or not len(id):
        return
    ui.head_content(
        ui.HTML(
            f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={id}"></script>
            <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{id}');
            </script>
            """
        )
    )


def fit_ellipse_fixed_center(points, center=(0, 0)):
    """
    Fit an ellipse to points with fixed center using least squares.
    
    Args:
        points: List of (x, y) tuples
        center: (cx, cy) center coordinates
        
    Returns:
        (a, b, theta): semi-major axis, semi-minor axis, rotation angle (radians)
    """
    cx, cy = center
    points = np.array(points)
    
    # Transform points to center
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    
    # Standard ellipse equation: (x/a)^2 + (y/b)^2 = 1
    # For rotated ellipse: ((x*cos(theta) + y*sin(theta))/a)^2 + ((-x*sin(theta) + y*cos(theta))/b)^2 = 1
    
    def ellipse_residuals(params):
        a, b, theta = params
        if a <= 0 or b <= 0:
            return np.inf * np.ones(len(x))
        
        # Rotate points
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        
        # Calculate residuals
        residuals = (x_rot / a)**2 + (y_rot / b)**2 - 1
        return residuals
    
    # Initial guess: use bounding box
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    a_init = max(x_range, y_range) / 2
    b_init = min(x_range, y_range) / 2
    theta_init = 0
    
    # Fit using least squares
    try:
        result = least_squares(ellipse_residuals, [a_init, b_init, theta_init], 
                             bounds=([0.1, 0.1, -np.pi/2], [np.inf, np.inf, np.pi/2]))
        a, b, theta = result.x
        return a, b, theta
    except:
        # Fallback to simple bounding box
        return a_init, b_init, theta_init


def normalize(magnitude, contrast=2.0):
    """
    Normalize FFT magnitude data for display.
    
    Args:
        magnitude: FFT magnitude array
        contrast: Number of standard deviations to include in range
        
    Returns:
        Normalized array (0-255 uint8)
    """
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    m1 = np.max(magnitude)
    # Adjust clip max based on contrast value
    clip_max = min(m1, mean + contrast * std)
    clip_min = 0
    magnitude_clipped = np.clip(magnitude, clip_min, clip_max)
    normalized = 255 * (magnitude_clipped - clip_min) / (clip_max - clip_min + 1e-8)
    return normalized


def normalize_image(img: np.ndarray, contrast=2.0, use_percentiles=False, low_percentile=1.0, high_percentile=99.0) -> np.ndarray:
    """
    Normalize image data using either mean/std or percentile-based clipping.
    
    Args:
        img: Input image array
        contrast: Number of standard deviations to include in range (if not using percentiles)
        use_percentiles: If True, use percentile-based clipping instead of mean/std
        low_percentile: Lower percentile for clipping (0-100)
        high_percentile: Upper percentile for clipping (0-100)
        
    Returns:
        Normalized image array (0-255 uint8)
    """
    # Convert to float32 for calculations
    img_float = img.astype(np.float32)
    
    if use_percentiles:
        # Use percentile-based clipping to remove outliers
        clip_min = np.percentile(img_float, low_percentile)
        clip_max = np.percentile(img_float, high_percentile)
    else:
        # Use mean ± contrast * std
        mean = np.mean(img_float)
        std = np.std(img_float)
        clip_min = max(0, mean - contrast * std)
        clip_max = min(img_float.max(), mean + contrast * std)
    
    # Clip and normalize to 0-255 range
    img_clipped = np.clip(img_float, clip_min, clip_max)
    
    # Avoid division by zero
    range_val = clip_max - clip_min
    if range_val < 1e-8:
        return np.full_like(img_clipped, 128, dtype=np.uint8)
    
    img_normalized = 255 * (img_clipped - clip_min) / range_val
    
    return img_normalized.astype(np.uint8)


def read_mrc_as_image(mrc_path: str) -> Image.Image:
    """
    Read an MRC file and convert it to a PIL Image.
    
    Args:
        mrc_path: Path to the MRC file
        
    Returns:
        PIL Image object
    """
    with mrcfile.open(mrc_path) as mrc:
        # Get the data and convert to float32
        data = mrc.data.astype(np.float32)
        
        # Create PIL Image (normalization will be done later)
        return Image.fromarray(data.astype(np.uint8))


def load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    """
    Load an image file or MRC file and return as PIL Image and raw data.
    Automatically normalizes bright images for proper display.
    
    Args:
        path: Path to the image or MRC file
        
    Returns:
        Tuple of (PIL Image object, raw numpy array)
    """
    if path.suffix.lower() == '.mrc':
        with mrcfile.open(str(path)) as mrc:
            data = mrc.data.astype(np.float32)
            # Use percentile-based normalization for MRC files to handle outliers
            normalized_data = normalize_image(data, use_percentiles=True, low_percentile=1.0, high_percentile=99.0)
            return Image.fromarray(normalized_data), data
    else:
        img = Image.open(path)
        raw_data = np.array(img.convert("L")).astype(np.float32)
        
        # Check if the image is very bright (e.g., from projection images)
        # If the image has values much higher than typical 8-bit range, normalize it
        if raw_data.max() > 1000 or raw_data.std() > 1000:
            print(f"Detected bright image (max: {raw_data.max():.1f}, std: {raw_data.std():.1f}). Applying percentile-based normalization...")
            # Use percentile-based normalization to remove outliers on both ends
            normalized_data = normalize_image(raw_data, use_percentiles=True, low_percentile=0.5, high_percentile=99.5)
            return Image.fromarray(normalized_data), raw_data
        else:
            return img, raw_data


def fft_image_with_matplotlib(region: np.ndarray, contrast=2.0, return_array=False):
    """
    Compute FFT of a region and return as PIL Image.
    
    Args:
        region: Input image array
        contrast: Contrast parameter for normalization
        return_array: Whether to return array instead of image
        
    Returns:
        PIL Image of FFT
    """
    # Validate region size to prevent FFT errors
    if region.size == 0 or region.shape[0] == 0 or region.shape[1] == 0:
        raise ValueError(f"Invalid region size for FFT: {region.shape}")
    
    f = np.fft.fft2(region)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    normalized = normalize(magnitude, contrast)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(normalized, cmap='gray')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def compute_fft_image_region(cropped: Image.Image, contrast=2.0) -> Image.Image:
    """
    Compute FFT image from a cropped region.
    
    Args:
        cropped: PIL Image to process
        contrast: Contrast parameter for normalization
        
    Returns:
        PIL Image of FFT
    """
    arr = np.array(cropped.convert("L")).astype(np.float32)
    return fft_image_with_matplotlib(arr, contrast)


def compute_average_fft(cropped: Image.Image, apix: float = 1.0) -> Image.Image:
    """
    Compute the 1D rotational average of the 2D FFT from a cropped image.

    Args:
        cropped: A PIL.Image object (grayscale or RGB).
        apix: Pixel size in Ångstrom per pixel.

    Returns:
        A PIL.Image containing the 1D plot of average FFT intensity vs. 1/resolution.
    """
    arr = np.array(cropped.convert("L")).astype(np.float32)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Compute radial coordinates
    cy, cx = np.array(magnitude.shape) // 2
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(np.int32)
    # Compute radial average
    radial_sum = np.bincount(r.ravel(), magnitude.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / (radial_count + 1e-8)

    # Convert to spatial frequency
    freqs = np.arange(len(radial_profile)) / (arr.shape[0] * apix)
    inverse_resolution = freqs  # in 1/Å

    # Determine index range for 1/3.7 to 1/2
    x_min, x_max = 1 / 3.7, 1 / 2.0
    mask = (inverse_resolution >= x_min) & (inverse_resolution <= x_max)

    # Plot
    fig, ax = plt.subplots(dpi=100)
    ax.plot(inverse_resolution[mask], np.log1p(radial_profile[mask]))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(radial_profile[mask].min(), radial_profile[mask].max())
    ax.set_xlabel("1 / Resolution (1/Å)")
    ax.set_ylabel("Log(Average FFT intensity)")
    ax.set_title("1D FFT Radial Profile")
    ax.grid(True)

    # Save to PIL.Image
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def calculate_apix_from_distance(distance_pixels: float, resolution: float, size: int) -> float:
    """
    Calculate apix from distance in pixels.
    
    Args:
        distance_pixels: Distance from center in pixels
        resolution: Resolution in Angstroms
        size: Image size in pixels
        
    Returns:
        Apix value in Å/pixel, or None if invalid
    """
    if distance_pixels <= 0:
        return None
    return (distance_pixels * resolution) / size


def calculate_distance_from_apix(apix_value: float, resolution: float, size: int) -> float:
    """
    Calculate distance in pixels from apix value.
    
    Args:
        apix_value: Apix value in Å/pixel
        resolution: Resolution in Angstroms
        size: Image size in pixels
        
    Returns:
        Distance from center in pixels, or None if invalid
    """
    if apix_value <= 0:
        return None
    return (apix_value * size) / resolution


def calculate_tilt_angle(small_axis: float, large_axis: float) -> float:
    """
    Calculate tilt angle from ellipse axes.
    
    Args:
        small_axis: Semi-minor axis length
        large_axis: Semi-major axis length
        
    Returns:
        Tilt angle in radians
    """
    if large_axis <= 0:
        return 0.0
    return math.acos(small_axis / large_axis)


def get_resolution_info(resolution_type: str, custom_resolution: float = None) -> tuple[float, str]:
    """
    Get resolution value and color based on resolution type.
    
    Args:
        resolution_type: Type of resolution (Graphene, Gold, Ice, Custom)
        custom_resolution: Custom resolution value if type is Custom
        
    Returns:
        Tuple of (resolution_value, color)
    """
    if resolution_type == "Graphene (2.13 Å)":
        return 2.13, "red"
    elif resolution_type == "Gold (2.355 Å)":
        return 2.355, "orange"
    elif resolution_type == "Ice (3.661 Å)":
        return 3.661, "blue"
    elif resolution_type == "Custom":
        return custom_resolution, "green"
    return None, None


def resolution_to_radius(res_angstrom: float, image_size: int, apix: float) -> float:
    """
    Calculate radius in pixels from resolution in Angstroms.
    
    Args:
        res_angstrom: Resolution in Angstroms
        image_size: Image size in pixels
        apix: Pixel size in Å/pixel
        
    Returns:
        Radius in pixels
    """
    return (image_size * apix) / res_angstrom


def get_image(filename: str, target_apix: float = None, low_pass_angstrom: float = 0, high_pass_angstrom: float = 0) -> tuple[np.ndarray, float, float]:
    """
    Load and process an image file (MRC, TIFF, PNG, etc.) with optional filtering.
    
    Args:
        filename: Path to the image file
        target_apix: Target pixel size in Angstroms (if None, use original)
        low_pass_angstrom: Low-pass filter in Angstroms (0 = no filter)
        high_pass_angstrom: High-pass filter in Angstroms (0 = no filter)
        
    Returns:
        Tuple of (processed_data, target_apix, original_apix)
    """
    # Load the image
    if filename.lower().endswith('.mrc'):
        with mrcfile.open(filename) as mrc:
            original_apix = round(float(mrc.voxel_size.x), 4)
            data = mrc.data.squeeze()
    else:
        # For other formats, assume 1 Å/pixel if not specified
        original_apix = 1.0
        img = Image.open(filename)
        data = np.array(img.convert("L")).astype(np.float32)
    
    # If no target apix specified, use original
    if target_apix is None:
        target_apix = original_apix
    
    ny, nx = data.shape
    
    # Resize if target apix is different
    if abs(target_apix - original_apix) > 1e-6:
        new_ny = int(ny * original_apix / target_apix + 0.5) // 2 * 2
        new_nx = int(nx * original_apix / target_apix + 0.5) // 2 * 2
        data = resize_local_mean(image=data, output_shape=(new_ny, new_nx))
    
    # Apply filters if specified
    if low_pass_angstrom > 0 or high_pass_angstrom > 0:
        # Simple frequency domain filtering
        f = np.fft.fft2(data)
        fshift = np.fft.fftshift(f)
        
        # Create frequency mask
        cy, cx = np.array(fshift.shape) // 2
        y, x = np.indices(fshift.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Low-pass filter
        if low_pass_angstrom > 0:
            low_pass_freq = 2 * target_apix / low_pass_angstrom
            low_pass_mask = r <= low_pass_freq
            fshift = fshift * low_pass_mask
        
        # High-pass filter
        if high_pass_angstrom > 0:
            high_pass_freq = 2 * target_apix / high_pass_angstrom
            high_pass_mask = r >= high_pass_freq
            fshift = fshift * high_pass_mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        data = np.real(np.fft.ifft2(f_ishift))
    
    return data, target_apix, original_apix


def plot_image(image_data: np.ndarray, title: str, apix: float, plot_height: int = None, plot_width: int = None) -> 'plotly.graph_objects.Figure':
    """
    Create a Plotly heatmap figure for displaying image data using plotly.express.imshow.
    """
    fig = px.imshow(
        image_data,
        color_continuous_scale="gray",
        aspect="equal",  # Force square aspect ratio
        origin="upper",
        labels=dict(x="", y="", color=""),
    )
    # Hide axes and colorbar
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_coloraxes(showscale=False)
    # Remove title if not provided
    if title and title.strip():
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=None)
    # Set autosize and margins
    fig.update_layout(
        autosize=True,
        width=plot_width,
        height=plot_height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
    )
    
    # Enable zoom and selection interactions
    fig.update_layout(
        dragmode='zoom',
        modebar=dict(
            add=['zoom', 'pan', 'reset+autorange', 'select2d', 'lasso2d']
        )
    )
    
    return fig


def create_fft_1d_plotly_figure(plot_data: dict, resolution: float, region: Image.Image, 
                               size: int, zoom_state: dict, shared_x_range=None) -> 'plotly.graph_objects.Figure':
    """
    Create a Plotly figure for the 1D FFT radial profile.
    
    Args:
        plot_data: Dictionary containing plot data from compute_fft_1d_data
        resolution: Current resolution value
        region: Current image region
        size: Image size constant
        zoom_state: Current zoom state dictionary
        
    Returns:
        Plotly Figure object
    """
    if plot_data is None:
        return go.Figure()

    # Create plotly figure
    fig = go.Figure()
    
    # Add main trace with custom hover text
    hover_text = []
    
    for i, x_val in enumerate(plot_data['x_data']):
        # Check if x_val is spatial frequency or radius
        if 'target_resolution' in plot_data:
            # This is from non-uniform FFT, x_val is spatial frequency (1/Å)
            if x_val > 0:
                resolution_val = 1 / x_val
                resolution_str = f"{resolution_val:.2f} Å"
                freq_str = f"{x_val:.4f} Å⁻¹"
                
                # Calculate corresponding apix for the target resolution
                if region is not None:
                    region_size = region.size[0] if hasattr(region, 'size') else region.shape[0]
                    # For non-uniform FFT, we need to estimate the equivalent radius
                    # Using the target resolution from plot_data
                    target_res = plot_data.get('target_resolution', resolution_val)
                    equivalent_radius = (region_size * plot_data.get('nominal_apix', 1.0)) / target_res
                    apix_value = (equivalent_radius * resolution_val) / region_size
                    apix_str = f"{apix_value:.3f} Å/px"
                else:
                    apix_str = "N/A"
            else:
                resolution_str = "∞ Å"
                freq_str = f"{x_val:.4f} Å⁻¹"
                apix_str = "N/A"
            
            # Create hover info with spatial frequency, resolution, and apix
            hover_info = f"Spatial frequency: {freq_str}<br>Resolution: {resolution_str}<br>Apix: {apix_str}"
        else:
            # This is from traditional FFT, x_val is radius in pixels
            if resolution is not None and x_val > 0:
                if region is not None:
                    region_size = region.size[0]  # Unbinned region size
                    # x_val is in unbinned FFT pixel coordinates
                    fft_radius = x_val
                    
                    # Calculate apix using the correct formula for unbinned coordinates
                    apix_value = (fft_radius * resolution) / region_size
                    apix_str = f"{apix_value:.3f}"
                else:
                    apix_str = "N/A"
            else:
                apix_str = "N/A"
            
            # Create hover info with radius and apix
            hover_info = f"Radius: {x_val:.1f} pixels<br>Apix: {apix_str} Å/px"
        
        hover_text.append(hover_info)
    
    fig.add_trace(go.Scatter(
        x=plot_data['x_data'],
        y=plot_data['y_data'],
        mode='lines',
        name=plot_data['profile_label'],
        line=dict(color='blue', width=2),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    # Set axis limits based on shared range, zoom state, or defaults
    if shared_x_range is not None:
        xlim = shared_x_range
    elif zoom_state['x_range'] is not None and zoom_state['y_range'] is not None:
        xlim = zoom_state['x_range']
        ylim = zoom_state['y_range']
    else:
        xlim = (plot_data['x_min'], plot_data['x_max'])
        
    # Set y limits if not already set
    if 'ylim' not in locals():
        # Calculate y limits
        y_min = plot_data['y_data'].min()
        y_max = plot_data['y_data'].max()
        if y_max > y_min:
            y_range = y_max - y_min
            if y_range > 0:
                ylim = (y_min - y_range * 0.1, y_max + y_range * 0.1)
            else:
                ylim = (y_min, y_max * 1.1)
        else:
            if y_max > 0:
                ylim = (y_max * 0.9, y_max * 1.1)
            else:
                ylim = (-0.1, 0.1)

    # Update layout with hover functionality
    # Set x-axis title and formatting based on data type
    if 'target_resolution' in plot_data:
        x_axis_title = "Spatial frequency (1/Res)"
        
        # Create custom tick values and labels for 1/Res format
        x_min, x_max = xlim
        # Generate reasonable tick positions
        n_ticks = 6  # Number of ticks
        tick_vals = np.linspace(x_min, x_max, n_ticks)
        tick_text = []
        for val in tick_vals:
            if val > 0:
                res = 1 / val
                tick_text.append(f"1/{res:.2f}")
            else:
                tick_text.append("1/∞")
        
        xaxis_config = dict(
            range=xlim, 
            showgrid=True, 
            matches='x',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickmode='array'
        )
    else:
        x_axis_title = "Radius (pixels)"
        xaxis_config = dict(range=xlim, showgrid=True, matches='x')
        
    fig.update_layout(
        title="1D FFT Radial Profile",
        xaxis_title=x_axis_title,
        yaxis_title=plot_data['y_axis_title'],
        xaxis=xaxis_config,
        yaxis=dict(range=ylim, showgrid=True),
        showlegend=True,
        legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom'),
        height=200,
        width=650,  # Fixed width to match other plots
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=False,
        hovermode="x unified"
    )
    return fig


def compute_fft_polar_heatmap_data(region: Image.Image, apix: float, resolution_type: str = None, 
                                  custom_resolution: float = None, use_for_range: bool = True) -> dict:
    """
    Calculate polar heatmap data for FFT profile near radius of interest.
    
    Args:
        region: Image region to analyze
        apix: Pixel size in Å/pixel (used for radius range calculation)
        resolution_type: Type of resolution for position calculation
        custom_resolution: Custom resolution value
        use_for_range: Whether to use this apix for range calculation (vs just returning info)
        
    Returns:
        Dictionary containing polar heatmap data
    """
    # Compute FFT and get power spectrum
    arr = np.array(region.convert("L")).astype(np.float32)
    
    # Validate array size
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Invalid region size for FFT: {arr.shape}")
    
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    pwr = np.abs(fshift)
    
    cy, cx = np.array(pwr.shape) // 2
    
    # Get expected radius based on resolution
    if resolution_type and resolution_type != "Custom":
        resolution_map = {
            "Graphene (100)": 2.13,
            "Graphene (110)": 1.23,
            "Gold (111)": 2.35,
            "Gold (200)": 2.04,
            "Gold (220)": 1.44
        }
        resolution = resolution_map.get(resolution_type, 2.13)
    else:
        resolution = custom_resolution if custom_resolution else 2.13
    
    # Calculate expected radius in pixels using unbinned coordinates
    # Use the actual region size for accurate calculations
    region_size = arr.shape[0]
    expected_radius = resolution_to_radius(resolution, region_size, apix)
    
    # Match the 1D FFT range: 10% to 75% of total radius
    total_radius = min(cy, cx)
    r_min = max(1, int(total_radius * 0.1))  # Start from 10% of total radius  
    r_max = int(total_radius * 0.75)  # End at 75% of total radius
    
    # Create angular and radial coordinates
    angles = np.linspace(0, 360, 360, endpoint=False)  # 0-360 degrees
    radii = np.arange(r_min, r_max + 1)
    
    # Create the heatmap data (angles x radii for correct axis orientation)
    heatmap_data = np.zeros((len(angles), len(radii)))
    
    for i, angle in enumerate(angles):
        for j, r in enumerate(radii):
            # Convert polar to cartesian
            theta = np.radians(angle)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            
            # Bilinear interpolation for sub-pixel accuracy
            x0, x1 = int(np.floor(x)), int(np.ceil(x))
            y0, y1 = int(np.floor(y)), int(np.ceil(y))
            
            # Check bounds
            if (x0 >= 0 and x1 < pwr.shape[1] and y0 >= 0 and y1 < pwr.shape[0]):
                # Bilinear interpolation weights
                wx = x - x0
                wy = y - y0
                
                # Interpolate
                intensity = (1-wx)*(1-wy)*pwr[y0,x0] + wx*(1-wy)*pwr[y0,x1] + \
                           (1-wx)*wy*pwr[y1,x0] + wx*wy*pwr[y1,x1]
                
                heatmap_data[i, j] = intensity
    
    return {
        'heatmap_data': heatmap_data,
        'angles': angles,
        'radii': radii,
        'expected_radius': expected_radius,
        'resolution': resolution
    }


def compute_fft_1d_data(region: Image.Image, apix: float, use_mean_profile: bool = False, 
                       log_y: bool = False, smooth: bool = False, window_size: int = 3,
                       detrend: bool = False, resolution_type: str = None, 
                       custom_resolution: float = None) -> dict:
    """
    Calculate the data needed for the 1D FFT plot.
    
    Args:
        region: Image region to analyze
        apix: Pixel size in Å/pixel
        use_mean_profile: Whether to use mean or max profile
        log_y: Whether to use log scale for y-axis
        smooth: Whether to apply smoothing
        window_size: Window size for smoothing
        detrend: Whether to detrend the signal
        resolution_type: Type of resolution for position calculation
        custom_resolution: Custom resolution value
        
    Returns:
        Dictionary containing plot data
    """
    # Compute FFT and get power spectrum
    arr = np.array(region.convert("L")).astype(np.float32)
    
    # Validate array size to prevent FFT errors
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Invalid region size for FFT: {arr.shape}")
    
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    pwr = np.abs(fshift)  # Power spectrum

    if use_mean_profile:
        # Compute radial average profile
        cy, cx = np.array(pwr.shape) // 2
        y, x = np.indices(pwr.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(np.int32)
        radial_sum = np.bincount(r.ravel(), pwr.ravel())
        radial_count = np.bincount(r.ravel())
        pwr_1d = radial_sum / (radial_count + 1e-8)
        profile_label = "FFT radial average"
    else:
        # Calculate radial max profile - max value at each radius
        cy, cx = np.array(pwr.shape) // 2
        y, x = np.indices(pwr.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(np.int32)
        
        # Find max value at each radius
        max_radial = np.zeros(r.max() + 1)
        for radius in range(r.max() + 1):
            mask = (r == radius)
            if np.any(mask):
                max_radial[radius] = np.max(pwr[mask])
        
        pwr_1d = max_radial
        profile_label = "FFT radial max"

    # Use radius in pixels as x-axis
    radius_pixels = np.arange(len(pwr_1d))

    # Set x-axis limits to start from 10% of the total radius
    x_min = int(len(pwr_1d) * 0.1)  # Start from 10% of total radius
    x_max = int(len(pwr_1d) * 0.75)  # Keep the upper limit
    mask = (radius_pixels >= x_min) & (radius_pixels <= x_max)

    # Plot data
    y_data = pwr_1d[mask]
    
    # Ensure we have valid data
    if len(y_data) == 0 or np.all(y_data == 0):
        # Fallback: create a simple plot with some data
        y_data = np.ones_like(radius_pixels[mask])
    
    if log_y:
        y_data = np.log1p(y_data)  # log1p is safe for positive values
        y_axis_title = "Log(FFT intensity)"
    else:
        y_axis_title = "FFT intensity"

    # Apply smoothing to y_data using a moving average
    if smooth:
        kernel = np.ones(window_size) / window_size
        # Determine padding amount for mode='same'
        pad_amount = (len(kernel) - 1) // 2
        
        # Pad the signal with 'reflect' mode
        padded_y_data = np.pad(y_data, pad_width=pad_amount, mode='reflect')
        
        # Perform convolution with the padded signal
        y_data = np.convolve(padded_y_data, kernel, mode='valid')
        
        y_data = y_data - y_data.min()
        
    # Detrend the signal by fitting and subtracting a linear baseline
    if detrend:
        # Fit a first-degree polynomial to get trend
        m, b = np.polyfit(radius_pixels[mask], y_data, 1)
        # Compute and subtract baseline
        baseline = m * radius_pixels[mask] + b
        y_data = y_data - baseline
        # Shift back to positive values
        y_data = y_data - y_data.min()

    # Calculate expected resolution positions for hover information
    resolution_positions = {}
    resolution, _ = get_resolution_info(resolution_type, custom_resolution)
    if resolution is not None:
        if resolution_type == "Graphene (2.13 Å)":
            radius_213 = (arr.shape[0] * apix) / 2.13
            if x_min <= radius_213 <= x_max:
                resolution_positions['graphene'] = radius_213
        elif resolution_type == "Gold (2.355 Å)":
            radius_235 = (arr.shape[0] * apix) / 2.355
            if x_min <= radius_235 <= x_max:
                resolution_positions['gold'] = radius_235
        elif resolution_type == "Ice (3.661 Å)":
            radius_366 = (arr.shape[0] * apix) / 3.661
            if x_min <= radius_366 <= x_max:
                resolution_positions['ice'] = radius_366
        elif resolution_type == "Custom":
            radius_custom = (arr.shape[0] * apix) / custom_resolution
            if x_min <= radius_custom <= x_max:
                resolution_positions['custom'] = radius_custom

    # Use unbinned coordinates for accurate apix calculations
    return {
        'x_data': radius_pixels[mask],
        'y_data': y_data,
        'profile_label': profile_label,
        'y_axis_title': y_axis_title,
        'x_min': x_min,
        'x_max': x_max,
        'arr_shape': arr.shape,
        'resolution_positions': resolution_positions
    }


def bin_image(image_data: np.ndarray, target_size: int = 1000) -> np.ndarray:
    """
    Bin an image to approximately target_size x target_size pixels.
    
    Args:
        image_data: Input image array
        target_size: Target size for the binned image
        
    Returns:
        Binned image array
    """
    h, w = image_data.shape
    
    # Calculate binning factor to get close to target size
    bin_factor = max(1, int(min(h, w) / target_size))
    
    # Calculate new dimensions
    new_h = h // bin_factor
    new_w = w // bin_factor
    
    # Use resize_local_mean for high-quality downsampling
    binned_data = resize_local_mean(image=image_data, output_shape=(new_h, new_w))
    
    return binned_data


def get_image_with_binning(filename: str, target_size: int = 1000, target_apix: float = None, 
                          low_pass_angstrom: float = 0, high_pass_angstrom: float = 0) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Load and process an image file with binning for display.
    
    Args:
        filename: Path to the image file
        target_size: Target size for binned image (default 1000)
        target_apix: Target pixel size in Angstroms (if None, use original)
        low_pass_angstrom: Low-pass filter in Angstroms (0 = no filter)
        high_pass_angstrom: High-pass filter in Angstroms (0 = no filter)
        
    Returns:
        Tuple of (original_data, binned_data, target_apix, original_apix)
    """
    # Load the original image
    if filename.lower().endswith('.mrc'):
        with mrcfile.open(filename) as mrc:
            original_apix = round(float(mrc.voxel_size.x), 4)
            original_data = mrc.data.squeeze()
    else:
        # For other formats, assume 1 Å/pixel if not specified
        original_apix = 1.0
        img = Image.open(filename)
        original_data = np.array(img.convert("L")).astype(np.float32)
    
    # If no target apix specified, use original
    if target_apix is None:
        target_apix = original_apix
    
    # Apply filters to original data if specified
    if low_pass_angstrom > 0 or high_pass_angstrom > 0:
        # Simple frequency domain filtering
        f = np.fft.fft2(original_data)
        fshift = np.fft.fftshift(f)
        
        # Create frequency mask
        cy, cx = np.array(fshift.shape) // 2
        y, x = np.indices(fshift.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Low-pass filter
        if low_pass_angstrom > 0:
            low_pass_freq = 2 * target_apix / low_pass_angstrom
            low_pass_mask = r <= low_pass_freq
            fshift = fshift * low_pass_mask
        
        # High-pass filter
        if high_pass_angstrom > 0:
            high_pass_freq = 2 * target_apix / high_pass_angstrom
            high_pass_mask = r >= high_pass_freq
            fshift = fshift * high_pass_mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        original_data = np.real(np.fft.ifft2(f_ishift))
    
    # Create binned version for display
    binned_data = bin_image(original_data, target_size)
    
    return original_data, binned_data, target_apix, original_apix


def extract_region_from_original(original_data: np.ndarray, binned_data: np.ndarray, 
                                x_range: tuple, y_range: tuple, target_size: int = 1000) -> np.ndarray:
    """
    Extract a region from the original image based on zoom coordinates from binned image.
    
    Args:
        original_data: Original full-resolution image data
        binned_data: Binned image data used for display
        x_range: (x_min, x_max) in binned image coordinates
        y_range: (y_min, y_max) in binned image coordinates
        target_size: Target size for FFT calculation
        
    Returns:
        Extracted region as numpy array
    """
    orig_h, orig_w = original_data.shape
    binned_h, binned_w = binned_data.shape
    
    # Calculate scale factors
    scale_x = orig_w / binned_w
    scale_y = orig_h / binned_h
    
    # Convert binned coordinates to original coordinates
    orig_x1 = int(x_range[0] * scale_x)
    orig_y1 = int(y_range[0] * scale_y)
    orig_x2 = int(x_range[1] * scale_x)
    orig_y2 = int(y_range[1] * scale_y)
    
    # Ensure bounds are within original image
    orig_x1 = max(0, orig_x1)
    orig_y1 = max(0, orig_y1)
    orig_x2 = min(orig_w, orig_x2)
    orig_y2 = min(orig_h, orig_y2)
    
    # Extract region
    region = original_data[orig_y1:orig_y2, orig_x1:orig_x2]
    
    # If region is smaller than target_size, return as is
    if region.shape[0] < target_size or region.shape[1] < target_size:
        return region
    
    # Otherwise, bin the region to target_size
    return bin_image(region, target_size)


def extract_region_no_binning(original_data: np.ndarray, binned_data: np.ndarray, 
                             x_range: tuple, y_range: tuple) -> np.ndarray:
    """
    Extract a region from the original image based on zoom coordinates from binned image.
    Never bins the data - returns full-resolution region for accurate FFT analysis.
    
    Args:
        original_data: Original full-resolution image data
        binned_data: Binned image data used for display
        x_range: (x_min, x_max) in binned image coordinates
        y_range: (y_min, y_max) in binned image coordinates
        
    Returns:
        Extracted region as numpy array (full resolution, no binning)
    """
    orig_h, orig_w = original_data.shape
    binned_h, binned_w = binned_data.shape
    
    # Calculate scale factors
    scale_x = orig_w / binned_w
    scale_y = orig_h / binned_h
    
    # Convert binned coordinates to original coordinates
    orig_x1 = int(x_range[0] * scale_x)
    orig_y1 = int(y_range[0] * scale_y)
    orig_x2 = int(x_range[1] * scale_x)
    orig_y2 = int(y_range[1] * scale_y)
    
    # Ensure bounds are within original image
    orig_x1 = max(0, orig_x1)
    orig_y1 = max(0, orig_y1)
    orig_x2 = min(orig_w, orig_x2)
    orig_y2 = min(orig_h, orig_y2)
    
    # Extract region (no binning)
    region = original_data[orig_y1:orig_y2, orig_x1:orig_x2]
    
    return region


def create_fft_2d_plotly_figure(
    fft_data: np.ndarray,
    overlays: dict = None,
    apix: float = 1.0,
    resolution_type: str = None,
    custom_resolution: float = None,
    size: int = 360,
    contrast: float = 2.0,
    title: str = None
) -> 'plotly.graph_objects.Figure':
    """
    Create a Plotly figure for the 2D FFT with overlays (resolution circles, markers, ellipses).
    Args:
        fft_data: 2D FFT magnitude array (already normalized to 0-255)
        overlays: dict with keys 'mode', 'resolution_click_x', 'resolution_click_y', 'lattice_points', 'ellipse_params', 'zoom_factor'
        apix: pixel size in Angstroms
        resolution_type: string for resolution type
        custom_resolution: float for custom resolution
        size: image size (for scaling overlays)
        contrast: contrast parameter for normalization
        title: optional plot title
    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    from plotly.colors import make_colorscale

    # Create the base heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=fft_data,
            colorscale="gray",
            showscale=False,
            zmin=0,
            zmax=255,
            hoverinfo="skip",
        )
    )
    # Hide axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Set layout with square aspect ratio
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
        dragmode='zoom',
        title=title or None,
        clickmode='event',
    )
    # Force square aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # Add overlays if provided
    if overlays is not None:
        mode = overlays.get('mode', 'Resolution Ring')
        zoom_factor = overlays.get('zoom_factor', 1.0)
        center = size / 2 * zoom_factor
        # Resolution circles
        if mode == 'Resolution Ring':
            from compute import resolution_to_radius
            if resolution_type == "Graphene (2.13 Å)":
                r = resolution_to_radius(2.13, size, apix) * zoom_factor
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=center - r, y0=center - r, x1=center + r, y1=center + r,
                              line_color="red", line_width=2)
            if resolution_type == "Gold (2.355 Å)":
                r = resolution_to_radius(2.355, size, apix) * zoom_factor
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=center - r, y0=center - r, x1=center + r, y1=center + r,
                              line_color="orange", line_width=2)
            if resolution_type == "Ice (3.661 Å)":
                r = resolution_to_radius(3.661, size, apix) * zoom_factor
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=center - r, y0=center - r, x1=center + r, y1=center + r,
                              line_color="blue", line_width=2)
            if resolution_type == "Custom":
                r = resolution_to_radius(custom_resolution, size, apix) * zoom_factor
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=center - r, y0=center - r, x1=center + r, y1=center + r,
                              line_color="green", line_width=2)
        # Crosshair
        if mode == 'Resolution Ring' and overlays.get('resolution_click_x') is not None:
            x = overlays['resolution_click_x'] * zoom_factor
            y = overlays['resolution_click_y'] * zoom_factor
            marker_size = 10
            fig.add_shape(type="line", x0=x-marker_size, y0=y, x1=x+marker_size, y1=y, line_color="yellow", line_width=2)
            fig.add_shape(type="line", x0=x, y0=y-marker_size, x1=x, y1=y+marker_size, line_color="yellow", line_width=2)
        # Lattice points
        if mode == 'Lattice Point' and overlays.get('lattice_points'):
            for pt in overlays['lattice_points']:
                x, y = pt[0] * zoom_factor, pt[1] * zoom_factor
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=x-8, y0=y-8, x1=x+8, y1=y+8,
                              line_color="green", line_width=2)
        # Ellipse
        if mode == 'Lattice Point' and overlays.get('ellipse_params') is not None:
            a, b, theta = overlays['ellipse_params']
            a_scaled = a * zoom_factor
            b_scaled = b * zoom_factor
            cx, cy = center, center
            # Parametric ellipse
            t = np.linspace(0, 2*np.pi, 100)
            x_ellipse = a_scaled * np.cos(t)
            y_ellipse = b_scaled * np.sin(t)
            x_rot = x_ellipse * np.cos(theta) - y_ellipse * np.sin(theta)
            y_rot = x_ellipse * np.sin(theta) + y_ellipse * np.cos(theta)


def nufft_resolution_range(images, apix, res_low=0, res_high=0, r_samples=-1, theta_samples=180, return_R_only=False):
    """
    Non-uniform FFT implementation similar to fft_resolution_range from images2star.py.
    
    Args:
        images: Input image(s) as numpy array
        apix: Pixel size in Å/pixel
        res_low: Low resolution limit (Å), 0 means no limit
        res_high: High resolution limit (Å), 0 means Nyquist
        r_samples: Number of radial samples, -1 means min(image_shape)//2
        theta_samples: Number of angular samples
        return_R_only: If True, return only the R array
        
    Returns:
        Non-uniform FFT result or R array if return_R_only=True
    """
    if not FINUFFT_AVAILABLE:
        raise ImportError("finufft package is required for non-uniform FFT. Please install with: pip install finufft")
    
    R0 = 1 / res_low if res_low > 0 else 0
    R1 = 1 / res_high if res_high > 0 else 1 / (2 * apix)
    nr = r_samples if r_samples > 0 else min(images.shape[-2:]) // 2
    R = np.linspace(start=R0, stop=R1, num=nr, endpoint=True)
    
    if return_R_only:
        return R
    
    Theta = np.linspace(start=0, stop=np.pi, num=theta_samples, endpoint=False)
    Theta, R = np.meshgrid(Theta, R, indexing="ij")
    Y = (2 * np.pi * apix * R * np.sin(Theta)).flatten(order="C")
    X = (2 * np.pi * apix * R * np.cos(Theta)).flatten(order="C")
    
    if len(images.shape) > 2:
        if len(images) > 1:
            fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
        else:
            fft = nufft2d2(x=X, y=Y, f=images[0].astype(np.complex64), eps=1e-6)
    else:
        fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
    
    if len(images.shape) > 2:
        new_shape = list(images.shape[:-2]) + list(R.shape)
    else:
        new_shape = R.shape
    
    fft = fft.reshape(new_shape)
    return fft


def nufft_highest_point(images, apix, res_low=0, res_high=0, percentile=5, return_coords=False):
    """
    Non-uniform FFT implementation that samples only the highest intensity points.
    
    Args:
        images: Input image(s) as numpy array
        apix: Pixel size in Å/pixel
        res_low: Low resolution limit (Å), 0 means no limit
        res_high: High resolution limit (Å), 0 means Nyquist
        percentile: Percentage of highest intensity points to sample (default 5%)
        return_coords: If True, also return the spatial frequency coordinates
        
    Returns:
        Complex FFT result sampled at highest intensity points
        If return_coords=True, returns (fft_result, spatial_frequencies)
    """
    if not FINUFFT_AVAILABLE:
        raise ImportError("finufft package is required for non-uniform FFT. Please install with: pip install finufft")
    
    # Handle single image case
    if len(images.shape) == 2:
        images = images[np.newaxis, ...]
    
    # Get image dimensions
    ny, nx = images.shape[-2:]
    center_y, center_x = ny // 2, nx // 2
    
    # Create coordinate arrays
    y_coords, x_coords = np.ogrid[:ny, :nx]
    y_coords = y_coords - center_y
    x_coords = x_coords - center_x
    
    # Calculate radial distances in frequency space
    R_pixels = np.sqrt(x_coords**2 + y_coords**2)
    
    # Convert pixel distances to spatial frequencies (1/Å)
    R_freq = R_pixels / (min(ny, nx) * apix)
    
    # Apply resolution limits
    R0 = 1 / res_low if res_low > 0 else 0
    R1 = 1 / res_high if res_high > 0 else 1 / (2 * apix)
    
    # Create mask for resolution range
    mask = (R_freq >= R0) & (R_freq <= R1)
    
    # Get the first image for intensity-based sampling
    sample_image = images[0] if len(images) > 1 else images[0]
    
    # Calculate FFT for intensity analysis
    fft_full = np.fft.fftshift(np.fft.fft2(sample_image))
    fft_intensity = np.abs(fft_full)
    
    # Apply resolution mask
    masked_intensity = np.where(mask, fft_intensity, 0)
    
    # Find highest intensity points within the resolution range
    valid_indices = np.where(mask)
    valid_intensities = masked_intensity[valid_indices]
    
    if len(valid_intensities) == 0:
        # Handle division by zero for R0 for error message
        res_high_str = f"{1/R1:.2f}" if R1 > 0 else "inf"
        res_low_str = f"{1/R0:.2f}" if R0 > 0 else "inf"
        print(f"ERROR: No points found in the specified resolution range {res_high_str} - {res_low_str} Å")
        return None if not return_coords else (None, None)
    
    # Calculate threshold for top percentile
    threshold = np.percentile(valid_intensities, 100 - percentile)
    
    # Get coordinates of highest intensity points
    high_intensity_mask = masked_intensity >= threshold
    y_indices, x_indices = np.where(high_intensity_mask)
    
    # Convert to centered coordinates
    y_centered = y_indices - center_y
    x_centered = x_indices - center_x
    
    # Calculate spatial frequencies for the selected points
    spatial_freqs = np.sqrt(x_centered**2 + y_centered**2) / (min(ny, nx) * apix)
    
    # Convert to frequency coordinates for NUFFT
    # NUFFT expects coordinates in [-π, π] range
    Y = 2 * np.pi * y_centered / ny
    X = 2 * np.pi * x_centered / nx
    
    print(f"Selected {len(X)} points ({percentile}% of {np.sum(mask)} valid points)")
    # Handle division by zero for R0
    res_high_str = f"{1/R1:.2f}" if R1 > 0 else "inf"
    res_low_str = f"{1/R0:.2f}" if R0 > 0 else "inf"
    print(f"Resolution range: {res_high_str} - {res_low_str} Å")
    print(f"Intensity threshold: {threshold:.2e}")
    
    # Perform NUFFT for each image
    results = []
    for i in range(len(images)):
        if len(images) > 1:
            img = images[i]
        else:
            img = images[0] if len(images.shape) > 2 else images
            
        # Perform NUFFT at selected high-intensity points
        fft_result = nufft2d2(x=X, y=Y, f=img.astype(np.complex128), eps=1e-6)
        results.append(fft_result)
    
    if len(results) == 1:
        final_result = results[0]
    else:
        final_result = np.array(results)
    
    if return_coords:
        return final_result, spatial_freqs
    else:
        return final_result


def calibrateMag_process_one_region(region_image, apix, resolution, resolution_range_percent=10, r_samples=100, theta_samples=360):
    """
    Process one image region using non-uniform FFT similar to calibrateMag_process_one_micrograph.
    
    Args:
        region_image: PIL Image or numpy array of the region
        apix: Nominal pixel size in Å/pixel
        resolution: Resolution of interest in Å
        resolution_range_percent: Range around resolution as percentage (default 10%)
        r_samples: Number of radial samples
        theta_samples: Number of angular samples
        
    Returns:
        tuple: (pwr_curve, pwr) - 1D power curve and 2D power array
    """
    if not FINUFFT_AVAILABLE:
        # Fallback to regular FFT if finufft is not available
        print("Warning: finufft not available, using regular FFT fallback")
        return _fallback_fft_processing(region_image, apix, resolution, resolution_range_percent)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(region_image, Image.Image):
        images = np.array(region_image.convert("L")).astype(np.float32)
    else:
        images = region_image.astype(np.float32)
    
    # Ensure we have proper shape for nufft
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)
    
    # Calculate resolution range
    range_value = resolution * (resolution_range_percent / 100.0)
    res_low = resolution + range_value  # Lower spatial frequency (higher Å value)
    res_high = resolution - range_value  # Higher spatial frequency (lower Å value)
    
    # Ensure res_high doesn't go below Nyquist limit
    nyquist_resolution = 2 * apix
    if res_high < nyquist_resolution:
        res_high = nyquist_resolution
    
    try:
        # Compute non-uniform FFT
        fft = nufft_resolution_range(
            images, apix, res_low, res_high, r_samples, theta_samples, return_R_only=False
        )
        
        # Calculate power spectrum
        pwr = np.abs(fft)
        
        # Get 1D profile by taking maximum across angular samples
        pwr_1d = pwr.max(axis=tuple(range(len(pwr.shape) - 1)))
        
        # Normalize the signal
        pwr_1d -= np.median(pwr_1d)
        pwr_curve = pwr_1d / median_abs_deviation(pwr_1d)
        
        return (pwr_curve, pwr)
        
    except Exception as e:
        print(f"Non-uniform FFT failed: {e}, falling back to regular FFT")
        return _fallback_fft_processing(region_image, apix, resolution, resolution_range_percent)


def _fallback_fft_processing(region_image, apix, resolution, resolution_range_percent):
    """
    Fallback FFT processing using regular numpy FFT when finufft is not available.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(region_image, Image.Image):
        arr = np.array(region_image.convert("L")).astype(np.float32)
    else:
        arr = region_image.astype(np.float32)
    
    # Standard FFT processing
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    pwr_2d = np.abs(fshift)
    
    # Calculate radial profile
    cy, cx = np.array(pwr_2d.shape) // 2
    y, x = np.indices(pwr_2d.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(np.int32)
    
    # Calculate max radial profile
    max_radial = np.zeros(r.max() + 1)
    for radius in range(r.max() + 1):
        mask = (r == radius)
        if np.any(mask):
            max_radial[radius] = np.max(pwr_2d[mask])
    
    # Focus on resolution range of interest
    center_radius = (arr.shape[0] * apix) / resolution
    range_pixels = int(center_radius * (resolution_range_percent / 100.0))
    start_idx = max(0, int(center_radius - range_pixels))
    end_idx = min(len(max_radial), int(center_radius + range_pixels))
    
    pwr_1d = max_radial[start_idx:end_idx]
    
    # Normalize
    if len(pwr_1d) > 0:
        pwr_1d -= np.median(pwr_1d)
        pwr_curve = pwr_1d / (median_abs_deviation(pwr_1d) + 1e-8)
    else:
        pwr_curve = np.array([0])
    
    # Create a simple 2D representation for consistency
    pwr_2d_region = pwr_2d[cy-50:cy+50, cx-50:cx+50] if pwr_2d.shape[0] > 100 else pwr_2d
    
    return (pwr_curve, pwr_2d_region)


def compute_fft_1d_data_nufft(region: Image.Image, apix: float, use_mean_profile: bool = False,
                             log_y: bool = False, smooth: bool = False, window_size: int = 3,
                             detrend: bool = False, resolution_type: str = None,
                             custom_resolution: float = None) -> dict:
    """
    Calculate 1D FFT data using non-uniform FFT based on resolution of interest.
    This replaces the traditional uniform FFT with a targeted approach around specific resolutions.
    
    Args:
        region: Image region to analyze
        apix: Nominal pixel size in Å/pixel
        use_mean_profile: Whether to use mean or max profile (ignored in nufft version)
        log_y: Whether to use log scale for y-axis
        smooth: Whether to apply smoothing
        window_size: Window size for smoothing
        detrend: Whether to detrend the signal
        resolution_type: Type of resolution for targeted analysis
        custom_resolution: Custom resolution value
        
    Returns:
        Dictionary containing plot data
    """
    # Get resolution of interest
    resolution, _ = get_resolution_info(resolution_type, custom_resolution)
    if resolution is None:
        # Default to a reasonable resolution if none specified
        resolution = 3.0  # Å
    
    try:
        # Use non-uniform FFT processing
        pwr_curve, pwr_2d = calibrateMag_process_one_region(
            region_image=region,
            apix=apix,
            resolution=resolution,
            resolution_range_percent=10,  # 10% range around resolution
            r_samples=100,
            theta_samples=360
        )
        
        # Create x-axis data in spatial frequency (1/Å)
        range_value = resolution * 0.1  # 10% range
        res_low = resolution + range_value
        res_high = resolution - range_value
        
        # Convert resolutions to spatial frequencies (1/Å)
        freq_low = 1 / res_low if res_low > 0 else 0
        freq_high = 1 / res_high if res_high > 0 else 1 / (2 * apix)
        
        # Create x-axis in spatial frequency
        x_data = np.linspace(freq_low, freq_high, len(pwr_curve))
        y_data = pwr_curve
        
        print(f"Non-uniform FFT results:")
        print(f"  Resolution target: {resolution:.2f} Å")
        print(f"  Resolution range: {res_low:.2f} - {res_high:.2f} Å")
        print(f"  Spatial frequency range: {freq_low:.4f} - {freq_high:.4f} Å⁻¹")
        print(f"  Power curve length: {len(pwr_curve)}")
        print(f"  X-axis range: {x_data[0]:.4f} - {x_data[-1]:.4f} Å⁻¹")
        
        # Apply log scale if requested
        if log_y:
            y_data = np.log1p(np.abs(y_data))
            y_axis_title = "Log(FFT intensity)"
        else:
            y_axis_title = "FFT intensity"
        
        # Apply smoothing if requested
        if smooth and len(y_data) > window_size:
            kernel = np.ones(window_size) / window_size
            pad_amount = (len(kernel) - 1) // 2
            padded_y_data = np.pad(y_data, pad_width=pad_amount, mode='reflect')
            y_data = np.convolve(padded_y_data, kernel, mode='valid')
            y_data = y_data - y_data.min()
        
        # Apply detrending if requested
        if detrend and len(y_data) > 2:
            m, b = np.polyfit(x_data, y_data, 1)
            baseline = m * x_data + b
            y_data = y_data - baseline
            y_data = y_data - y_data.min()
        
        # Calculate resolution positions for display (in spatial frequency)
        resolution_positions = {}
        if resolution_type == "Graphene (2.13 Å)":
            resolution_positions['graphene'] = 1 / 2.13
        elif resolution_type == "Gold (2.355 Å)":
            resolution_positions['gold'] = 1 / 2.355
        elif resolution_type == "Ice (3.661 Å)":
            resolution_positions['ice'] = 1 / 3.661
        elif resolution_type == "Custom" and custom_resolution:
            resolution_positions['custom'] = 1 / custom_resolution
        
        return {
            'x_data': x_data,
            'y_data': y_data,
            'profile_label': f"Non-uniform FFT profile ({resolution:.2f}Å ±10%)",
            'y_axis_title': y_axis_title,
            'x_min': freq_low,
            'x_max': freq_high,
            'arr_shape': np.array(region).shape if isinstance(region, Image.Image) else region.shape,
            'resolution_positions': resolution_positions,
            'resolution_range': (res_low, res_high),
            'target_resolution': resolution,
            'nominal_apix': apix
        }
        
    except Exception as e:
        print(f"Non-uniform FFT computation failed: {e}")
        # Fallback to original uniform FFT method
        return compute_fft_1d_data(region, apix, use_mean_profile, log_y, smooth, window_size, detrend, resolution_type, custom_resolution)


def fft_resolution_range_region(region_data, apix, res_low=0, res_high=0, r_samples=-1, theta_samples=180, return_R_only=False):
    """
    Similar to fft_resolution_range from images2star.py but takes image region data instead of reading from file.
    
    Args:
        region_data: numpy array of image region data
        apix: Pixel size in Å/pixel
        res_low: Low resolution limit (Å), 0 means no limit
        res_high: High resolution limit (Å), 0 means Nyquist
        r_samples: Number of radial samples, -1 means min(image_shape)//2
        theta_samples: Number of angular samples
        return_R_only: If True, return only the R array
        
    Returns:
        Non-uniform FFT result or R array if return_R_only=True
    """
    if not FINUFFT_AVAILABLE:
        raise ImportError("finufft package is required for non-uniform FFT. Please install with: pip install finufft")
    
    # Ensure we have proper data type and shape
    if isinstance(region_data, Image.Image):
        images = np.array(region_data.convert("L")).astype(np.float32)
    else:
        images = region_data.astype(np.float32)
    
    # Ensure proper shape for nufft
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)
    
    R0 = 1 / res_low if res_low > 0 else 0
    R1 = 1 / res_high if res_high > 0 else 1 / (2 * apix)
    nr = r_samples if r_samples > 0 else min(images.shape[-2:]) // 2
    R = np.linspace(start=R0, stop=R1, num=nr, endpoint=True)
    
    if return_R_only:
        return R
    
    Theta = np.linspace(start=0, stop=np.pi, num=theta_samples, endpoint=False)
    Theta, R = np.meshgrid(Theta, R, indexing="ij")
    Y = (2 * np.pi * apix * R * np.sin(Theta)).flatten(order="C")
    X = (2 * np.pi * apix * R * np.cos(Theta)).flatten(order="C")
    
    if len(images.shape) > 2:
        if len(images) > 1:
            fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
        else:
            fft = nufft2d2(x=X, y=Y, f=images[0].astype(np.complex128), eps=1e-6)
    else:
        fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
    
    if len(images.shape) > 2:
        new_shape = list(images.shape[:-2]) + list(R.shape)
    else:
        new_shape = R.shape
    
    fft = fft.reshape(new_shape)
    return fft


def calibrateMag_process_one_region_advanced(region_data, apix, res_low, res_high, r_samples, theta_samples):
    """
    Similar to calibrateMag_process_one_micrograph from images2star.py but takes image region data instead of reading from file.
    
    Args:
        region_data: numpy array or PIL Image of the region data
        apix: Pixel size in Å/pixel
        res_low: Low resolution limit (Å)
        res_high: High resolution limit (Å)
        r_samples: Number of radial samples
        theta_samples: Number of angular samples
        
    Returns:
        tuple: (pwr_curve, pwr) - 1D power curve and 2D power array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(region_data, Image.Image):
        images = np.array(region_data.convert("L")).astype(np.float32)
    else:
        images = region_data.astype(np.float32)
    
    # Ensure proper shape for processing
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)
    
    try:
        # Use the non-uniform FFT function
        fft = fft_resolution_range_region(
            images,
            apix,
            res_low,
            res_high,
            r_samples,
            theta_samples,
            return_R_only=False,
        )
        
        # Calculate power spectrum
        pwr = np.abs(fft)
        
        # Get 1D profile by taking maximum across all dimensions except the last one
        pwr_1d = pwr.max(axis=tuple(range(len(pwr.shape) - 1)))
        
        # Normalize the signal similar to original function
        pwr_1d -= np.median(pwr_1d)
        pwr_curve = pwr_1d / median_abs_deviation(pwr_1d)
        
        n_ptcl = len(images)
        return (pwr_curve, pwr)
        
    except Exception as e:
        print(f"Advanced region processing failed: {e}, falling back to basic method")
        # Fallback to the basic calibrateMag_process_one_region function
        return calibrateMag_process_one_region(
            region_image=images[0] if len(images.shape) > 2 else images,
            apix=apix,
            resolution=(res_low + res_high) / 2,  # Use average as target resolution
            resolution_range_percent=((res_low - res_high) / ((res_low + res_high) / 2)) * 100,
            r_samples=r_samples,
            theta_samples=theta_samples
        )


def calibrateMag_process_one_region_fast(region_data, apix, res_low, res_high, r_samples, theta_samples):
    """
    Fast version using the proven nufft_resolution_range method.
    
    Args:
        region_data: numpy array or PIL Image of the region data
        apix: Pixel size in Å/pixel
        res_low: Low resolution limit (Å)
        res_high: High resolution limit (Å)
        r_samples: Number of radial samples
        theta_samples: Number of angular samples
        
    Returns:
        tuple: (pwr_curve, pwr) - 1D power curve and 2D power array
    """
    import time
    start_time = time.time()
    print(f"🚀 Starting FAST NuFFT calculation...")
    
    # Convert PIL Image to numpy array if needed
    if isinstance(region_data, Image.Image):
        images = np.array(region_data.convert("L")).astype(np.float32)
    else:
        images = region_data.astype(np.float32)
    
    # Ensure proper shape for processing (remove batch dimension if present)
    if len(images.shape) == 3:
        images = images[0]  # Take first image if batched
    
    try:
        # Use the fast nufft_resolution_range method that we proved is fastest
        print(f"📊 Using fast nufft_resolution_range method...")
        calc_time = time.time()
        
        fft_data = nufft_resolution_range(
            images, 
            apix, 
            res_low=res_low, 
            res_high=res_high,
            r_samples=r_samples, 
            theta_samples=theta_samples
        )
        
        calc_duration = time.time() - calc_time
        print(f"⚡ NuFFT calculation completed in {calc_duration:.3f} seconds")
        
        # Calculate power spectrum
        pwr = np.abs(fft_data)
        
        # Get 1D profile by taking maximum across angles (axis 0)
        pwr_curve = np.max(pwr, axis=0)
        
        # Normalize the signal similar to original function
        pwr_curve -= np.median(pwr_curve)
        pwr_curve = pwr_curve / median_abs_deviation(pwr_curve)
        
        total_time = time.time() - start_time
        print(f"✅ FAST NuFFT processing completed in {total_time:.3f} seconds total")
        
        return (pwr_curve, pwr)
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Fast method failed after {error_time:.3f}s: {e}")
        print(f"🔄 Falling back to original method...")
        
        # Fallback to original method
        return calibrateMag_process_one_region_advanced(
            region_data, apix, res_low, res_high, r_samples, theta_samples
        )


def estimate_best_apix_from_nufft(
    region_image,
    nominal_apix: float,
    target_resolution: float,
    res_window_frac: float = 0.05,   # ± window around target resolution as fraction
    r_samples: int = 200,            # radial samples from slider
    theta_samples: int = 360,        # angular samples from slider
    refine_once: bool = True,        # do one fixed-point refinement pass
    res_min_abs: float = None,       # absolute minimum resolution (Å), overrides res_window_frac
    res_max_abs: float = None,       # absolute maximum resolution (Å), overrides res_window_frac
):
    """
    Estimate the best apix value using NuFFT similar to estimate_best_apix from images2star.py.
    Always uses nominal apix as baseline and performs _single_pass analysis.
    
    Args:
        region_image: Image region data (PIL Image or numpy array)
        nominal_apix: Initial nominal pixel size (baseline)
        target_resolution: Target resolution in Angstroms
        res_window_frac: ± window around target resolution as fraction (e.g., 0.05 = 5%)
        r_samples: Number of radial samples (from slider)
        theta_samples: Number of angular samples (from slider)
        refine_once: Whether to do one refinement iteration
        res_min_abs: Absolute minimum spatial frequency (1/Å), overrides res_window_frac if provided
        res_max_abs: Absolute maximum spatial frequency (1/Å), overrides res_window_frac if provided
        
    Returns:
        tuple: (apix_est, peak_res_meas_A, meta_dict)
    """
    
    def _single_pass(apix):
        # Calculate resolution window around target
        if res_min_abs is not None and res_max_abs is not None:
            # Use absolute resolution bounds (convert from 1/Å to Å)
            res_min = 1.0 / res_max_abs  # Higher frequency -> lower resolution value in Å
            res_max = 1.0 / res_min_abs  # Lower frequency -> higher resolution value in Å
        else:
            # Use relative window around target resolution
            res_min = target_resolution * (1.0 - res_window_frac)
            res_max = target_resolution * (1.0 + res_window_frac)
        
        # Process the region using NuFFT with current apix estimate
        pwr_curve, pwr2d_raw = calibrateMag_process_one_region_advanced(
            region_data=region_image,
            apix=apix,
            res_low=res_min,
            res_high=res_max,
            r_samples=r_samples,
            theta_samples=theta_samples
        )
        
        if not isinstance((pwr_curve, pwr2d_raw), tuple) or len((pwr_curve, pwr2d_raw)) < 2:
            raise RuntimeError("calibrateMag_process_one_region_advanced must return (pwr_curve, pwr2d).")
        
        pwr_curve = np.asarray(pwr_curve, dtype=float)      # 1D max-over-angles profile
        pwr2d_raw = np.asarray(pwr2d_raw)                   # 2D power map
        
        # --- normalize pwr2d to shape (theta, r) with true radial length n_r ---
        p = np.squeeze(pwr2d_raw)
        while p.ndim > 2:      # collapse extra batch dims if present
            p = p.max(axis=0)
        if p.ndim != 2:
            raise RuntimeError(f"Expected 2D pwr2d after squeeze, got {p.shape}")
        
        # orient so first axis = theta, second = radius
        if p.shape[0] == theta_samples:
            p_theta_r = p
        elif p.shape[1] == theta_samples:
            p_theta_r = p.T
        else:
            p_theta_r = p if p.shape[0] <= p.shape[1] else p.T
        
        n_theta, n_r = p_theta_r.shape
        res_grid = np.linspace(res_min, res_max, n_r, endpoint=True)
        
        # --- envelope and fixed-angle selection ---
        r_env = p_theta_r.max(axis=0)           # envelope over theta
        # use 1D curve if it matches length; otherwise use envelope for peak index
        env = pwr_curve if len(pwr_curve) == n_r else r_env
        k = int(np.argmax(env))                 # radial index of global peak
        j_star = int(np.argmax(p_theta_r[:, k]))  # winning angle at that radius
        radial_slice = p_theta_r[j_star, :].astype(float)
        
        # --- sub-sample (3-point parabolic) refinement in Å (no helper) ---
        if 0 < k < n_r - 1:
            y1, y2, y3 = radial_slice[k-1], radial_slice[k], radial_slice[k+1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0:
                dx = (res_grid[1] - res_grid[0])
                delta = 0.5 * (y1 - y3) / denom
                r_peak_A = res_grid[k] + delta * dx
            else:
                r_peak_A = res_grid[k]
        else:
            r_peak_A = res_grid[k]
        
        peak_res_meas_A = float(r_peak_A)
        apix_est = apix * (target_resolution / peak_res_meas_A)
        
        # Calculate winning theta angle from j_star index
        theta_grid = np.linspace(0, 360, n_theta, endpoint=False)
        winning_theta = theta_grid[j_star] if j_star < len(theta_grid) else 0.0
        
        return apix_est, peak_res_meas_A, {
            "k": k,
            "j_star": j_star,
            "winning_theta": winning_theta,
            "peak_resolution": peak_res_meas_A,
            "res_grid": res_grid,
            "n_r": n_r,
            "n_theta": n_theta,
        }
    
    try:
        # first pass at nominal apix (always start from nominal!)
        apix_est, peak_res_meas_A, meta = _single_pass(nominal_apix)
        
        # optional single refinement pass starting from the corrected apix
        if refine_once:
            apix_est2, peak_res_meas_A2, meta2 = _single_pass(apix_est)
            if abs(peak_res_meas_A2 - target_resolution) <= abs(peak_res_meas_A - target_resolution):
                apix_est, peak_res_meas_A, meta = apix_est2, peak_res_meas_A2, meta2
        
        print(f"NuFFT Apix Estimation:")
        print(f"  Nominal apix: {nominal_apix:.4f} Å/px")
        print(f"  Target resolution: {target_resolution:.3f} Å")
        if res_min_abs is not None and res_max_abs is not None:
            print(f"  Spatial frequency range: {res_min_abs:.1f} - {res_max_abs:.1f} 1/Å")
            print(f"  Resolution window: {1.0/res_max_abs:.3f} - {1.0/res_min_abs:.3f} Å (absolute bounds)")
        else:
            print(f"  Resolution window: {target_resolution * (1.0 - res_window_frac):.3f} - {target_resolution * (1.0 + res_window_frac):.3f} Å (relative window)")
        print(f"  Measured peak resolution: {peak_res_meas_A:.3f} Å")
        print(f"  Winning theta angle: {meta['winning_theta']:.1f}°")
        print(f"  Peak position: θ={meta['winning_theta']:.1f}°, res={peak_res_meas_A:.3f}Å")
        print(f"  Estimated apix: {apix_est:.4f} Å/px")
        print(f"  Correction factor: {apix_est/nominal_apix:.4f}")
        
        return apix_est, peak_res_meas_A, meta
        
    except Exception as e:
        print(f"Error in NuFFT apix estimation: {e}")
        import traceback
        traceback.print_exc()
        # Return nominal values as fallback
        return nominal_apix, target_resolution, {
            "error": str(e),
            "k": 0,
            "j_star": 0,
            "res_grid": np.array([target_resolution]),
            "n_r": r_samples,
            "n_theta": theta_samples
        }


def detect_elliptical_distortion(fft_data: np.ndarray, apix: float, resolution: float,
                                 pixel_band_width: int = 10, min_peaks: int = 6) -> dict:
    """
    Automatically detect elliptical distortion in FFT by finding local maxima around a resolution band.

    Args:
        fft_data: 2D FFT magnitude array (already shifted to center)
        apix: Pixel size in Å/pixel
        resolution: Target resolution in Angstroms
        pixel_band_width: Width of band in pixels to search for peaks (default ±10 pixels)
        min_peaks: Minimum number of peaks required to fit ellipse (default 6)

    Returns:
        Dictionary containing:
            - 'success': bool - whether ellipse was successfully detected
            - 'ellipse_params': tuple (a, b, theta) - semi-major, semi-minor axis, rotation angle
            - 'peak_points': list of (x, y) peak coordinates
            - 'center': tuple (cx, cy) - center of FFT
            - 'target_radius': float - expected radius for the resolution
            - 'eccentricity': float - ellipse eccentricity
            - 'tilt_angle_deg': float - tilt angle in degrees
            - 'error_message': str - error message if failed
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import find_peaks

    # Get FFT center
    cy, cx = np.array(fft_data.shape) // 2

    # Calculate expected radius for the target resolution
    image_size = fft_data.shape[0]
    target_radius = resolution_to_radius(resolution, image_size, apix)

    # Define inner and outer radius for the search band
    r_inner = target_radius - pixel_band_width
    r_outer = target_radius + pixel_band_width

    print(f"Ellipse detection parameters:")
    print(f"  Target resolution: {resolution:.3f} Å")
    print(f"  Target radius: {target_radius:.2f} pixels")
    print(f"  Search band: {r_inner:.2f} - {r_outer:.2f} pixels")
    print(f"  Band width: ±{pixel_band_width} pixels")

    # Create coordinate arrays
    y_coords, x_coords = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
    y_centered = y_coords - cy
    x_centered = x_coords - cx

    # Calculate radial distance from center
    r_dist = np.sqrt(x_centered**2 + y_centered**2)

    # Create mask for the annular band
    band_mask = (r_dist >= r_inner) & (r_dist <= r_outer)

    if not np.any(band_mask):
        return {
            'success': False,
            'error_message': f'No pixels found in band {r_inner:.2f}-{r_outer:.2f} pixels',
            'peak_points': [],
            'center': (cx, cy),
            'target_radius': target_radius
        }

    print(f"  Pixels in search band: {np.sum(band_mask)}")

    # Apply local maximum filter to find peaks
    # Use a footprint of 5x5 to find local maxima
    footprint_size = 5
    local_max = maximum_filter(fft_data, size=footprint_size)

    # Find points that are local maxima AND in the band
    is_peak = (fft_data == local_max) & band_mask

    # Get peak coordinates and intensities
    peak_coords = np.argwhere(is_peak)
    peak_intensities = fft_data[is_peak]

    print(f"  Initial peaks found: {len(peak_coords)}")

    if len(peak_coords) < min_peaks:
        return {
            'success': False,
            'error_message': f'Only found {len(peak_coords)} peaks, need at least {min_peaks}',
            'peak_points': [],
            'center': (cx, cy),
            'target_radius': target_radius
        }

    # Sort peaks by intensity and keep the strongest ones
    # Keep at least min_peaks, but allow more if they're strong
    sorted_indices = np.argsort(peak_intensities)[::-1]

    # Use top peaks (at least min_peaks, but up to 2x if intensity is > 50% of max)
    max_intensity = peak_intensities[sorted_indices[0]]
    threshold = max_intensity * 0.5

    # Take at least min_peaks, or more if they're above threshold
    n_peaks_to_use = min_peaks
    for i in range(min_peaks, len(sorted_indices)):
        if peak_intensities[sorted_indices[i]] > threshold:
            n_peaks_to_use = i + 1
        else:
            break

    # Cap at reasonable number (e.g., 20 peaks)
    n_peaks_to_use = min(n_peaks_to_use, 20)

    selected_indices = sorted_indices[:n_peaks_to_use]
    selected_peaks = peak_coords[selected_indices]

    print(f"  Selected peaks: {len(selected_peaks)}")
    print(f"  Intensity threshold: {threshold:.2f} (50% of max {max_intensity:.2f})")

    # Convert to (x, y) coordinates centered at FFT center
    peak_points = []
    for peak in selected_peaks:
        y, x = peak
        peak_points.append((x - cx, y - cy))

    # Fit ellipse using the existing fit_ellipse_fixed_center function
    try:
        a, b, theta = fit_ellipse_fixed_center(peak_points, center=(0, 0))

        # Calculate eccentricity
        if a > b:
            eccentricity = np.sqrt(1 - (b**2 / a**2))
            # Calculate tilt angle from semi-axes
            tilt_angle_rad = calculate_tilt_angle(b, a)
        else:
            eccentricity = np.sqrt(1 - (a**2 / b**2))
            tilt_angle_rad = calculate_tilt_angle(a, b)

        tilt_angle_deg = np.degrees(tilt_angle_rad)

        print(f"  Ellipse fitted successfully:")
        print(f"    Semi-major axis: {max(a, b):.2f} pixels")
        print(f"    Semi-minor axis: {min(a, b):.2f} pixels")
        print(f"    Rotation angle: {np.degrees(theta):.1f}°")
        print(f"    Eccentricity: {eccentricity:.4f}")
        print(f"    Tilt angle: {tilt_angle_deg:.1f}°")

        # Convert peak points back to image coordinates for visualization
        peak_points_image = [(x + cx, y + cy) for x, y in peak_points]

        return {
            'success': True,
            'ellipse_params': (a, b, theta),
            'peak_points': peak_points_image,
            'center': (cx, cy),
            'target_radius': target_radius,
            'eccentricity': eccentricity,
            'tilt_angle_deg': tilt_angle_deg,
            'semi_major_axis': max(a, b),
            'semi_minor_axis': min(a, b),
            'rotation_angle_deg': np.degrees(theta),
            'n_peaks_used': len(selected_peaks)
        }

    except Exception as e:
        print(f"  Ellipse fitting failed: {e}")
        return {
            'success': False,
            'error_message': f'Ellipse fitting failed: {str(e)}',
            'peak_points': [(x + cx, y + cy) for x, y in peak_points],
            'center': (cx, cy),
            'target_radius': target_radius
        }