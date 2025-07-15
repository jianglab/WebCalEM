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


def normalize_image(img: np.ndarray, contrast=2.0) -> np.ndarray:
    """
    Normalize image data using mean and standard deviation.
    
    Args:
        img: Input image array
        contrast: Number of standard deviations to include in range
        
    Returns:
        Normalized image array (0-255 uint8)
    """
    # Convert to float32 for calculations
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    std = np.std(img_float)
    
    # Calculate clip range based on mean ± contrast * std
    clip_min = max(0, mean - contrast * std)
    clip_max = min(img_float.max(), mean + contrast * std)
    
    # Clip and normalize to 0-255 range
    img_clipped = np.clip(img_float, clip_min, clip_max)
    img_normalized = 255 * (img_clipped - clip_min) / (clip_max - clip_min + 1e-8)
    
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
    
    Args:
        path: Path to the image or MRC file
        
    Returns:
        Tuple of (PIL Image object, raw numpy array)
    """
    if path.suffix.lower() == '.mrc':
        with mrcfile.open(str(path)) as mrc:
            data = mrc.data.astype(np.float32)
            return Image.fromarray(data.astype(np.uint8)), data
    else:
        img = Image.open(path)
        return img, np.array(img.convert("L")).astype(np.float32)


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
                               size: int, zoom_state: dict) -> 'plotly.graph_objects.Figure':
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
        # Calculate apix using the radius and current resolution
        if resolution is not None and x_val > 0:
            # Convert from region coordinates to full FFT coordinates
            if region is not None:
                region_size = region.size[0]
                full_fft_size = size
                fft_radius = x_val * (full_fft_size / region_size)
                
                # Calculate apix using the same formula as other modes
                apix_value = (fft_radius * resolution) / full_fft_size
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
    
    # Set axis limits based on zoom state or defaults
    if zoom_state['x_range'] is not None and zoom_state['y_range'] is not None:
        xlim = zoom_state['x_range']
        ylim = zoom_state['y_range']
    else:
        xlim = (plot_data['x_min'], plot_data['x_max'])
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
    fig.update_layout(
        title="1D FFT Radial Profile",
        xaxis_title="Radius (pixels)",
        yaxis_title=plot_data['y_axis_title'],
        xaxis=dict(range=xlim, showgrid=True),
        yaxis=dict(range=ylim, showgrid=True),
        showlegend=True,
        legend=dict(x=0.02, y=0.02, xanchor='left', yanchor='bottom'),
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified"
    )
    return fig


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
            x_final = cx + x_rot
            y_final = cy + y_rot
            fig.add_trace(go.Scatter(x=x_final, y=y_final, mode='lines', line=dict(color='red', width=2), showlegend=False, hoverinfo='skip'))
        

    return fig 