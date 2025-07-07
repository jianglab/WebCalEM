from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import mrcfile
from scipy.stats import norm
import time
import asyncio
# ---------- Documentation ----------
"""Magnification Calibration Tool

This tool helps calibrate electron microscopes by analyzing test specimen images.
It calculates the pixel size (Angstroms/pixel) by measuring diffraction patterns
from known specimens like graphene, gold, or ice.

Key Features:
- Supports common image formats (.png, .tif) and MRC files
- Interactive FFT analysis with resolution circles
- Automatic pixel size detection
- Radial averaging for enhanced signal detection
- Customizable resolution measurements

Usage:
1. Upload a test specimen image
2. Select the expected diffraction pattern (graphene/gold/ice)
3. Adjust the region size to analyze
4. Click points in the FFT to measure distances
5. Use auto-search to find the best pixel size match

The tool will display:
- Original image with selected region
- FFT with resolution circles
- 1D radial average plot
- Calculated pixel size (Angstroms/pixel)
"""
import argparse

def print_help():
    """Print usage instructions and help information."""
    help_text = """
Magnification Calibration Tool
---------------------------

Usage:
    Run the Shiny app and follow the web interface.
    
Input Files:
    - Image formats: PNG, TIFF
    - MRC files from microscopes
    
Key Parameters:
    Apix: Pixel size in Angstroms/pixel (0.01-6.0)
    Region: Size of FFT analysis region (1-100%)
    Resolution circles:
        - Graphene: 2.13 Å
        - Gold: 2.355 Å 
        - Ice: 3.661 Å
        - Custom: User-defined resolution
        
Analysis Features:
    - Interactive FFT region selection
    - Resolution circle overlay
    - Automatic pixel size detection
    - Radial averaging
    - Click-to-measure distances
    
Output:
    - Processed FFT image
    - Radial intensity profile
    - Calculated pixel size
    """
    print(help_text)
    
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("upload", "Upload an image of test specimens (e.g., graphene)(.mrc,.tiff,.png)", accept=["image/*", ".mrc", ".tif", ".png"]),
        ui.input_select("resolution_type", "Resolution Type", 
                      choices=["Graphene (2.13 Å)", "Gold (2.355 Å)", "Ice (3.661 Å)", "Custom"], 
                      selected="Graphene (2.13 Å)"),
        ui.panel_conditional(
            "input.resolution_type == 'Custom'",
            ui.div(
                {"style": "display: flex; align-items: center;"},
                ui.input_numeric("custom_resolution", "Custom Res (Å):", value=3.0, min=0.1, max=10.0, step=0.01, width="80px"),
            ),
        ),
        ui.input_select("label_mode", "Label", 
                      choices=["Resolution Ring", "Lattice Point"], 
                      selected="Resolution Ring"),
        ui.div(
            {"style": "padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px; display: flex; flex-direction: column; gap: 5px;"},
            ui.div(
                {"style": "flex: 1;"},
                ui.input_slider("apix_slider", "Apix (Å/px)", min=0.01, max=2.0, value=1.0, step=0.001),
            ),
            ui.div(
                {"style": "display: flex; justify-content: flex-start; align-items: bottom; gap: 5px; margin-top: 5px; width: 100%;"},
                ui.input_text("apix_exact_str", None, value="1.0", width="70px"),
                ui.input_action_button("apix_set_btn", ui.tags.span("Set", style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%;"), class_="btn-primary", style="height: 38px; display: flex; align-items: center;", width="50px"),
            ),
        ),

        ui.div(
            {"style": "display: none;"},  # Hidden div for data persistence
            ui.output_text("lattice_points_data"),
        ),
        title=ui.h2("Magnification Calibration", style="font-size: 36px; font-weight: bold; padding: 15px;"),
        open="open",
        width="400px",
        min_width="250px",
        max_width="500px",
        resize=True,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Original Image"),
            ui.div(
                {"style": "display: flex;"},
                ui.div(
                    {"class": "image-output"},
                    ui.output_image("image_display", click=True, height="360px"),
                ),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; margin-left: 10px; width: 200px;"},
                    ui.input_slider("zoom1", "Zoom (%)", min=50, max=300, value=100),
                    ui.input_slider("rg_size", "Region size (%)", min=1, max=100, value=30),
                    ui.input_slider("contrast1", "Range (±σ)", min=0.1, max=5.0, value=2.0, step=0.1),
                ),
            ),
            ui.div(
                {"class": "card-footer"},
                "Green denotes the region selected for FFT. Click to select region.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("FFT Analysis"),
            ui.div(
                {"style": "display: flex;"},
                ui.div(
                    {"class": "image-output"},
                    ui.output_image("fft_with_circle", click=True),
                ),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; margin-left: 10px; width: 200px;"},
                    ui.input_slider("zoom2", "Zoom (%)", min=50, max=300, value=100),
                    ui.input_slider("contrast", "Range (±σ)", min=0.1, max=5.0, value=2.0, step=0.1),
                    ui.input_action_button("clear_markers", "Clear", class_="btn-secondary"),
                    ui.input_action_button("fit_markers", "Fit", class_="btn-secondary"),
                    ui.input_action_button("estimate_tilt", "Estimate Tilt", class_="btn-secondary"),
                    ui.output_text("tilt_output"),
                ),
            ),
            ui.div(
                {"class": "card-footer"},
                "Click to mark points based on selected Label mode in sidebar.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("1D FFT Radial Profile"),
            ui.div(
                {"style": "display: flex;"},
                ui.output_plot("fft_1d_plot", click=True, dblclick=True, brush=True, width="600px", height="400px"),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; margin-left: 10px; width: 200px;"},
                    ui.input_checkbox("log_y", "Log Scale", value=False),
                    ui.input_checkbox("use_mean_profile", "Use Average Profile", value=False),
                    ui.input_checkbox("smooth", "Smooth Signal", value=False),
                    ui.input_checkbox("detrend", "Detrend Signal", value=False),
                    ui.div(
                        {"style": "margin-bottom: 10px;"},
                        ui.panel_conditional(
                            "input.smooth",
                            ui.input_slider("window_size", "Window Size", min=1, max=11, value=3, step=2),
                        ),
                    ),
                    ui.input_action_button("reset_zoom", "Reset Zoom"),
                    ui.input_action_button("estimate_tilt_1d", "Estimate Tilt", class_="btn-secondary"),
                    ui.output_text("tilt_1d_output"),
                ),
            ),
            ui.div(
                {"class": "card-footer", "style": "justify-content: flex-start;"},
                "Radial Max of the 2D FFT. Drag to zoom, double-click to reset."
            ),
            full_screen=True,
        ),
        col_widths=[6, 6, 12],
    ),

    fillable=True,
)
# Add custom CSS for layout
app_ui = ui.tags.div(
    ui.tags.style("""
        /* Image container styles */
        .image-output {
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 10px;
            margin-bottom: 10px;
            width: 100%;
            min-height: 300px;
            flex: 1;
            /* Ensure scrollbars are always visible and not hidden */
            scrollbar-width: auto;
            scrollbar-color: rgba(0, 0, 0, 0.3) transparent;
        }
        .image-output::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        .image-output::-webkit-scrollbar-track {
            background: transparent;
        }
        .image-output::-webkit-scrollbar-thumb {
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            border: 2px solid transparent;
        }
        .image-output img {
            height: auto;
            width: auto;
            max-width: none;
            max-height: none;
            /* Add margin to ensure scrollbar is not covered */
            margin-bottom: 12px;
        }
        /* Footer styles */
        .card-footer {
            height: 40px;
            padding: 8px;
            background-color: rgba(0, 0, 0, 0.03);
            border-top: 1px solid rgba(0, 0, 0, 0.125);
            display: flex;
            align-items: center;
            flex-shrink: 0;
            margin-top: 0;
            width: 100%;
        }
        .sidebar > h2, 
        .sidebar-title,
        .shiny-sidebar-title {
            font-size: 36px !important;
            font-weight: bold !important;
            padding: 15px !important;
        }
    """),
    app_ui
)
size = 360

# ---------- Helper Functions ----------
def fit_ellipse_fixed_center(points, center=(0, 0)):
    """
    Fit an ellipse to points with fixed center using least squares.
    
    Args:
        points: List of (x, y) tuples
        center: (cx, cy) center coordinates
        
    Returns:
        (a, b, theta): semi-major axis, semi-minor axis, rotation angle (radians)
    """
    import numpy as np
    from scipy.optimize import least_squares
    
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

# ---------- Server ----------
def server(input: Inputs, output: Outputs, session: Session):
    # Central reactive state for FFT panel
    fft_state = reactive.Value({
        'mode': 'Resolution Ring',
        'resolution_radius': None,
        'resolution_click_x': None,
        'resolution_click_y': None,
        'lattice_points': [],
        'ellipse_params': None,
        'tilt_info': None,
        'zoom_factor': 1.0,
        'plot_1d_markers': [],  # List of (x, y) tuples for 1D plot markers in Lattice Point mode
        'tilt_info_1d': None  # (smaller_x, larger_x, tilt_angle, apix_value) for 1D tilt estimation
    })
    
    # Add reactive values for region center and FFT click position
    region_center = reactive.Value({
        'x': None,
        'y': None
    })

    # --- Single source of truth for apix ---
    apix_master = reactive.Value(1.0)



    # Add reactive value to cache the base FFT image
    cached_fft_image = reactive.Value(None)

    # Add reactive value to cache the FFT image without resolution circles
    cached_fft_image_no_circles = reactive.Value(None)

    # Add plot zoom state
    plot_zoom = reactive.Value({
        'x_range': None,
        'y_range': None
    })

    # Add reactive values for raw data and region
    raw_image_data = reactive.Value({
        'img': None,
        'data': None
    })



    # Initialize Fit button state
    @reactive.Effect
    def _():
        """Initialize Fit button state."""
        is_disabled = input.label_mode() != "Lattice Point"
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt_1d", disabled=is_disabled, session=session)

    # --- All events update apix_master ---
    @reactive.Effect
    @reactive.event(input.apix_slider)
    def _():
        apix_master.set(input.apix_slider())
        # Clear 1D plot clicked position when apix changes from slider
        #plot_1d_click_pos.set({'x': None, 'y': None})

    @reactive.Effect
    @reactive.event(input.apix_set_btn)
    def _():
        try:
            val = float(input.apix_exact_str())
            if 0.001 <= val <= 6.0:
                #apix_master.set(val)
                ui.update_slider("apix_slider", value=val, session=session)
                ui.update_text("apix_exact_str", value=str(round(val, 3)), session=session)
                # Clear 1D plot clicked position when apix changes from Set button
                #plot_1d_click_pos.set({'x': None, 'y': None})
        except Exception:
            pass



    def get_first_checked_resolution():
        """Return the first checked resolution value or None if none are checked."""
        if input.resolution_type() == "Graphene (2.13 Å)":
            return 2.13, "red"
        elif input.resolution_type() == "Gold (2.355 Å)":
            return 2.355, "orange"
        elif input.resolution_type() == "Ice (3.661 Å)":
            return 3.661, "blue"
        elif input.resolution_type() == "Custom":
            return input.custom_resolution(), "green"
        return None, None

    # --- Mouse Click Effects ---
    @reactive.Effect
    @reactive.event(input.image_display_click)
    def _():
        click_data = input.image_display_click()
        if click_data is not None:
            # Scale the coordinates based on the zoom level
            zoom_factor = input.zoom1() / 100
            region_center.set({
                'x': int(click_data['x'] / zoom_factor),
                'y': int(click_data['y'] / zoom_factor)
            })

    @reactive.Effect
    @reactive.event(input.fft_with_circle_click)
    def _():
        click_data = input.fft_with_circle_click()
        if click_data is not None:
            zoom_factor = input.zoom2() / 100
            x = click_data['x'] / zoom_factor
            y = click_data['y'] / zoom_factor
            
            current_state = fft_state.get()
            
            if input.label_mode() == "Resolution Ring":
                # Resolution Ring mode: set crosshair and calculate radius
                resolution, color = get_first_checked_resolution()
                if resolution is None:
                    return
                
                # Calculate distance from center in pixels
                center_x = size / 2
                center_y = size / 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Update state
                new_state = current_state.copy()
                new_state['mode'] = 'Resolution Ring'
                new_state['resolution_radius'] = distance
                new_state['resolution_click_x'] = x
                new_state['resolution_click_y'] = y
                new_state['lattice_points'] = []  # Clear lattice points
                fft_state.set(new_state)
                
                # Update apix if distance is valid
                if distance > 0:
                    new_apix = (distance * resolution) / size
                    if 0.01 <= new_apix <= 6.0:
                        new_apix = round(new_apix, 3)
                        ui.update_slider("apix_slider", value=new_apix, session=session)
                        ui.update_text("apix_exact_str", value=str(new_apix), session=session)
            
            elif input.label_mode() == "Lattice Point":
                # Lattice Point mode: add point to list
                new_state = current_state.copy()
                new_state['mode'] = 'Lattice Point'
                new_state['lattice_points'].append((x, y))
                fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.clear_markers)
    def _():
        """Clear all markers based on current mode."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        
        if current_state['mode'] == 'Resolution Ring':
            # Clear resolution ring markers
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
        elif current_state['mode'] == 'Lattice Point':
            # Clear lattice points, ellipse, tilt info, and 1D plot markers
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        
        fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.fit_markers)
    def _():
        """Handle Fit button click to fit ellipse to lattice points."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        points = list(current_state['lattice_points'])
        if len(points) == 0:
            return
            
        # Compute image center
        cx, cy = size / 2, size / 2
        
        # If fewer than 3 points, create mirrored and jittered points
        if len(points) < 3:
            working_points = points.copy()
            for x, y in points:
                # Mirror through center
                mx, my = 2 * cx - x, 2 * cy - y
                # Add small Gaussian noise
                jittered_x = mx + np.random.normal(scale=1.0)
                jittered_y = my + np.random.normal(scale=1.0)
                working_points.append((jittered_x, jittered_y))
        else:
            working_points = points
        
        # Fit ellipse
        try:
            a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
            
            # Update state with ellipse parameters
            new_state = current_state.copy()
            new_state['ellipse_params'] = (a, b, theta)
            fft_state.set(new_state)
        except Exception as e:
            print(f"Ellipse fitting failed: {e}")

    @reactive.Effect
    @reactive.event(input.estimate_tilt)
    def _():
        """Handle Estimate Tilt button click to compute tilt angle from ellipse."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        # Ensure an ellipse is fitted
        if current_state['ellipse_params'] is None:
            points = list(current_state['lattice_points'])
            if len(points) == 0:
                return
                
            # Compute image center
            cx, cy = size / 2, size / 2
            
            # If fewer than 3 points, create mirrored and jittered points
            if len(points) < 3:
                working_points = points.copy()
                for x, y in points:
                    # Mirror through center
                    mx, my = 2 * cx - x, 2 * cy - y
                    # Add small Gaussian noise
                    jittered_x = mx + np.random.normal(scale=1.0)
                    jittered_y = my + np.random.normal(scale=1.0)
                    working_points.append((jittered_x, jittered_y))
            else:
                working_points = points
            
            # Fit ellipse
            try:
                a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
                
                # Update state with ellipse parameters
                new_state = current_state.copy()
                new_state['ellipse_params'] = (a, b, theta)
                fft_state.set(new_state)
                current_state = new_state
            except Exception as e:
                print(f"Ellipse fitting failed: {e}")
                return
        
        # Compute tilt from ellipse parameters
        a, b, _ = current_state['ellipse_params']
        # Always use the smaller value as the x (horizontal) axis
        if a < b:
            small_axis = a  # x-axis
            large_axis = b  # y-axis
        else:
            small_axis = b  # x-axis
            large_axis = a  # y-axis
        import math
        tilt_angle = math.acos(small_axis / large_axis)
        
        # Calculate apix from small x-axis (horizontal)
        resolution, _ = get_first_checked_resolution()
        if resolution is not None and small_axis > 0:
            # Use the same formula as resolution ring mode: apix = (distance * resolution) / size
            new_apix = (small_axis * resolution) / size
            if 0.01 <= new_apix <= 6.0:
                new_apix = round(new_apix, 3)
                # Update UI controls
                ui.update_slider("apix_slider", value=new_apix, session=session)
                ui.update_text("apix_exact_str", value=str(new_apix), session=session)
        
        # Update state with tilt info
        new_state = current_state.copy()
        new_state['tilt_info'] = (small_axis, large_axis, tilt_angle)
        fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.fft_1d_plot_click)
    def _():
        """Handle clicks on the 1D FFT plot based on current mode."""
        click_data = input.fft_1d_plot_click()
        if click_data is not None:
            current_state = fft_state.get()
            
            if current_state['mode'] == 'Resolution Ring':
                # Resolution Ring mode: update apix based on clicked position (existing behavior)
                # Step 1: Get the x location (radius in region pixels) from the click
                region_radius = click_data['x']
                
                # Step 2: Convert region radius to full FFT coordinates
                # The 1D plot uses region coordinates, but 2D FFT uses full image coordinates
                region = get_current_region()
                if region is None:
                    return
                    
                # Get the region size and full image size
                region_size = region.size[0]  # This is the cropped region size
                full_fft_size = size  # This is the full FFT image size (360)
                
                # Scale the radius from region coordinates to full FFT coordinates
                fft_radius = region_radius * (full_fft_size / region_size)
                
                # Get the selected resolution and color
                resolution, color = get_first_checked_resolution()
                
                # Step 3: Get the selected resolution
                if resolution is None or fft_radius == 0:
                    return
                
                # Step 4: Calculate new apix using the same formula as 2D FFT click
                # new_apix = (distance * resolution) / size
                # where distance = fft_radius (scaled to full FFT coordinates)
                new_apix = (fft_radius * resolution) / full_fft_size
                
                # Step 5: Update the apix slider if within bounds
                if 0.01 <= new_apix <= 6.0:
                    new_apix = round(new_apix, 3)
                    # Update UI controls directly to avoid double redraws
                    ui.update_slider("apix_slider", value=new_apix, session=session)
                    ui.update_text("apix_exact_str", value=str(new_apix), session=session)
            
            elif current_state['mode'] == 'Lattice Point':
                # Lattice Point mode: add vertical line marker (up to 2)
                x = click_data['x']
                y = click_data['y']
                
                # Add new marker
                new_markers = current_state['plot_1d_markers'].copy()
                new_markers.append((x, y))
                
                # Keep only the 2 most recent markers
                if len(new_markers) > 2:
                    new_markers = new_markers[-2:]
                
                # Update state
                new_state = current_state.copy()
                new_state['plot_1d_markers'] = new_markers
                fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.fft_1d_plot_brush)
    def _():
        brush_data = input.fft_1d_plot_brush()
        if brush_data is not None:
            plot_zoom.set({
                'x_range': (brush_data['xmin'], brush_data['xmax']),
                'y_range': (brush_data['ymin'], brush_data['ymax'])
            })

    @reactive.Effect
    @reactive.event(input.fft_1d_plot_dblclick)
    def _():
        if input.fft_1d_plot_dblclick() is not None:
            plot_zoom.set({'x_range': None, 'y_range': None})

    @reactive.Effect
    @reactive.event(input.reset_zoom)
    async def _():
        # Clear the brush selection first by sending a custom message
        await session.send_custom_message("shiny:brushed", {"plot_id": "fft_1d_plot", "coords": None})
        # Then reset the zoom state
        plot_zoom.set({'x_range': None, 'y_range': None})

    @reactive.Calc
    def image_path():
        file = input.upload()
        if not file:
            return None
        return Path(file[0]["datapath"])

    def save_temp_image(img: Image.Image) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return tmp.name

    def normalize(magnitude, contrast=2.0):
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
        """Normalize image data using mean and standard deviation.
        
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
        """Read an MRC file and convert it to a PIL Image.
        
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
        """Load an image file or MRC file and return as PIL Image and raw data.
        
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

    @reactive.Calc
    def get_apix():
        return apix_master.get()

    @reactive.Calc
    def get_apix_from_distance():
        """Calculate the apix value from a given distance in pixels and current resolution.
        
        Returns:
            A function that takes distance in pixels and returns the corresponding apix value.
        """
        resolution, _ = get_first_checked_resolution()
        if resolution is None:
            return lambda distance: None
        
        def calculate_apix(distance_pixels):
            """Calculate apix from distance in pixels.
            
            Args:
                distance_pixels: Distance from center in pixels
                
            Returns:
                Apix value in Å/pixel, or None if invalid
            """
            if distance_pixels <= 0:
                return None
            return (distance_pixels * resolution) / size
        
        return calculate_apix

    @reactive.Calc
    def get_distance_from_apix():
        """Calculate the distance in pixels from a given apix value and current resolution.
        
        Returns:
            A function that takes apix value and returns the corresponding distance in pixels.
        """
        resolution, _ = get_first_checked_resolution()
        if resolution is None:
            return lambda apix: None
        
        def calculate_distance(apix_value):
            """Calculate distance in pixels from apix value.
            
            Args:
                apix_value: Apix value in Å/pixel
                
            Returns:
                Distance from center in pixels, or None if invalid
            """
            if apix_value <= 0:
                return None
            return (apix_value * size) / resolution
        
        return calculate_distance

    def get_processed_image_for_display():
        """Get the contrast-adjusted image from raw data for display only."""
        if raw_image_data.get()['data'] is None:
            return None
            
        # Apply contrast normalization to raw data
        normalized_data = normalize_image(raw_image_data.get()['data'], input.contrast1())
        img = Image.fromarray(normalized_data)
        return img.convert("RGB")

    def get_base_image():
        """Get the base image without contrast adjustment for FFT calculations."""
        if raw_image_data.get()['data'] is None:
            return None
            
        return raw_image_data.get()['img'].convert("RGB")

    @reactive.Effect
    @reactive.event(input.upload)
    def _():
        """Update raw image data when a new file is uploaded."""
        path = image_path()
        if not path or not path.exists():
            raw_image_data.set({'img': None, 'data': None})
            return
            
        img, raw_data = load_image(path)
        raw_image_data.set({
            'img': img,
            'data': raw_data
        })

    @reactive.Effect
    @reactive.event(input.upload, input.rg_size, input.contrast, input.zoom1, input.image_display_click)
    def _():
        """Update cached FFT images when region or contrast changes."""
        region = get_current_region()
        if region is not None:
            # Generate base FFT image without circles
            fft_img = compute_fft_image_region(region, input.contrast())
            cached_fft_image.set(fft_img)
            
            # Generate FFT image without resolution circles for Lattice Point mode
            fft_img_no_circles = fft_img.copy()
            cached_fft_image_no_circles.set(fft_img_no_circles)
        else:
            cached_fft_image.set(None)
            cached_fft_image_no_circles.set(None)

    @reactive.Effect
    @reactive.event(fft_state)
    def _():
        """Force FFT image redraw when fft_state changes (including zoom)."""
        # This effect ensures the FFT image redraws when markers or zoom changes
        pass

    @output
    @render.image
    def image_display():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)
        
        # Get contrast-adjusted image
        img = get_processed_image_for_display()
        
        # Apply zoom
        base_size = size
        zoom_factor = input.zoom1() / 100
        new_size = int(base_size * zoom_factor)
        img = img.resize((new_size, new_size))
        
        # Calculate region size
        rg_sz = (input.rg_size()*img.size[0]/100, input.rg_size()*img.size[1]/100)
        draw = ImageDraw.Draw(img)

        # Use clicked position if available, otherwise use center
        if region_center.get()['x'] is not None:
            # Scale the coordinates based on the zoom level
            center = (
                int(region_center.get()['x'] * zoom_factor),
                int(region_center.get()['y'] * zoom_factor)
            )
        else:
            center = (img.size[0] // 2, img.size[1] // 2)

        x1, y1 = center[0] - rg_sz[0] // 2, center[1] - rg_sz[1] // 2
        x2, y2 = center[0] + rg_sz[0] // 2, center[1] + rg_sz[1] // 2
        draw.rectangle([(x1, y1), (x2, y2)], outline='green', width=2)
        
        return {"src": save_temp_image(img)}

    def get_current_region():
        """Get the current region for FFT calculation."""
        if raw_image_data.get()['data'] is None:
            return None
            
        # Use base image without contrast adjustment for FFT
        img = get_base_image()
        if img is None:
            return None
            
        rg_sz = input.rg_size()*img.size[0]/100

        # Use clicked position if available, otherwise use center
        if region_center.get()['x'] is not None:
            center = (region_center.get()['x'], region_center.get()['y'])
        else:
            center = (img.size[0] // 2, img.size[0] // 2)

        x1, y1 = center[0] - rg_sz // 2, center[1] - rg_sz // 2
        x2, y2 = center[0] + rg_sz // 2, center[1] + rg_sz // 2
        region = img.crop((x1, y1, x2, y2))
        return region

    @output
    @render.image
    def fft_with_circle():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)
        
        # Get current FFT state
        state = fft_state.get()
        
        # Choose appropriate cached image based on mode
        if state['mode'] == 'Lattice Point':
            req(cached_fft_image_no_circles.get() is not None)
            fft_img = cached_fft_image_no_circles.get().copy()
        else:
            req(cached_fft_image.get() is not None)
            fft_img = cached_fft_image.get().copy()
        
        # Apply zoom
        base_size = size
        zoom_factor = input.zoom2() / 100
        new_size = int(base_size * zoom_factor)
        fft_img = fft_img.resize((new_size, new_size))
        draw = ImageDraw.Draw(fft_img)
        center = (new_size // 2, new_size // 2)

        def resolution_to_radius(res_angstrom):
            return (fft_img.size[0] * get_apix()) / res_angstrom

        # Draw resolution circles only in Resolution Ring mode
        if state['mode'] == 'Resolution Ring':
            if input.resolution_type() == "Graphene (2.13 Å)":
                r = resolution_to_radius(2.13)
                draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="red", width=2)
            if input.resolution_type() == "Gold (2.355 Å)":
                r = resolution_to_radius(2.355)
                draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="orange", width=2)
            if input.resolution_type() == "Ice (3.661 Å)":
                r = resolution_to_radius(3.661)
                draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="blue", width=2)
            if input.resolution_type() == "Custom":
                r = resolution_to_radius(input.custom_resolution())
                draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="green", width=2)

        # Draw markers based on current mode
        if state['mode'] == 'Resolution Ring' and state['resolution_radius'] is not None:
            # Draw crosshair at the clicked point
            x = state['resolution_click_x']
            y = state['resolution_click_y']
            if x is not None and y is not None:
                # Apply coordinate transformation for zoom
                cx, cy = size / 2, size / 2  # Original image center
                disp_x = center[0] + (x - cx) * state['zoom_factor']
                disp_y = center[1] + (y - cy) * state['zoom_factor']
                
                marker_size = 10
                color = "yellow"  # Default color for crosshair
                draw.line([(disp_x - marker_size, disp_y), (disp_x + marker_size, disp_y)], fill=color, width=2)
                draw.line([(disp_x, disp_y - marker_size), (disp_x, disp_y + marker_size)], fill=color, width=2)
                
                # Draw apix label background and text
                current_apix_str = f"Apix: {get_apix():.3f} Å/px"
                # Determine color based on selected resolution
                if input.resolution_type() == "Graphene (2.13 Å)":
                    apix_color = "red"
                elif input.resolution_type() == "Gold (2.355 Å)":
                    apix_color = "orange"
                elif input.resolution_type() == "Ice (3.661 Å)":
                    apix_color = "blue"
                elif input.resolution_type() == "Custom":
                    apix_color = "green"
                else:
                    apix_color = "black"
                try:
                    font = ImageFont.truetype("Arial", 16)
                except OSError:
                    font = ImageFont.load_default()
                text_bbox = draw.textbbox((0, 0), current_apix_str, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                padding = 10
                draw.rectangle((padding, padding, padding + text_width + 10, padding + text_height + 10), 
                              fill=(255, 255, 255, 180))
                draw.text((padding + 5, padding + 5), current_apix_str, fill=apix_color, font=font)
        
        elif state['mode'] == 'Lattice Point':
            # Draw lattice points as hollow circles
            for x, y in state['lattice_points']:
                # Apply coordinate transformation for zoom
                cx, cy = size / 2, size / 2  # Original image center
                disp_x = center[0] + (x - cx) * state['zoom_factor']
                disp_y = center[1] + (y - cy) * state['zoom_factor']
                
                circle_radius = 8
                draw.ellipse(
                    (disp_x - circle_radius, disp_y - circle_radius, disp_x + circle_radius, disp_y + circle_radius),
                    outline="green",
                    width=2
                )
            
            # Draw fitted ellipse if available
            if state['ellipse_params'] is not None:
                a, b, theta = state['ellipse_params']
                # Apply zoom scaling to ellipse parameters
                a_scaled = a * state['zoom_factor']
                b_scaled = b * state['zoom_factor']
                
                # For rotated ellipse, we need to approximate with multiple line segments
                # since PIL doesn't support rotated ellipses directly
                import math
                num_points = 50
                points = []
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Parametric ellipse equation
                    x_ellipse = a_scaled * math.cos(angle)
                    y_ellipse = b_scaled * math.sin(angle)
                    # Rotate
                    x_rot = x_ellipse * math.cos(theta) - y_ellipse * math.sin(theta)
                    y_rot = x_ellipse * math.sin(theta) + y_ellipse * math.cos(theta)
                    # Translate to center
                    x_final = center[0] + x_rot
                    y_final = center[1] + y_rot
                    points.append((x_final, y_final))
                
                # Draw ellipse as connected line segments
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    draw.line([start, end], fill="red", width=2)

        return {"src": save_temp_image(fft_img), "click": True}

    @output
    @render.plot
    def fft_1d_plot():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)

        region = get_current_region()
        if region is None:
            return None

        # Compute FFT and get power spectrum
        arr = np.array(region.convert("L")).astype(np.float32)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        pwr = np.abs(fshift)  # Power spectrum

        if input.use_mean_profile():
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

        # Set x-axis limits to 0.25 to 0.75 of the largest radius
        x_min = int(len(pwr_1d) * 0.25)
        x_max = int(len(pwr_1d) * 0.75)
        mask = (radius_pixels >= x_min) & (radius_pixels <= x_max)

        # Create figure with adjusted margins
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.95)
        ax = fig.add_subplot(111)
        
        # Plot data
        y_data = pwr_1d[mask]
        
        # Ensure we have valid data
        if len(y_data) == 0 or np.all(y_data == 0):
            # Fallback: create a simple plot with some data
            y_data = np.ones_like(radius_pixels[mask])
        
        if input.log_y():
            y_data = np.log1p(y_data)  # log1p is safe for positive values
            ax.set_ylabel("Log(FFT intensity)")
        else:
            ax.set_ylabel("FFT intensity")

        # Apply smoothing to y_data using a moving average
        if input.smooth():
            window_size = input.window_size()
            kernel = np.ones(window_size) / window_size
            # Determine padding amount for mode='same'
            pad_amount = (len(kernel) - 1) // 2
            
            # Pad the signal with 'reflect' mode
            padded_y_data = np.pad(y_data, pad_width=pad_amount, mode='reflect')
            
            # Perform convolution with the padded signal
            y_data = np.convolve(padded_y_data, kernel, mode='valid')
            
            y_data = y_data - y_data.min()
        # Detrend the signal by fitting and subtracting a linear baseline
        if input.detrend():
            
            # Fit a first-degree polynomial to get trend
            m, b = np.polyfit(radius_pixels[mask], y_data, 1)
            # Compute and subtract baseline
            baseline = m * radius_pixels[mask] + b
            y_data = y_data - baseline
            # Shift back to positive values
            y_data = y_data - y_data.min()

        ax.plot(radius_pixels[mask], y_data, label=profile_label)
        
        # Immediately set y-axis limits based on the plotted data
        if not input.log_y():
            y_min = y_data.min()
            y_max = y_data.max()
            if y_max > y_min:
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
                else:
                    ax.set_ylim(y_min, y_max * 1.1)
            else:
                if y_max > 0:
                    ax.set_ylim(y_max * 0.9, y_max * 1.1)
                else:
                    ax.set_ylim(-0.1, 0.1)
        
        # Set axis limits based on zoom state or defaults
        zoom = plot_zoom.get()
        if zoom['x_range'] is not None and zoom['y_range'] is not None:
            xlim = zoom['x_range']
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(x_min, x_max)
            xlim = (x_min, x_max)
            
        if zoom['y_range'] is not None:
            ax.set_ylim(zoom['y_range'])
        # Note: y-axis limits are already set above for non-zoom case

        ax.set_xlabel("Radius (pixels)")
        ax.set_title("1D FFT Radial Profile")
        ax.grid(True)

        # Get current FFT state for mode-based rendering
        state = fft_state.get()
        
        # Draw vertical lines based on current mode
        if state['mode'] == 'Resolution Ring':
            # Resolution Ring mode: draw single vertical line based on resolution type
            if input.resolution_type() == "Graphene (2.13 Å)":
                radius_213 = (arr.shape[0] * get_apix()) / 2.13
                if x_min <= radius_213 <= x_max:
                    ax.axvline(radius_213, color="red", linestyle="--", label="Graphene at 2.13 Å")
            elif input.resolution_type() == "Gold (2.355 Å)":
                radius_235 = (arr.shape[0] * get_apix()) / 2.355
                if x_min <= radius_235 <= x_max:
                    ax.axvline(radius_235, color="orange", linestyle="--", label="Gold at 2.355 Å")
            elif input.resolution_type() == "Ice (3.661 Å)":
                radius_366 = (arr.shape[0] * get_apix()) / 3.661
                if x_min <= radius_366 <= x_max:
                    ax.axvline(radius_366, color="blue", linestyle="--", label="Ice at 3.661 Å")
            elif input.resolution_type() == "Custom":
                radius_custom = (arr.shape[0] * get_apix()) / input.custom_resolution()
                if x_min <= radius_custom <= x_max:
                    ax.axvline(radius_custom, color="green", linestyle="--", 
                              label=f"Custom at {input.custom_resolution():.2f} Å")
        
        elif state['mode'] == 'Lattice Point':
            # Lattice Point mode: draw up to 2 green vertical lines from clicked positions
            for i, (x, y) in enumerate(state['plot_1d_markers']):
                if x_min <= x <= x_max:
                    ax.axvline(x, color="green", linestyle="--", 
                              label=f"Marker {i+1}" if i < 2 else None)

        ax.legend(loc="lower left", fontsize="small")
        
        # Force matplotlib to use the set limits
        ax.autoscale_view()
        plt.tight_layout()

        return fig



    @output
    @render.text
    def lattice_points_data():
        """Hidden output to expose lattice points data for persistence."""
        state = fft_state.get()
        if state['mode'] == 'Lattice Point' and state['lattice_points']:
            # Return lattice points as JSON-like string for easy parsing
            points_str = ";".join([f"{x},{y}" for x, y in state['lattice_points']])
            return f"Lattice Points: {points_str}"
        return "Lattice Points: None"

    @output
    @render.text
    def tilt_output():
        """Display tilt estimation results."""
        state = fft_state.get()
        if state['tilt_info'] is not None:
            small_axis, large_axis, tilt_angle = state['tilt_info']
            import math
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            # Calculate apix from small axis
            resolution, _ = get_first_checked_resolution()
            apix_str = ""
            if resolution is not None and small_axis > 0:
                calculated_apix = (small_axis * resolution) / size
                if 0.01 <= calculated_apix <= 6.0:
                    apix_str = f", Apix: {calculated_apix:.3f} Å/px"
            
            return (f"Small axis: {small_axis:.2f}, "
                   f"Large axis: {large_axis:.2f}, "
                   f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°"
                   f"{apix_str}")
        return ""

    @output
    @render.text
    def tilt_1d_output():
        """Display 1D tilt estimation results."""
        state = fft_state.get()
        if state['tilt_info_1d'] is not None:
            smaller_x, larger_x, tilt_angle, apix_value = state['tilt_info_1d']
            import math
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            return (f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°, "
                   f"Apix at selected frequency: {apix_value:.3f} Å/px")
        return ""

    # --- All UI controls and plots react to apix_master ---
    @reactive.Effect
    @reactive.event(apix_master)
    def _():
        val = apix_master.get()
        
        # Only update if the change is significant (>= 0.001)
        # This prevents unnecessary updates for tiny changes
        current_slider_val = input.apix_slider()
        if abs(val - current_slider_val) < 0.001:
            return
            
        # Update UI controls
        # ui.update_slider("apix_slider", value=val, session=session)
        # ui.update_text("apix_exact_str", value=str(round(val, 3)), session=session)
        
        # Update FFT circle positions by clearing click positions
        # This will force the circles to use calculated positions instead of clicked positions
        current_state = fft_state.get()
        if current_state['mode'] == 'Resolution Ring':
            new_state = current_state.copy()
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
            fft_state.set(new_state)
        
        # Clear 1D plot clicked position when resolution type changes
        #plot_1d_click_pos.set({'x': None, 'y': None, 'color': None})

    @reactive.Effect
    @reactive.event(input.label_mode)
    def _():
        """Handle mode switching and clear appropriate markers."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        
        if input.label_mode() == "Resolution Ring":
            # Switching to Resolution Ring: clear lattice points, ellipse, tilt info, and 1D plot markers
            new_state['mode'] = 'Resolution Ring'
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        elif input.label_mode() == "Lattice Point":
            # Switching to Lattice Point: clear resolution radius, click coordinates, tilt info, and 1D plot markers
            new_state['mode'] = 'Lattice Point'
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            new_state['plot_1d_markers'] = []
            new_state['tilt_info_1d'] = None
        
        fft_state.set(new_state)
        
        # Update Fit button state
        is_disabled = input.label_mode() != "Lattice Point"
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt_1d", disabled=is_disabled, session=session)

    @reactive.Effect
    @reactive.event(input.zoom2)
    def _():
        """Update zoom factor in fft_state when zoom slider changes."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        new_state['zoom_factor'] = input.zoom2() / 100.0
        fft_state.set(new_state)

    @reactive.Effect
    @reactive.event(input.estimate_tilt_1d)
    def _():
        """Handle 1D Estimate Tilt button click to compute tilt angle from 1D plot markers."""
        current_state = fft_state.get()
        if current_state['mode'] != 'Lattice Point':
            return
            
        # Check if we have exactly 2 markers
        if len(current_state['plot_1d_markers']) != 2:
            return
            
        # Get the two x positions
        x1 = current_state['plot_1d_markers'][0][0]
        x2 = current_state['plot_1d_markers'][1][0]
        
        # Identify smaller and larger values
        smaller_x, larger_x = sorted([x1, x2])
        
        # Compute tilt angle as arccos(smaller/larger)
        import math
        if larger_x > 0:
            tilt_angle = math.acos(smaller_x / larger_x)
        else:
            return
        
        # Convert larger marker position to apix using resolution-to-apix mapping
        resolution, _ = get_first_checked_resolution()
        if resolution is not None:
            # Convert from region coordinates to full FFT coordinates
            region = get_current_region()
            if region is not None:
                region_size = region.size[0]
                full_fft_size = size
                fft_radius = smaller_x * (full_fft_size / region_size)
                
                # Calculate apix using the same formula as other modes
                apix_value = (fft_radius * resolution) / full_fft_size
                
                if 0.01 <= apix_value <= 6.0:
                    # Update state with 1D tilt info
                    new_state = current_state.copy()
                    new_state['tilt_info_1d'] = (smaller_x, larger_x, tilt_angle, apix_value)
                    fft_state.set(new_state)

app = App(app_ui, server)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Magnification Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help manually to use our custom format
    )
    parser.add_argument('--help', '-h', action='store_true', 
                       help='Show detailed help message')

    args = parser.parse_args()
    
    if args.help:
        print_help()
    else:
        app.run()
