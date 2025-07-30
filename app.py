from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_plotly, render_widget
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import math
import plotly.graph_objects as go
from plotly.graph_objects import FigureWidget
from compute import (
    fit_ellipse_fixed_center,
    normalize,
    normalize_image,
    read_mrc_as_image,
    load_image,
    fft_image_with_matplotlib,
    compute_fft_image_region,
    compute_average_fft,
    calculate_apix_from_distance,
    calculate_distance_from_apix,
    calculate_tilt_angle,
    get_resolution_info,
    compute_fft_1d_data,
    resolution_to_radius,
    create_fft_1d_plotly_figure,
    get_image,
    plot_image,
    get_image_with_binning,
    extract_region_from_original,
    extract_region_no_binning,
    bin_image
)
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
        ui.div(
            {"style": "padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px;"},
            ui.input_text("nominal_apix", "Nominal Apix (Å/px)", value="1.00", width="120px"),
        ),
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
            ui.input_action_button("add_to_table", ui.tags.span("Add to Table", style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%;"), class_="btn-success", style="height: 38px; display: flex; align-items: center;"),
        ),
        ),

        # ui.div(
        #     {"style": "display: none;"},  # Hidden div for data persistence
        #     ui.output_text("lattice_points_data"),
        # ),
        # ui.div(
        #     {"style": "display: none;"},  # Hidden div for lattice points count
        #     ui.output_text("lattice_points_count"),
        # ),
        
        # FFT Analysis Controls
        #ui.h3("FFT Analysis Controls", style="margin-top: 20px; margin-bottom: 10px;"),
        #ui.output_text("tilt_output"),
        
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
            output_widget("image_display"),
            ui.div(
                {"style": "display: flex; gap: 10px; padding: 10px; justify-content: center;"},
                ui.input_action_button("clear_drawn_region", "Clear Selection", class_="btn-secondary"),
                ui.input_action_button("calc_fft", "Calc FFT", class_="btn-primary"),
            ),
            ui.div(
                {"class": "card-footer"},
                "Use box selection tool to drag and select regions (you'll see red dots), then click 'Calc FFT' to analyze.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("FFT Spectrum"),
            output_widget("fft_with_circle"),
            ui.div(
                {"style": "display: flex; flex-direction: column; gap: 5px; padding: 8px; justify-content: center; height: 20%; min-height: 80px;"},
                ui.div(
                    {"style": "display: flex; gap: 15px; align-items: center;"},
                    ui.div(
                        {"style": "flex: 1;"},
                        ui.input_slider("contrast", "FFT Range (±σ)", min=0.1, max=5.0, value=1.0, step=0.1),
                    ),
                    ui.div(
                        {"style": "flex: 1; font-size: 12px; color: #666; padding-left: 10px;"},
                        ui.output_text("tilt_output"),
                    ),
                ),
                ui.div(
                    {"style": "display: flex; gap: 8px; justify-content: center;"},
                    ui.input_action_button("clear_markers", "Clear Markers", class_="btn-secondary", style="padding: 4px 8px; min-width: auto; white-space: nowrap;"),
                    ui.input_action_button("tune_markers", "Tune Markers", class_="btn-secondary", style="padding: 4px 8px; min-width: auto; white-space: nowrap;"),
                    #ui.input_action_button("clear_measurement", "Clear Measurement", class_="btn-secondary"),
                    ui.input_action_button("fit_markers", "Fit Ellipse", class_="btn-secondary", style="padding: 4px 8px; min-width: auto; white-space: nowrap;"),
                    ui.input_action_button("estimate_tilt", "Estimate Tilt", class_="btn-secondary", style="padding: 4px 8px; min-width: auto; white-space: nowrap;"),
                ),
            ),
            ui.div(
                {"class": "card-footer"},
                "Click to mark points or draw circles.",
            ),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("1D FFT Radial Profile"),

            ui.div(
                {"style": "display: flex;"},
                output_widget("fft_1d_plot"),
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
                    ui.div(
                        {"style": "margin-bottom: 10px;"},
                        ui.input_checkbox("super_resolution", "Super Resolution", value=True),
                        ui.panel_conditional(
                            "input.super_resolution",
                            ui.input_slider("gaussian_window", "Gaussian Window (pixels)", min=1, max=21, value=5, step=2),
                        ),
                    ),
                    ui.input_action_button("find_max", "Find Max", class_="btn-primary"),
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
    ui.card(
        ui.card_header("Region Analysis Table"),
        # Use row layout: table+buttons on left (55%), plot on right (45%), both full height
        ui.layout_columns(
            # Left column: Table and controls (55% width, 100% height)
            ui.div(
                {"style": "display: flex; flex-direction: column; height: 500px;"},
                ui.div(
                    {"style": "flex: 1; overflow-y: auto; padding: 10px; min-height: 0;"},
                    ui.output_data_frame("region_table"),
                ),
                ui.div(
                    {"style": "flex-shrink: 0; display: flex; gap: 10px; padding: 10px; justify-content: center; align-items: center; flex-wrap: wrap; border-top: 1px solid #dee2e6;"},
                    ui.div(
                        {"style": "display: flex; gap: 5px; align-items: center;"},
                        ui.input_action_button("random_generate", "Random Generate", class_="btn-info"),
                        ui.div(
                            {"style": "display: flex; flex-direction: column; align-items: center;"},
                            ui.div(
                                {"style": "font-size: 10px; color: #666; margin-bottom: 2px;"},
                                "Count"
                            ),
                            ui.input_numeric("random_count", None, value=5, min=1, max=100, step=1, width="70px"),
                        ),
                        ui.div(
                            {"style": "display: flex; flex-direction: column; align-items: center;"},
                            ui.div(
                                {"style": "font-size: 10px; color: #666; margin-bottom: 2px;"},
                                "Size %"
                            ),
                            ui.input_numeric("region_size_percent", None, value=0.2, min=0.1, max=1.0, step=0.1, width="70px"),
                        ),
                    ),
                    ui.input_action_button("delete_selected", "Delete Selected", class_="btn-danger"),
                    ui.input_action_button("clear_table", "Clear Table", class_="btn-secondary"),
                    ui.download_button("download_csv", "Download CSV", class_="btn-primary"),
                ),
            ),
            # Right column: Plot (45% width, 100% height)
            ui.div(
                {"style": "height: 500px; padding: 10px; display: flex; flex-direction: column;"},
                ui.div(
                    {"style": "flex: 1; min-height: 0;"},
                    output_widget("apix_centered_by_nominal_plot"),
                ),
            ),
            col_widths=[7, 5],  # 58.3%/41.7% split (closest to 55%/45% with integer grid)
        ),
        ui.div(
            {"class": "card-footer"},
            "Table tracks all analyzed regions with their calibrated pixel sizes. Use Random Generate (count, size%) to analyze random regions from the current image, or manually select/delete rows.",
        ),
        full_screen=True,
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
        
        /* Make Plotly widgets fill their containers */
        .js-plotly-plot {
            height: 100% !important;
            width: 100% !important;
        }
        
        /* Ensure cards with Plotly widgets use full height */
        .card .output_widget {
            height: 100%;
            min-height: 400px;
        }
        
        /* Ensure 2x2 grid layout cards have consistent heights */
        .layout-columns > .card {
            min-height: 500px;
            height: auto;
        }
        
        /* Style for the data table container */
        .data-table-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
        }
        
        /* Data table styling */
        .data-table-container table {
            width: 100%;
            font-size: 0.875rem;
        }
        
        .data-table-container th {
            background-color: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        /* Responsive adjustments for 2x2 grid */
        @media (max-width: 1200px) {
            .layout-columns > .card {
                min-height: 400px;
            }
        }
        
        @media (max-width: 768px) {
            .layout-columns > .card {
                min-height: 300px;
            }
        }

    """),
    ui.tags.script("""
        // Custom JavaScript to ensure only one shape at a time for image display
        document.addEventListener('DOMContentLoaded', function() {
            // Function to clear all shapes except the latest one (only for image display)
            function clearPreviousShapes() {
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(plot => {
                    // Only clear shapes for image display (not FFT display)
                    // Check if this is the image display plot by looking for specific characteristics
                    if (plot.layout && plot.layout.shapes && plot.layout.shapes.length > 1) {
                        // Check if this is likely the image display (has drawrect mode)
                        const isImageDisplay = plot.layout.dragmode === 'drawrect' || 
                                             plot.layout.modebar && plot.layout.modebar.add && 
                                             plot.layout.modebar.add.includes('drawrect');
                        
                        if (isImageDisplay) {
                            // Keep only the last shape for image display
                            const lastShape = plot.layout.shapes[plot.layout.shapes.length - 1];
                            plot.layout.shapes = [lastShape];
                            Plotly.relayout(plot, {shapes: [lastShape]});
                        }
                        // Don't clear shapes for FFT display - allow multiple circles
                    }
                });
            }
            
            // Listen for shape drawing events using MutationObserver
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList') {
                        const plots = document.querySelectorAll('.js-plotly-plot');
                        plots.forEach(plot => {
                            if (plot.layout && plot.layout.shapes && plot.layout.shapes.length > 1) {
                                setTimeout(clearPreviousShapes, 50);
                            }
                        });
                    }
                });
            });
            
            // Start observing
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            // Also listen for click events on plots
            document.addEventListener('click', function(e) {
                if (e.target.closest('.js-plotly-plot')) {
                    setTimeout(clearPreviousShapes, 100);
                }
            });
        });
    """),
    app_ui
)
size = 360

# ---------- Helper Functions ----------

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
        'drawn_circles': [],  # List of drawn circles on FFT image
        'current_measurement': None  # Current line measurement data
    })
    
    # Separate reactive values for FFT image rendering to avoid unnecessary re-renders
    fft_markers = reactive.Value({
        'mode': 'Resolution Ring',
        'resolution_click_x': None,
        'resolution_click_y': None,
        'lattice_points': [],
        'ellipse_params': None,
        'zoom_factor': 1.0
    })
    


    # --- Single source of truth for apix ---
    apix_master = reactive.Value(1.0)

    # Effect to update fft_markers when relevant parts of fft_state change
    @reactive.Effect
    @reactive.event(fft_state)
    def _():
        """Update fft_markers when relevant parts of fft_state change."""
        state = fft_state.get()
        fft_markers.set({
            'mode': state['mode'],
            'resolution_click_x': state['resolution_click_x'],
            'resolution_click_y': state['resolution_click_y'],
            'lattice_points': state['lattice_points'].copy(),
            'ellipse_params': state['ellipse_params'],
            'zoom_factor': state['zoom_factor']
        })
    
    # Remove fft_1d_data since we're no longer using static markers

    # Add reactive value to cache the base FFT image
    cached_fft_image = reactive.Value(None)

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

    # Add reactive values for image display
    image_data = reactive.Value(None)
    image_apix = reactive.Value(1.0)
    image_filename = reactive.Value(None)
    
    # Add reactive values for original and binned image data
    original_image_data = reactive.Value(None)
    binned_image_data = reactive.Value(None)
    
    # Add reactive value for image zoom state
    image_zoom_state = reactive.Value({
        'x_range': None,
        'y_range': None,
        'is_zoomed': False,
        'drawn_region': None  # Store drawn rectangle coordinates
    })

    # Add reactive value to trigger FFT calculations
    fft_trigger = reactive.Value(0)
    
    # Add reactive value to store the 1D plot FigureWidget for in-place updates
    fft_1d_widget = reactive.Value(None)
    
    # Add reactive value to store the FFT FigureWidget for in-place overlay updates
    fft_widget = reactive.Value(None)
    
    # Add reactive value to store all drawn shapes
    drawn_shapes = reactive.Value([])
    
    # Add separate reactive value to store box coordinates directly
    box_coordinates = reactive.Value(None)
    
    # Add separate reactive value for lattice points to avoid FFT re-renders
    lattice_points_storage = reactive.Value([])
    
    # Add separate reactive value for tilt information to avoid FFT re-renders
    tilt_info_storage = reactive.Value(None)
    
    # Add separate reactive values for dual tilt information
    tilt_info_green_storage = reactive.Value(None)
    tilt_info_red_storage = reactive.Value(None)
    
    # Add separate reactive value for ellipse parameters to avoid FFT re-renders
    ellipse_params_storage = reactive.Value(None)
    
    # Add separate reactive value for tuned markers (red circles from local maxima)
    tuned_markers_storage = reactive.Value([])
    
    # Add separate reactive value for tuned resolution ring
    tuned_resolution_radius = reactive.Value(None)
    
    # Add separate reactive value for current mode to avoid FFT re-renders
    current_mode_storage = reactive.Value('Resolution Ring')
    
    # Add reactive value to trigger only overlay updates (not base FFT re-render)
    overlay_update_trigger = reactive.Value(0)
    
    # Add reactive value that only changes when base FFT image changes
    base_fft_trigger = reactive.Value(0)
    
    # Add reactive value to trigger autoscale on FFT calculation (not contrast changes)
    autoscale_trigger = reactive.Value(0)
    
    # Helper function to extract nominal value from filename
    def extract_nominal(filename):
        """Extract nominal value from filename (e.g., 0.75 from '130k-Pixel0.75A.tiff').
        Returns 1.0 if extraction fails."""
        import re
        if not isinstance(filename, str):
            return 1.0
        m = re.search(r"(\d+\.\d+)A", filename)
        return float(m.group(1)) if m else 1.0
    
    # Add reactive value to store the region analysis table data
    region_table_data = reactive.Value(pd.DataFrame({
        'Filename': [],
        'Region Size': [],
        'Region Location': [],
        'Apix': [],
        'Nominal': []
    }))
    
    # Add reactive value to store the region and parameters used for current FFT calculation
    fft_calculation_state = reactive.Value({
        'region': None,
        'apix': None,
        'resolution_type': None,
        'custom_resolution': None
    })
    
    # Update base FFT trigger and 1D plot when cached FFT image changes
    @reactive.Effect
    @reactive.event(cached_fft_image)
    def _():
        """Update base FFT trigger and 1D plot when the base FFT image changes."""
        base_fft_trigger.set(base_fft_trigger.get() + 1)
        
        # Also update 1D plot widget if it exists
        widget = fft_1d_widget.get()
        if widget is not None and len(widget.data) > 0:
            # Get updated plot data using stored calculation state
            plot_data = fft_1d_data()
            if plot_data is not None:
                # Update the trace data in-place
                with widget.batch_update():
                    widget.data[0].x = plot_data['x_data']
                    widget.data[0].y = plot_data['y_data']
                    widget.data[0].name = plot_data['profile_label']
                    
                    # Update y-axis title based on log_y setting
                    if input.log_y():
                        widget.layout.yaxis.title.text = "Log(FFT intensity)"
                    else:
                        widget.layout.yaxis.title.text = "FFT intensity"

    # Initialize Fit button state
    @reactive.Effect
    def _():
        """Initialize Fit button state."""
        is_disabled = input.label_mode() != "Lattice Point"
        # Tune Markers now works in both modes, so don't disable it
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)
    
    # Update Estimate Tilt button state based on ellipse fitting
    # @reactive.Effect
    # @reactive.event(fft_state)
    # def _():
    #     """Update Estimate Tilt button state based on ellipse fitting."""
    #     current_state = fft_state.get()
    #     if current_state['mode'] == 'Lattice Point':
    #         # Enable Estimate Tilt only if ellipse is fitted
    #         has_ellipse = current_state['ellipse_params'] is not None
    #         ui.update_action_button("estimate_tilt", disabled=not has_ellipse, session=session)

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







    # Note: Click events are now handled by Plotly's on_click callback
    # No need for Shiny event handlers


    @reactive.Effect
    @reactive.event(input.clear_markers)
    def _():
        """Clear all markers based on current mode."""
        current_state = fft_state.get()
        new_state = current_state.copy()
        
        # Check the actual UI mode instead of stored mode to avoid sync issues
        current_ui_mode = input.label_mode()
        
        if current_ui_mode == 'Resolution Ring':
            # Clear resolution ring markers
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
        elif current_ui_mode == 'Lattice Point':
            # Clear lattice points, ellipse, and tilt info
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            # Also clear the separate lattice points storage
            lattice_points_storage.set([])
            # Also clear the separate tilt info storage
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the dual tilt info storages
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the separate ellipse params storage
            ellipse_params_storage.set(None)
            # Also clear the tuned markers storage
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            # Also clear the tuned resolution ring storage
            tuned_resolution_radius.set(None)
        
        # Clear drawn circles for all modes
        new_state['drawn_circles'] = []
        
        # Clear ALL overlay traces and shapes directly from FFT widget (no re-render)
        fft_widget_instance = fft_widget.get()
        if fft_widget_instance is not None:

            
            with fft_widget_instance.batch_update():
                # Remove all ellipse_fit traces using index-based approach (more reliable)
                ellipse_indices = []
                for i, trace in enumerate(fft_widget_instance.data):
                    if hasattr(trace, 'name') and trace.name and ('ellipse_fit' in trace.name):
                        ellipse_indices.append(i)
                
                # Remove ellipse traces from the end to avoid index shifting
                for i in reversed(ellipse_indices):
                    fft_widget_instance.data = fft_widget_instance.data[:i] + fft_widget_instance.data[i+1:]
                
                # Clear all overlay shapes (keep only the base shapes if any)
                # This will remove all lattice point shapes and drawn circles
                fft_widget_instance.layout.shapes = []
                fft_widget_instance.layout.annotations = []
            # print("All overlay traces and shapes cleared from FFT widget (no re-render)")
        
        fft_state.set(new_state)

    # @reactive.Effect
    # @reactive.event(input.clear_measurement)
    # def _():
    #     """Clear current measurement."""
    #     current_state = fft_state.get()
    #     new_state = current_state.copy()
    #     new_state['current_measurement'] = None
    #     fft_state.set(new_state)
    #     print("Measurement cleared manually")







    @reactive.Effect
    @reactive.event(input.clear_drawn_region)
    def _():
        """Clear selected region only."""
        current_zoom_state = image_zoom_state.get()
        new_zoom_state = current_zoom_state.copy()
        new_zoom_state['drawn_region'] = None
        image_zoom_state.set(new_zoom_state)
        drawn_shapes.set([])
        box_coordinates.set(None)
        
        # Clear FFT calculation state when region is cleared
        fft_calculation_state.set({
            'region': None,
            'apix': None,
            'resolution_type': None,
            'custom_resolution': None
        })
        
        print("Selected region cleared")
    
    


    @reactive.Effect
    @reactive.event(input.calc_fft)
    def _():
        """Manually trigger FFT calculation."""
        try:
            print("=== MANUAL FFT CALCULATION TRIGGERED ===")
            
            # Clear all overlay storage from previous FFT analysis
            # This ensures lattice points, ellipse fits, and tilt info from previous regions/images don't persist
            lattice_points_storage.set([])
            ellipse_params_storage.set(None)
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            
            # Reset FFT state to clear drawn circles and measurements
            fft_state.set({
                'mode': 'Resolution Ring',
                'resolution_radius': None,
                'resolution_click_x': None,
                'resolution_click_y': None,
                'lattice_points': [],
                'ellipse_params': None,
                'tilt_info': None,
                'zoom_factor': 1.0,
                'drawn_circles': [],
                'current_measurement': None
            })
            
            # Check box_coordinates from callback
            box_coords = box_coordinates.get()
            print(f"Box coordinates: {box_coords}")
            
            # Use box_coordinates if available (preferred method)
            if box_coords is not None:
                print(f"✅ USING BOX COORDINATES: {box_coords}")
                
                # Update zoom state with the box coordinates
                current_zoom_state = image_zoom_state.get()
                new_zoom_state = current_zoom_state.copy()
                new_zoom_state['drawn_region'] = box_coords
                new_zoom_state['is_zoomed'] = True
                image_zoom_state.set(new_zoom_state)
                
                print(f"✅ Using box coordinates: x0={box_coords['x0']:.1f}, x1={box_coords['x1']:.1f}, y0={box_coords['y0']:.1f}, y1={box_coords['y1']:.1f}")
                
            else:
                # Check if there's a region in zoom state (fallback)
                zoom_state = image_zoom_state.get()
                if zoom_state.get('drawn_region') is not None:
                    selected_region = zoom_state['drawn_region']
                    print(f"✅ Using existing zoom state region: {selected_region}")
                else:
                    # No coordinates available
                    print("❌ ERROR: No box selection found!")
                    print("Please use the box selection tool to select a region on the image.")
                    print("You should see red dots appear in the selected area.")
                    return
            
            # Trigger FFT calculation
            fft_trigger.set(fft_trigger.get() + 1)
            
            # Trigger autoscale (only on calc_fft, not on contrast changes)
            autoscale_trigger.set(autoscale_trigger.get() + 1)
            
            print("✅ FFT calculation triggered successfully!")
            
        except Exception as e:
            print(f"❌ ERROR in calc_fft function: {e}")
            import traceback
            traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.fit_markers)
    def _():
        """Handle Fit button click to fit ellipse to lattice points."""
        current_state = fft_state.get()
        # Check the actual UI input instead of relying on fft_state mode sync
        if input.label_mode() != 'Lattice Point':
            return
            
        # Get both sets of points
        tuned_points = list(tuned_markers_storage.get())
        user_points = list(lattice_points_storage.get())
        
        if len(tuned_points) == 0 and len(user_points) == 0:
            print("No lattice points or tuned markers to fit ellipse to.")
            return
            
        print(f"Fitting ellipses: {len(user_points)} user points, {len(tuned_points)} tuned points")
            
        # Compute image center - use actual FFT image size, not hardcoded size
        # Get the actual FFT image size from the cached image
        cached_fft = cached_fft_image.get()
        if cached_fft is not None:
            fft_image_size = cached_fft.size[0]  # Assuming square image
            cx, cy = fft_image_size / 2, fft_image_size / 2
            print(f"Using actual FFT image size: {fft_image_size}, center: ({cx}, {cy})")
        else:
            # Fallback to hardcoded size
            cx, cy = size / 2, size / 2
            print(f"Using fallback size: {size}, center: ({cx}, {cy})")
        
        # Function to create working points for ellipse fitting
        def create_working_points(points):
            working_points = points.copy()
            # If fewer than 3 points, create additional points by mirroring and jittering
            if len(points) < 3:
                print(f"Only {len(points)} points available. Creating additional points for better ellipse fitting...")
                
                # Mirror each point through the center and add jittered versions
                for x, y in points:
                    # Mirror through center
                    mx, my = 2 * cx - x, 2 * cy - y
                    
                    # Add the mirrored point
                    working_points.append((mx, my))
                    
                    # Add jittered versions of both original and mirrored points
                    for _ in range(2):  # Create 2 jittered versions of each
                        # Jitter original point
                        jittered_x = x + np.random.normal(scale=2.0)
                        jittered_y = y + np.random.normal(scale=2.0)
                        working_points.append((jittered_x, jittered_y))
                        
                        # Jitter mirrored point
                        jittered_mx = mx + np.random.normal(scale=2.0)
                        jittered_my = my + np.random.normal(scale=2.0)
                        working_points.append((jittered_mx, jittered_my))
            return working_points
        
        # Function to fit ellipse and draw it
        def fit_and_draw_ellipse(points, color, ellipse_name):
            if len(points) == 0:
                return None
                
            working_points = create_working_points(points)
            print(f"Fitting {color} ellipse to {len(working_points)} points (including {len(points)} original points)")
            
            try:
                a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
                
                # Validate ellipse parameters
                max_reasonable_radius = fft_image_size if 'fft_image_size' in locals() else size
                if a > max_reasonable_radius or b > max_reasonable_radius:
                    print(f"Warning: {color} ellipse axes too large (a={a:.1f}, b={b:.1f}), max reasonable={max_reasonable_radius}")
                    return None
                
                print(f"{color.capitalize()} ellipse fitted: a={a:.1f}, b={b:.1f}, theta={theta:.3f}")
                
                # Create ellipse trace
                t = np.linspace(0, 2*np.pi, 100)
                x_ellipse = a * np.cos(t)
                y_ellipse = b * np.sin(t)
                x_rot = x_ellipse * np.cos(theta) - y_ellipse * np.sin(theta)
                y_rot = x_ellipse * np.sin(theta) + y_ellipse * np.cos(theta)
                x_final = cx + x_rot
                y_final = cy + y_rot
                
                return {
                    'params': (a, b, theta),
                    'trace_data': (x_final, y_final),
                    'color': color,
                    'name': ellipse_name
                }
            except Exception as e:
                print(f"{color.capitalize()} ellipse fitting failed: {e}")
                return None
        
        # Fit ellipses for both point sets
        green_ellipse = None
        red_ellipse = None
        
        if len(user_points) > 0:
            green_ellipse = fit_and_draw_ellipse(user_points, 'green', 'ellipse_fit_green')
        
        if len(tuned_points) > 0:
            red_ellipse = fit_and_draw_ellipse(tuned_points, 'red', 'ellipse_fit_red')
        
        # Store ellipse parameters (prioritize tuned markers for tilt calculations)
        if red_ellipse:
            ellipse_params_storage.set(red_ellipse['params'])
        elif green_ellipse:
            ellipse_params_storage.set(green_ellipse['params'])
        
        # Add ellipse overlays directly to existing FFT widget (no re-render)
        fft_widget_instance = fft_widget.get()
        if fft_widget_instance is not None:
            with fft_widget_instance.batch_update():
                # Remove any existing ellipse_fit traces (both green and red)
                ellipse_indices = []
                for i, trace in enumerate(fft_widget_instance.data):
                    if hasattr(trace, 'name') and trace.name and ('ellipse_fit' in trace.name):
                        ellipse_indices.append(i)
                
                # Remove ellipse traces from the end to avoid index shifting
                for i in reversed(ellipse_indices):
                    fft_widget_instance.data = fft_widget_instance.data[:i] + fft_widget_instance.data[i+1:]
                
                # Add green ellipse trace if available
                if green_ellipse:
                    x_final, y_final = green_ellipse['trace_data']
                    fft_widget_instance.add_trace(go.Scatter(
                        x=x_final, 
                        y=y_final, 
                        mode='lines', 
                        line=dict(color='green', width=2), 
                        showlegend=False, 
                        hoverinfo='skip',
                        name='ellipse_fit_green'
                    ))
                
                # Add red ellipse trace if available
                if red_ellipse:
                    x_final, y_final = red_ellipse['trace_data']
                    fft_widget_instance.add_trace(go.Scatter(
                        x=x_final, 
                        y=y_final, 
                        mode='lines', 
                        line=dict(color='red', width=2), 
                        showlegend=False, 
                        hoverinfo='skip',
                        name='ellipse_fit_red'
                    ))
            
            #print(f"Ellipse overlay added directly to FFT widget (no re-render)")
        else:
            print("Warning: FFT widget not available for ellipse overlay")

    @reactive.Effect
    @reactive.event(input.tune_markers)
    def _():
        """Handle Tune Markers button click to find super-resolution maxima."""
        current_mode = input.label_mode()
        
        if current_mode == 'Lattice Point':
            # Lattice Point mode: 2D Gaussian fitting for crosshairs
            lattice_points = list(lattice_points_storage.get())
            if len(lattice_points) == 0:
                print("No lattice points available for tuning.")
                return
                
            # Get the cached FFT image data
            cached_fft = cached_fft_image.get()
            if cached_fft is None:
                print("No FFT image available for tuning markers.")
                return
                
            # Convert PIL image to numpy array for processing
            fft_array = np.array(cached_fft)
            if len(fft_array.shape) == 3:  # RGB image
                fft_array = np.mean(fft_array, axis=2)  # Convert to grayscale
                
            tuned_points = []
            
            # Window size should match green circle radius (8 pixels)
            window_radius = 8
            
            # Define 2D Gaussian function
            def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
                x, y = coords
                return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + offset
            
            # Process each lattice point with 2D Gaussian fitting
            for x, y in lattice_points:
                # Convert to integer coordinates for window extraction
                ix, iy = int(round(x)), int(round(y))
                
                # Define window bounds (same size as green circle)
                x_min = max(0, ix - window_radius)
                x_max = min(fft_array.shape[1], ix + window_radius + 1)
                y_min = max(0, iy - window_radius)
                y_max = min(fft_array.shape[0], iy + window_radius + 1)
                
                # Extract window around the lattice point
                window = fft_array[y_min:y_max, x_min:x_max]
                
                if window.size < 25:  # Need minimum window size for fitting
                    tuned_points.append((x, y))
                    print(f"Window too small for point ({x:.1f}, {y:.1f}), keeping original")
                    continue
                
                try:
                    # Create coordinate arrays for the window
                    window_height, window_width = window.shape
                    xx, yy = np.meshgrid(np.arange(window_width), np.arange(window_height))
                    
                    # Flatten for fitting
                    coords = np.vstack([xx.ravel(), yy.ravel()])
                    data = window.ravel()
                    
                    # Initial parameter guesses
                    amplitude_guess = np.max(window) - np.min(window)
                    x0_guess = window_width / 2  # Center of window
                    y0_guess = window_height / 2  # Center of window
                    sigma_guess = window_radius / 3  # Reasonable sigma
                    offset_guess = np.min(window)
                    
                    # Fit 2D Gaussian
                    from scipy.optimize import curve_fit
                    popt, _ = curve_fit(
                        gaussian_2d, 
                        coords, 
                        data,
                        p0=[amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, offset_guess],
                        maxfev=1000,
                        bounds=([0, 0, 0, 0.5, 0.5, 0], 
                               [np.inf, window_width, window_height, window_radius, window_radius, np.inf])
                    )
                    
                    # Extract fitted center coordinates
                    _, fitted_x, fitted_y, sigma_x, sigma_y, _ = popt
                    
                    # Convert back to image coordinates
                    tuned_x = x_min + fitted_x
                    tuned_y = y_min + fitted_y
                    
                    tuned_points.append((tuned_x, tuned_y))
                    print(f"2D Gaussian fit: ({x:.1f}, {y:.1f}) -> ({tuned_x:.3f}, {tuned_y:.3f}), σ=({sigma_x:.2f}, {sigma_y:.2f})")
                    
                except Exception as e:
                    # Gaussian fitting failed, keep original point
                    tuned_points.append((x, y))
                    print(f"2D Gaussian fitting failed for ({x:.1f}, {y:.1f}): {e}, keeping original")
                
            # Store tuned markers separately
            tuned_markers_storage.set(tuned_points)
            print(f"Tuned {len(tuned_points)} lattice markers using 2D Gaussian fitting.")
            
        elif current_mode == 'Resolution Ring':
            # Resolution Ring mode: 1D radial profile Gaussian fitting
            current_state = fft_state.get()
            resolution_radius = current_state.get('resolution_radius')
            
            if resolution_radius is None:
                print("No resolution ring available for tuning.")
                return
                
            # Get 1D FFT data for radial profile
            plot_data = fft_1d_data()
            if plot_data is None:
                print("No 1D FFT data available for tuning.")
                return
                
            x_data = plot_data['x_data']
            y_data = plot_data['y_data']
            
            # Find the closest data point to the clicked resolution radius
            radius_idx = np.argmin(np.abs(x_data - resolution_radius))
            original_radius = x_data[radius_idx]
            
            # Use same window size as Find Max function for consistency
            window_half_size = input.gaussian_window() / 2.0
            
            # Find indices within ±window_half_size pixels (same as Find Max)
            mask = np.abs(x_data - original_radius) <= window_half_size
            
            if np.sum(mask) < 5:
                print(f"Not enough points for 1D Gaussian fitting around radius {original_radius:.2f}")
                return
                
            # Extract window data
            fit_x = x_data[mask]
            fit_y = y_data[mask]
            
            try:
                # Define Gaussian function (identical to Find Max)
                def gaussian(x, a, mu, sigma, c):
                    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                
                # Initial parameter guesses (identical to Find Max)
                max_y_fit = np.max(fit_y)  # Find maximum in fitting window
                a_guess = max_y_fit  # amplitude (absolute maximum value)
                mu_guess = original_radius  # center
                sigma_guess = 1.0     # width
                c_guess = np.min(fit_y)  # offset
                
                # Fit Gaussian (identical to Find Max)
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(
                    gaussian, 
                    fit_x, 
                    fit_y,
                    p0=[a_guess, mu_guess, sigma_guess, c_guess],
                    maxfev=1000
                )
                
                # Extract fitted parameters (identical to Find Max)
                a_fit, mu_fit, sigma_fit, c_fit = popt
                tuned_radius = mu_fit
                
                # Store tuned resolution ring
                tuned_resolution_radius.set(tuned_radius)
                print(f"1D Gaussian fit: radius {original_radius:.3f} -> {tuned_radius:.5f}, σ={sigma_fit:.3f}")
                
                # Calculate and update apix based on tuned radius
                resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
                if resolution is not None:
                    # Get FFT image size for apix calculation
                    cached_fft = cached_fft_image.get()
                    if cached_fft is not None:
                        fft_image_size = cached_fft.size[0]  # PIL image size
                        calculated_apix = (tuned_radius * resolution) / fft_image_size
                        
                        # Update the apix slider with the tuned value
                        ui.update_slider("apix_slider", value=calculated_apix, session=session)
                        ui.update_text("apix_exact_str", value=f"{calculated_apix:.3f}", session=session)
                        print(f"Updated apix to {calculated_apix:.3f} Å/px based on tuned resolution ring.")
                
                print(f"Tuned resolution ring using 1D radial profile Gaussian fitting.")
                
            except Exception as e:
                print(f"1D Gaussian fitting failed for radius {original_radius:.3f}: {e}")
                
        else:
            print(f"Tune Markers not supported for mode: {current_mode}")

    @reactive.Effect
    @reactive.event(input.estimate_tilt)
    def _():
        """Handle Estimate Tilt button click to compute tilt angle from ellipse(s)."""
        # Check the actual UI input instead of relying on fft_state mode sync
        if input.label_mode() != 'Lattice Point':
            return
        
        # Get both sets of points
        tuned_points = list(tuned_markers_storage.get())
        user_points = list(lattice_points_storage.get())
        
        if len(tuned_points) == 0 and len(user_points) == 0:
            print("No lattice points or tuned markers available for tilt estimation.")
            return
        
        print(f"Estimating tilt: {len(user_points)} user points, {len(tuned_points)} tuned points")
        
        # Get image center
        cached_fft = cached_fft_image.get()
        if cached_fft is not None:
            fft_image_size = cached_fft.size[0]  # Assuming square image
            cx, cy = fft_image_size / 2, fft_image_size / 2
        else:
            cx, cy = size / 2, size / 2
            fft_image_size = size
        
        # Function to calculate tilt from points
        def calculate_tilt_from_points(points, point_type):
            if len(points) == 0:
                return None
                
            # Create working points for ellipse fitting
            working_points = points.copy()
            
            # If fewer than 6 points, create additional points by mirroring and jittering
            if len(points) < 6:
                print(f"Only {len(points)} {point_type} points available. Creating additional points for better ellipse fitting...")
                
                # Mirror each point through the center and add jittered versions
                for x, y in points:
                    # Mirror through center
                    mx, my = 2 * cx - x, 2 * cy - y
                    
                    # Add the mirrored point
                    working_points.append((mx, my))
                    
                    # Add jittered versions of both original and mirrored points
                    for _ in range(2):  # Create 2 jittered versions of each
                        # Jitter original point
                        jittered_x = x + np.random.normal(scale=2.0)
                        jittered_y = y + np.random.normal(scale=2.0)
                        working_points.append((jittered_x, jittered_y))
                        
                        # Jitter mirrored point
                        jittered_mx = mx + np.random.normal(scale=2.0)
                        jittered_my = my + np.random.normal(scale=2.0)
                        working_points.append((jittered_mx, jittered_my))
            
            print(f"Fitting {point_type} ellipse to {len(working_points)} points for tilt estimation...")
            
            # Fit ellipse
            try:
                a, b, theta = fit_ellipse_fixed_center(working_points, center=(cx, cy))
                
                small_axis, large_axis = sorted([a, b])
                tilt_angle = calculate_tilt_angle(small_axis, large_axis)
                
                # Calculate apix using the minor axis (untilted apix)
                resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
                untilted_apix = None
                if resolution is not None and small_axis > 0:
                    untilted_apix = (small_axis * resolution) / fft_image_size
                    if not (0.01 <= untilted_apix <= 6.0):
                        print(f"Warning: {point_type} calculated apix {untilted_apix:.3f} is outside valid range [0.01, 6.0]")
                        untilted_apix = None
                
                apix_display = f"{untilted_apix:.3f}" if untilted_apix else "N/A"
                print(f"{point_type.capitalize()} ellipse tilt: {math.degrees(tilt_angle):.2f}°, untilted apix: {apix_display}")
                
                return (small_axis, large_axis, tilt_angle, untilted_apix)
            except Exception as e:
                print(f"{point_type.capitalize()} ellipse fitting failed: {e}")
                return None
        
        # Calculate tilt for both point sets
        green_tilt = None
        red_tilt = None
        
        if len(user_points) > 0:
            green_tilt = calculate_tilt_from_points(user_points, "green")
            
        if len(tuned_points) > 0:
            red_tilt = calculate_tilt_from_points(tuned_points, "red")
        
        # Store tilt info in separate storages
        tilt_info_green_storage.set(green_tilt)
        tilt_info_red_storage.set(red_tilt)
        
        # Store primary tilt info (prioritize tuned markers)
        primary_tilt = red_tilt if red_tilt else green_tilt
        tilt_info_storage.set(primary_tilt)
        
        # Update UI with the primary (best) untilted apix
        if primary_tilt and primary_tilt[3] is not None:
            untilted_apix = primary_tilt[3]
            ui.update_slider("apix_slider", value=untilted_apix, session=session)
            ui.update_text("apix_exact_str", value=str(round(untilted_apix, 3)), session=session)
            print(f"Updated UI with untilted apix: {untilted_apix:.3f} Å/px")
        
        print(f"Tilt information stored in separate storages")

    # Remove click handler for 1D plot since we're using hover instead of static markers
    
    # Note: Plotly handles zoom and pan automatically, so we don't need separate brush/dblclick handlers
    # The plot_zoom state is still used for programmatic zoom control



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


    @reactive.Calc
    def get_apix():
        return apix_master.get()

    @reactive.Calc
    def get_apix_from_distance():
        """Calculate the apix value from a given distance in pixels and current resolution.
        
        Returns:
            A function that takes distance in pixels and returns the corresponding apix value.
        """
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
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
            return calculate_apix_from_distance(distance_pixels, resolution, size)
        
        return calculate_apix

    @reactive.Calc
    def get_distance_from_apix():
        """Calculate the distance in pixels from a given apix value and current resolution.
        
        Returns:
            A function that takes apix value and returns the corresponding distance in pixels.
        """
        resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
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
            return calculate_distance_from_apix(apix_value, resolution, size)
        
        return calculate_distance



    @reactive.Effect
    @reactive.event(input.upload)
    def _():
        """Update image data when a new file is uploaded."""
        path = image_path()
        file_info = input.upload()
        
        if not path or not path.exists() or not file_info:
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            cached_fft_image.set(None)

            
            # Clear all overlay storage when upload fails
            lattice_points_storage.set([])
            ellipse_params_storage.set(None)
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            fft_state.set({
                'mode': 'Resolution Ring',
                'resolution_radius': None,
                'resolution_click_x': None,
                'resolution_click_y': None,
                'lattice_points': [],
                'ellipse_params': None,
                'tilt_info': None,
                'zoom_factor': 1.0,
                'drawn_circles': [],
                'current_measurement': None
            })
            return
            
        # Load image using the original get_image function (no binning for FFT analysis)
        try:
            print(f"Loading image from: {path}")
            # Load original image without binning for FFT analysis
            original_data, target_apix, original_apix = get_image(str(path))
            print(f"Image loaded successfully: original_shape={original_data.shape}, apix={target_apix}")
            
            # Create binned version for display only
            binned_data = bin_image(original_data, target_size=1000)
            print(f"Created binned version for display: binned_shape={binned_data.shape}")
            
            # Get the original filename from the upload info instead of the temporary path
            original_filename = file_info[0]["name"]
            #print(f"Original filename: {original_filename}")
            
            # Extract nominal apix from filename and update the textbox
            nominal_value = extract_nominal(original_filename)
            ui.update_text("nominal_apix", value=f"{nominal_value:.2f}", session=session)
            print(f"Extracted nominal apix from filename: {nominal_value:.2f}")
            
            # Set the binned data for display (always 1000x1000)
            image_data.set(binned_data)
            image_apix.set(target_apix)
            image_filename.set(original_filename)
            
            # Store original and binned data separately
            original_image_data.set(original_data)
            binned_image_data.set(binned_data)
            
            # Reset zoom state
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            
            # Reset FFT trigger and clear cached FFT images (but keep table data)
            fft_trigger.set(0)
            cached_fft_image.set(None)
            drawn_shapes.set([])
            
            # Clear FFT calculation state when a new image is loaded
            # This will make FFT displays empty until user calculates FFT for the new image
            fft_calculation_state.set({
                'region': None,
                'apix': None,
                'resolution_type': None,
                'custom_resolution': None
            })
            
            # Clear all overlay storage from previous image analysis
            # This ensures lattice points, ellipse fits, and tilt info don't persist to new images
            lattice_points_storage.set([])
            ellipse_params_storage.set(None)
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            fft_state.set({
                'mode': 'Resolution Ring',
                'resolution_radius': None,
                'resolution_click_x': None,
                'resolution_click_y': None,
                'lattice_points': [],
                'ellipse_params': None,
                'tilt_info': None,
                'zoom_factor': 1.0,
                'drawn_circles': [],
                'current_measurement': None
            })
            
            # Note: region_table_data is NOT cleared to allow comparison across multiple images
            
            # Also keep the old format for compatibility with FFT calculations
            img = Image.fromarray(binned_data.astype(np.uint8))
            raw_image_data.set({
                'img': img,
                'data': binned_data
            })
            print(f"Image data set successfully")
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)

    @reactive.Effect
    @reactive.event(fft_trigger)
    def _():
        """Update cached FFT images when FFT is manually triggered (Calc FFT button)."""
        #print("FFT calculation triggered - checking drawn region")
        
        region = get_current_region()
        if region is not None:
            print(f"FFT region size: {region.size}")
            # Generate base FFT image with current contrast
            fft_img = compute_fft_image_region(region, input.contrast())
            cached_fft_image.set(fft_img)
            
            # Store the calculation state for 1D FFT consistency
            fft_calculation_state.set({
                'region': region,
                'apix': get_apix(),
                'resolution_type': input.resolution_type(),
                'custom_resolution': input.custom_resolution()
            })
            
            print("FFT image updated successfully")
        else:
            cached_fft_image.set(None)
            print("No region available for FFT calculation")

    # Remove the effect that forces FFT redraws - this was causing unnecessary re-renders

    @output
    @render_widget
    def image_display():
        # Only show the image if it is available
        current_image_data = image_data.get()
        if current_image_data is None:
            # Return None to show nothing when no image is uploaded
            return None
        
        print(f"=== RENDERING IMAGE DISPLAY ===")
        print(f"Image data shape: {current_image_data.shape}")
        
        # Create a FigureWidget for box selection
        figw = go.FigureWidget()
        
        # Add heatmap for the image
        figw.add_trace(go.Heatmap(
            z=current_image_data,
            colorscale="gray",
            showscale=False,
            zmin=0,
            zmax=255,
            hoverinfo="skip",
            opacity=1.0
        ))
        
        # Add scatter overlay to capture selection events
        # Create a grid of points covering the entire image
        height, width = current_image_data.shape
        step = 50  # Every 50 pixels for sparse coverage
        y_coords, x_coords = np.meshgrid(
            np.arange(0, height, step),
            np.arange(0, width, step),
            indexing='ij'
        )
        
        scatter = go.Scatter(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            mode='markers',
            marker=dict(
                size=2,  # Small markers
                opacity=0.0,  # Invisible by default
                color='red'
            ),
            showlegend=False,
            hoverinfo='none',
            name='selection_overlay',
            # Enable selection on this trace
            selected=dict(marker=dict(opacity=0.6, color='red', size=3)),
            unselected=dict(marker=dict(opacity=0.0)),
        )
        figw.add_trace(scatter)
        
        # Configure layout for box selection
        figw.update_layout(
            # Enable box selection mode (equivalent to R's dragmode = "select")
            dragmode='select',
            # Configure selection behavior
            selectdirection='any',
            newselection=dict(
                mode='immediate'
            ),
            modebar=dict(
                add=['select2d', 'lasso2d', 'zoom', 'pan', 'reset+autorange'],
                remove=['drawrect', 'eraseshape']
            ),
            # Ensure selection events are captured
            uirevision='box_selection',
            # Add explicit event handling
            clickmode='event+select',
            hovermode=False,
            # Set autosize and margins
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="white",
            title=None,
            # Configure axes
            xaxis=dict(
                fixedrange=False,
                showgrid=False,
            ),
            yaxis=dict(
                fixedrange=False,
                showgrid=False,
            )
        )
        
        # Hide axes but keep them functional for events
        figw.update_xaxes(
            visible=False,
            rangeslider_visible=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        figw.update_yaxes(
            visible=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        
        # Force square aspect ratio
        figw.update_xaxes(scaleanchor="y", scaleratio=1)
        
        # === attach on_selection handler ===
        def _on_box_selection(trace, points, selector):
            # points.xs, points.ys are the coordinates of the selected pts
            if not points.point_inds:
                box_coordinates.set(None)
                print("❌ No points in selection")
                return
            xs, ys = points.xs, points.ys
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            
            coords = {
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1
            }
            box_coordinates.set(coords)
            
            # Also update legacy formats for compatibility
            selection_shape = {
                'type': 'rect',
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1
            }
            drawn_shapes.set([selection_shape])
            
            # Update zoom state
            current_zoom_state = image_zoom_state.get()
            new_zoom_state = current_zoom_state.copy()
            new_zoom_state['drawn_region'] = coords
            new_zoom_state['is_zoomed'] = True
            image_zoom_state.set(new_zoom_state)
            
            print(f"📦 Captured region via callback: X[{x0:.1f},{x1:.1f}] "
                  f"Y[{y0:.1f},{y1:.1f}] pts={len(points.point_inds)}")
        
        # attach to the scatter trace (trace index 1)
        figw.data[1].on_selection(_on_box_selection)
        
        print(f"=== IMAGE DISPLAY RENDERED WITH BOX SELECTION CALLBACK ===")
        return figw

    def get_current_region():
        """Get the current region for FFT calculation based on drawn square or use entire image.
        Uses extract_region_no_binning from compute.py to ensure full-resolution regions for accurate FFT analysis."""
        if image_data.get() is None:
            print("No image data available")
            return None
            
        zoom_state = image_zoom_state.get()
        original_data = original_image_data.get()
        binned_data = binned_image_data.get()
        
        if original_data is None or binned_data is None:
            print("No original or binned data available")
            return None
            
        print(f"Zoom state: {zoom_state}")
        print(f"Original data shape: {original_data.shape}")
        print(f"Binned data shape: {binned_data.shape}")
        
        # Check if there's a drawn region (square)
        if zoom_state.get('drawn_region') is not None:
            drawn_region = zoom_state['drawn_region']
            print(f"Using drawn region: {drawn_region}")
            
            # Calculate original rectangle dimensions
            width = drawn_region['x1'] - drawn_region['x0']
            height = drawn_region['y1'] - drawn_region['y0']
            
            # Use the smaller dimension to create a square
            square_size = min(width, height)
            
            # Ensure minimum size of 50 pixels to prevent FFT errors with 0-sized regions
            if square_size < 50:
                square_size = 50
                print(f"Original region: {width:.1f} x {height:.1f}, too small - using minimum size: {square_size:.1f}")
            else:
                print(f"Original region: {width:.1f} x {height:.1f}, making square with size: {square_size:.1f}")
            
            # Center the square within the original selection
            center_x = (drawn_region['x0'] + drawn_region['x1']) / 2
            center_y = (drawn_region['y0'] + drawn_region['y1']) / 2
            
            # Calculate square coordinates
            half_size = square_size / 2
            square_region = {
                'x0': center_x - half_size,
                'x1': center_x + half_size,
                'y0': center_y - half_size,
                'y1': center_y + half_size
            }
            
            print(f"Square region: x=({square_region['x0']:.1f}, {square_region['x1']:.1f}), y=({square_region['y0']:.1f}, {square_region['y1']:.1f})")
            
            # Extract square region from original image using the compute.py function
            try:
                region_data = extract_region_no_binning(
                    original_data=original_data,
                    binned_data=binned_data,
                    x_range=(square_region['x0'], square_region['x1']),
                    y_range=(square_region['y0'], square_region['y1'])
                )
                region_img = Image.fromarray(region_data.astype(np.uint8))
                print(f"Extracted square region size: {region_img.size} (no binning)")
                return region_img
            except Exception as e:
                print(f"Error extracting square region: {e}")
                # Fallback to entire original image
                region_img = Image.fromarray(original_data.astype(np.uint8))
                return region_img
        
        # If no drawn region, return None - user must explicitly select a region before FFT calculation
        else:
            print("No region selected - user must select a region before FFT calculation")
            return None

    def get_drawn_shapes_from_figure():
        """Get the current drawn shapes from the image display figure."""
        try:
            # This is a placeholder - in a real implementation, we would need to access the current figure state
            # For now, we'll rely on the stored drawn_region in zoom_state
            return None
        except Exception as e:
            print(f"Error getting drawn shapes: {e}")
            return None

    @output
    @render_widget  
    def fft_with_circle():
        from shiny import req
        req(image_data.get() is not None)
        
        # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            # Return None to show nothing when no FFT has been calculated
            return None
        
        print("=== CREATING NEW FFT WIDGET (should only happen on Calc FFT) ===")
        
        # Use the cached FFT image (already has current contrast applied)
        fft_img = cached_fft.copy()
        
        # Convert PIL image to numpy array for Plotly
        fft_arr = np.array(fft_img.convert('L')).astype(np.uint8)

        # Create the FFT figure manually to ensure click events work
        fig = go.Figure()
        
        # Add heatmap for display
        fig.add_trace(go.Heatmap(
            z=fft_arr,
            colorscale="gray",
            showscale=False,
            zmin=0,
            zmax=255,
            hoverinfo="skip",  # Disable hover for heatmap
            opacity=1.0
        ))
        
        # Add transparent scatter overlay to capture clicks and mouse events
        # Create a grid of invisible points that cover the entire FFT
        y_coords, x_coords = np.meshgrid(
            np.arange(0, fft_arr.shape[0], 3),  # Every 3 pixels for performance
            np.arange(0, fft_arr.shape[1], 3),  # Every 3 pixels for performance
            indexing='ij'
        )
        scatter_trace = go.Scatter(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            mode='markers',
            marker=dict(
                size=2,  # Small size for dense grid
                opacity=0,  # Completely transparent
                color='rgba(0,0,0,0)'  # Valid transparent color
            ),
            hoverinfo='skip',  # Enable hover for click events
            showlegend=False,
            hovertemplate='<b>FFT</b><br>x: %{x}<br>y: %{y}<extra></extra>'
        )
        fig.add_trace(scatter_trace)
        

        
        # Hide axes but keep them functional for events
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        # Set layout with square aspect ratio and click events enabled
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="white",
            dragmode='pan',
            title=None,
            clickmode='event',
            hovermode=False
        )
        
        # Force square aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        
        # Note: All shapes (circles, measurements, lattice points) are now handled by 
        # separate overlay effects to avoid re-rendering base FFT on state changes
        

        
        # Note: Lattice points are now handled by separate overlay effect
        # to avoid re-rendering base FFT when mode changes
        
        # Note: Ellipse will be added dynamically via FigureWidget callback when "Fit Ellipse" is clicked
        # This prevents re-rendering the entire FFT figure
        
        # Configure interactive layout
        fig.update_layout(
            height=None,  # Allow natural sizing like original image
            margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins
            autosize=True,
            dragmode='pan',  # Keep pan mode for now, will be updated after circle is drawn
            modebar=dict(
                #add=['drawcircle', 'drawline', 'eraseshape', 'zoom', 'pan', 'reset+autorange'],
                add=[ 'zoom', 'pan', 'reset+autorange'],

                remove=['select2d', 'lasso2d'],
                bgcolor='rgba(255,255,255,0.8)',
                color='black',
                activecolor='red'
            ),
            # Use simple uirevision based only on base FFT trigger to prevent multiple figure creation
            uirevision=f"fft-base-{base_fft_trigger.get()}",
            # Enable click events
            clickmode='event',
            hovermode='closest',
            # Configure newshape for line drawing
            newshape=dict(
                line_color='red',
                line_width=2,
                fillcolor='rgba(255,0,0,0.1)',
                drawdirection='diagonal',
                layer='above'
            ),
        )
        
        
        # Create FigureWidget from the Figure
        fw = FigureWidget(fig)
        
        # Get the scatter trace for click handling
        scatter_trace = fw.data[1]
        


        
        # Define the click callback function
        def update_point(trace, points, selector):
            if points.point_inds:
                # Get the clicked point coordinates
                point_idx = points.point_inds[0]
                click_x = scatter_trace.x[point_idx]
                click_y = scatter_trace.y[point_idx]
                
                # Get current mode dynamically
                current_mode_now = current_mode_storage.get()
                
                # Only handle clicks in Resolution Ring mode
                if current_mode_now == 'Resolution Ring':
                    # Calculate circle centered at image center through click point
                    N = fft_arr.shape[0]
                    cx = cy = N/2
                    r = ((click_x-cx)**2 + (click_y-cy)**2)**0.5
                    
                    # Get current resolution setting
                    resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
                    
                    # Calculate apix from the circle radius
                    if resolution is not None and r > 0:
                        # Use the actual FFT image size instead of hardcoded size
                        fft_image_size = fft_arr.shape[0]  # Should be the same as fft_arr.shape[1] for square image
                        calculated_apix = (r * resolution) / fft_image_size
                        print(f"Circle: center=({cx}, {cy}), radius={r:.2f}")
                        print(f"Major axis: {r:.2f}, Minor axis: {r:.2f} (circle)")
                        print(f"Resolution: {resolution} Å, Calculated Apix: {calculated_apix:.3f} Å/px")
                        #print(f"Debug: r={r}, resolution={resolution}, fft_image_size={fft_image_size}")
                        
                        # Update the apix slider with the calculated value
                        ui.update_slider("apix_slider", value=calculated_apix, session=session)
                        ui.update_text("apix_exact_str", value=f"{calculated_apix:.3f}", session=session)
                    else:
                        print(f"Circle: center=({cx}, {cy}), radius={r:.2f}")
                        print("Could not calculate apix - resolution or radius is invalid")
                    
                    # Create new circle shape
                    new_circle = {
                        'type': 'circle',
                        'x0': cx-r, 'y0': cy-r, 'x1': cx+r, 'y1': cy+r,
                        'line': {'color': 'cyan', 'width': 2},
                        'layer': 'above',
                        'editable': True  # Make the shape editable
                    }
                    
                    # Update FFT state with resolution ring information
                    current_state = fft_state.get()
                    new_state = current_state.copy()
                    new_state['resolution_radius'] = r
                    new_state['resolution_click_x'] = click_x
                    new_state['resolution_click_y'] = click_y
                    fft_state.set(new_state)
                    
                    # Update the figure with the new shape and enable shape editing
                    with fw.batch_update():
                        fw.layout.shapes = [new_circle]
                        fw.layout.dragmode = 'select'  # Enable shape selection/editing
                else: # current_mode_now == 'Lattice Point':
                    print(f"=== LATTICE POINT MODE ===")
                    print(f"Click coordinates: x={click_x}, y={click_y}")
                    
                    # Note: The click_x, click_y are from the 3-pixel grid scatter points
                    # But we want to allow clicking anywhere and snap to a reasonable position
                    # For now, use the clicked coordinates as-is since they're already on the grid
                    snapped_x, snapped_y = click_x, click_y
                    
                    # Store the lattice point in separate storage (doesn't trigger FFT re-render)
                    current_points = lattice_points_storage.get()
                    new_points = current_points + [(snapped_x, snapped_y)]  # Create new list to trigger reactive update
                    lattice_points_storage.set(new_points)
                    
                    print(f"Added lattice point: ({snapped_x}, {snapped_y}). Total points: {len(new_points)}")
                    # print(f"[DEBUG] Current mode storage: {current_mode_storage.get()}")
                    
                    # Note: Green circle display is handled by the reactive effect for lattice_points_storage
                    # No need to add shapes directly here - it will be handled automatically
        
        # Attach the click callback to the scatter trace for backward compatibility
        scatter_trace.on_click(update_point)
        
        # Also attach a general click handler to the entire figure to capture clicks anywhere
        def handle_figure_click(trace, points, selector):
            """Handle clicks anywhere on the figure, not just on scatter points."""
            if hasattr(points, 'xs') and hasattr(points, 'ys') and len(points.xs) > 0:
                # Get the actual click coordinates from the event
                click_x, click_y = points.xs[0], points.ys[0]
                
                # Get current mode dynamically
                current_mode_now = current_mode_storage.get()
                
                # Only handle lattice point mode here (Resolution Ring is handled by scatter trace)
                if current_mode_now == 'Lattice Point':
                    # print(f"=== LATTICE POINT MODE (Figure Click) ===")
                    # print(f"Raw click coordinates: x={click_x}, y={click_y}")
                    
                    # Snap to nearest grid point (multiples of 3)
                    snapped_x = round(click_x / 3) * 3
                    snapped_y = round(click_y / 3) * 3
                    
                    # Store the snapped lattice point
                    current_points = lattice_points_storage.get()
                    new_points = current_points + [(snapped_x, snapped_y)]
                    lattice_points_storage.set(new_points)
                    
                    print(f"Added lattice point (snapped): ({snapped_x}, {snapped_y}). Total points: {len(new_points)}")
        
        # Attach general click handler to the figure widget
        fw.data[0].on_click(handle_figure_click)  # Attach to heatmap trace
        
        # Store the widget for in-place overlay updates (ellipse, etc.)
        fft_widget.set(fw)
        
        # Remove direct on_relayout handler from FigureWidget (was not working)
        # Relayout events will be handled by Shiny's input.fft_with_circle_relayout event
        
        print(f"=== FFT FIGURE CREATED WITH CLICK AND SHAPE UPDATE CALLBACKS ENABLED ===")
        return fw

    # Effect to update FFT heatmap data when contrast changes (without recreating the entire figure)
    @reactive.Effect
    @reactive.event(input.contrast)
    def _():
        """Update FFT heatmap data in-place when contrast changes."""
        #print("Contrast changed - updating FFT heatmap data in-place")
        
        widget = fft_widget.get()
        region = get_current_region()
        
        if widget is not None and region is not None:
            # Generate new FFT image with updated contrast
            fft_img = compute_fft_image_region(region, input.contrast())
            fft_arr = np.array(fft_img.convert('L')).astype(np.uint8)
            
            # Update the heatmap data in-place without recreation
            with widget.batch_update():
                widget.data[0].z = fft_arr
            
            print("FFT heatmap data updated successfully")
        else:
            print("No FFT widget or region available for contrast update")

    # Effect to update overlays when lattice points, mode, or ellipse parameters change
    @reactive.Effect
    @reactive.event(lattice_points_storage, current_mode_storage, ellipse_params_storage, tuned_markers_storage, tuned_resolution_radius)
    def _():
        """Update FFT overlays when lattice points, tuned markers, tuned resolution, or mode changes."""
        # print("Updating FFT overlays for lattice points or mode change")
        
        widget = fft_widget.get()
        if widget is None:
            # print("[DEBUG] No FFT widget available for lattice point overlay")
            return
        
        lattice_points = lattice_points_storage.get()
        tuned_markers = tuned_markers_storage.get()
        current_mode = current_mode_storage.get()
        # print(f"[DEBUG] Lattice overlay effect - points: {len(lattice_points)}, tuned: {len(tuned_markers)}, mode: {current_mode}")
        
        # Get current shapes and filter out lattice point circles and ellipses
        current_shapes = list(widget.layout.shapes) if widget.layout.shapes else []
        preserved_shapes = []
        for s in current_shapes:
            # Check if this is a lattice point circle (green, width 2)
            is_lattice_circle = (
                hasattr(s, 'type') and s.type == 'circle' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'green' and
                hasattr(s.line, 'width') and s.line.width == 2
            )
            # Check if this is a tuned marker crosshair (red line, width 2)
            is_tuned_crosshair = (
                hasattr(s, 'type') and s.type == 'line' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'red' and
                hasattr(s.line, 'width') and s.line.width == 2
            )
            # Check if this is a tuned resolution ring (red circle, width 2)
            is_tuned_resolution_ring = (
                hasattr(s, 'type') and s.type == 'circle' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'red' and
                hasattr(s.line, 'width') and s.line.width == 2
            )
            # Check if this is a user-clicked resolution ring (cyan circle, width 2)
            is_user_resolution_ring = (
                hasattr(s, 'type') and s.type == 'circle' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'cyan' and
                hasattr(s.line, 'width') and s.line.width == 2
            )
            # Check if this is a fitted ellipse shape (legacy - ellipses are now traces, not shapes)
            is_fitted_ellipse_shape = (
                hasattr(s, 'type') and s.type == 'path' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'red'
            )
            # Always remove fitted ellipse shapes (legacy cleanup) and lattice/tuned markers based on mode
            should_remove = is_fitted_ellipse_shape  # Always remove legacy ellipse shapes
            if current_mode == 'Resolution Ring':
                # Remove lattice circles and tuned crosshairs when in Ring mode, but keep cyan and red resolution rings
                should_remove = should_remove or is_lattice_circle or is_tuned_crosshair
                # Remove tuned resolution rings to re-add them fresh (but keep user-clicked cyan rings)
                should_remove = should_remove or is_tuned_resolution_ring
            else:  # Lattice Point mode
                # Remove lattice circles, tuned crosshairs, tuned resolution rings, and user resolution rings (lattice items will be re-added)
                should_remove = should_remove or is_lattice_circle or is_tuned_crosshair or is_tuned_resolution_ring or is_user_resolution_ring
            
            if not should_remove:
                preserved_shapes.append(s)
        
        # Add lattice points if in Lattice Point mode
        if current_mode == 'Lattice Point':
            # print(f"[DEBUG] Adding {len(lattice_points)} green circles for lattice points")
            for pt in lattice_points:
                x, y = pt[0], pt[1]
                # Add green circle for each lattice point
                lattice_circle = {
                    'type': 'circle',
                    'x0': x-8, 'y0': y-8, 'x1': x+8, 'y1': y+8,
                    'line': {'color': 'green', 'width': 2},
                    'layer': 'above',
                    'editable': False
                }
                preserved_shapes.append(lattice_circle)
                # print(f"[DEBUG] Added green circle at ({x}, {y})")
                
            # Add tuned markers as red crosshairs
            # print(f"[DEBUG] Adding {len(tuned_markers)} red crosshairs for tuned markers")
            for pt in tuned_markers:
                x, y = pt[0], pt[1]
                crosshair_size = 6  # Half-length of crosshair arms
                
                # Add horizontal line of crosshair
                horizontal_line = {
                    'type': 'line',
                    'x0': x - crosshair_size, 'y0': y, 'x1': x + crosshair_size, 'y1': y,
                    'line': {'color': 'red', 'width': 2},
                    'layer': 'above',
                    'editable': False
                }
                preserved_shapes.append(horizontal_line)
                
                # Add vertical line of crosshair
                vertical_line = {
                    'type': 'line',
                    'x0': x, 'y0': y - crosshair_size, 'x1': x, 'y1': y + crosshair_size,
                    'line': {'color': 'red', 'width': 2},
                    'layer': 'above',
                    'editable': False
                }
                preserved_shapes.append(vertical_line)
                # print(f"[DEBUG] Added red crosshair at ({x}, {y})")
        
        # Add tuned resolution ring if in Resolution Ring mode and available
        if current_mode == 'Resolution Ring':
            tuned_radius = tuned_resolution_radius.get()
            if tuned_radius is not None:
                # Get FFT image center (same as original resolution ring logic)
                cached_fft = cached_fft_image.get()
                if cached_fft is not None:
                    fft_array = np.array(cached_fft)
                    N = fft_array.shape[0]
                    cx = cy = N/2
                    
                    # Create red tuned resolution ring
                    tuned_ring = {
                        'type': 'circle',
                        'x0': cx - tuned_radius, 'y0': cy - tuned_radius, 
                        'x1': cx + tuned_radius, 'y1': cy + tuned_radius,
                        'line': {'color': 'red', 'width': 2},
                        'layer': 'above',
                        'editable': False
                    }
                    preserved_shapes.append(tuned_ring)
                    # print(f"[DEBUG] Added red tuned resolution ring at radius {tuned_radius:.3f}")
        # else:
            # print(f"[DEBUG] Not adding lattice circles - mode is: {current_mode}")
        
        # Note: Fitted ellipse is now handled as a TRACE by fit_markers function, not as a shape
        # This prevents duplicate ellipses (one trace + one shape)
        
        # Update shapes in-place
        with widget.batch_update():
            widget.layout.shapes = preserved_shapes
            
            # Only handle ellipse traces for mode switching - remove ellipse_fit trace when switching to Ring mode
            if current_mode == 'Resolution Ring':
                # Remove any ellipse_fit traces

                
                # Use index-based removal (more reliable than reassigning data)
                ellipse_indices = []
                for i, trace in enumerate(widget.data):
                    if hasattr(trace, 'name') and trace.name and ('ellipse_fit' in trace.name):
                        ellipse_indices.append(i)
                
                # Remove ellipse traces from the end to avoid index shifting
                for i in reversed(ellipse_indices):
                    widget.data = widget.data[:i] + widget.data[i+1:]
        
        # print(f"Updated FFT overlays - lattice points: {len(lattice_points)}, mode: {current_mode}")

    # Effect to update FFT overlays when fft_state changes (circles, measurements)
    @reactive.Effect
    @reactive.event(fft_state, current_mode_storage)
    def _():
        """Update FFT overlays when drawn circles or measurements change."""
        # print("Updating FFT overlays for drawn circles or measurements")
        
        widget = fft_widget.get()
        if widget is None:
            return
        
        current_state = fft_state.get()
        
        # Get current mode to filter shapes appropriately
        current_mode = current_mode_storage.get()
        
        # Get current shapes and filter out previous drawn circles and measurements
        current_shapes = list(widget.layout.shapes) if widget.layout.shapes else []
        preserved_shapes = []
        for s in current_shapes:
            # Keep lattice point circles (green, width 2)
            is_lattice_circle = (
                hasattr(s, 'type') and s.type == 'circle' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'green' and
                hasattr(s.line, 'width') and s.line.width == 2
            )
            # Check if this is a fitted ellipse (path type with red color)
            is_fitted_ellipse = (
                hasattr(s, 'type') and s.type == 'path' and
                hasattr(s, 'line') and s.line and
                hasattr(s.line, 'color') and s.line.color == 'red'
            )
            # Check if this is a drawn circle or measurement line
            is_drawn_circle_or_line = (
                hasattr(s, 'type') and s.type in ['circle', 'line'] and
                not is_lattice_circle
            )
            
            # Decide what to preserve based on current mode
            should_preserve = False
            if current_mode == 'Resolution Ring':
                # In Ring mode: preserve drawn circles/lines but not lattice circles or ellipses
                should_preserve = is_drawn_circle_or_line and not is_fitted_ellipse
            else:  # Lattice Point mode
                # In Lattice Point mode: preserve lattice circles, ellipses, and drawn circles/lines
                should_preserve = not is_drawn_circle_or_line or is_lattice_circle or is_fitted_ellipse
            
            if should_preserve:
                preserved_shapes.append(s)
        
        # Add drawn circles from state
        drawn_circles = current_state.get('drawn_circles', [])
        for circle_data in drawn_circles:
            preserved_shapes.append(circle_data)
        
        # Add current measurement if available
        current_measurement = current_state.get('current_measurement')
        if current_measurement is not None:
            # Add the line shape
            line_shape = {
                'type': 'line',
                'x0': current_measurement['x0'], 'y0': current_measurement['y0'],
                'x1': current_measurement['x1'], 'y1': current_measurement['y1'],
                'line': {'color': 'red', 'width': 2},
                'layer': 'above'
            }
            preserved_shapes.append(line_shape)
        
        # Update shapes in-place
        with widget.batch_update():
            widget.layout.shapes = preserved_shapes
            
            # Handle ellipse traces based on current mode
            if current_mode == 'Resolution Ring':
                # Remove any ellipse_fit traces when in Ring mode

                
                # Use index-based removal (more reliable than reassigning data)
                ellipse_indices = []
                for i, trace in enumerate(widget.data):
                    if hasattr(trace, 'name') and trace.name and ('ellipse_fit' in trace.name):
                        ellipse_indices.append(i)
                
                # Remove ellipse traces from the end to avoid index shifting
                for i in reversed(ellipse_indices):
                    widget.data = widget.data[:i] + widget.data[i+1:]
            
            # Update annotations for measurements
            widget.layout.annotations = []
            if current_measurement is not None:
                mid_x = (current_measurement['x0'] + current_measurement['x1']) / 2
                mid_y = (current_measurement['y0'] + current_measurement['y1']) / 2
                
                widget.layout.annotations = [{
                    'x': mid_x,
                    'y': mid_y,
                    'text': f"{current_measurement['distance']:.1f} px",
                    'showarrow': False,
                    'font': {'color': 'red', 'size': 12},
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': 'red',
                    'borderwidth': 1
                }]
        
        # print(f"Updated FFT overlays - circles: {len(drawn_circles)}, measurement: {current_measurement is not None}")

    # Effect to autoscale FFT plot when triggered by Calc FFT (not contrast changes)
    @reactive.Effect
    @reactive.event(autoscale_trigger)
    def _():
        """Autoscale FFT plot when triggered by Calc FFT button."""
        print("Autoscaling FFT plot to fit card")
        
        widget = fft_widget.get()
        if widget is not None:
            with widget.batch_update():
                # Reset zoom to show full image
                widget.layout.xaxis.autorange = True
                widget.layout.yaxis.autorange = True
                # Ensure square aspect ratio for FFT images
                widget.layout.xaxis.scaleanchor = "y"
                widget.layout.xaxis.scaleratio = 1
                # Set margins to maximize image size within card
                widget.layout.margin = dict(l=40, r=40, t=40, b=40)
            print("✅ FFT plot auto-scaled to fit card with square aspect ratio")
        else:
            print("No FFT widget available for autoscaling")

    # Restore the Shiny relayout event handler for fft_with_circle
    @reactive.Effect
    @reactive.event(input.fft_with_circle_relayout)
    def _on_fft_relayout():
        evt = input.fft_with_circle_relayout()
        print(f"=== SHINY RELAYOUT EVENT RECEIVED ===")
        print(f"Raw event: {evt}")
        if not evt:
            #print("No relayout event data (evt is None or empty)")
            return
        if 'shapes' not in evt:
            #print("No 'shapes' key in relayout event")
            return
        shapes = evt['shapes']
        print(f"Shapes: {shapes}")
        if shapes and len(shapes) > 0:
            latest_shape = shapes[-1]
            shape_type = latest_shape.get('type')
            #print(f"Latest shape type: {shape_type}")
            if shape_type == 'line':
                x0 = latest_shape.get('x0')
                y0 = latest_shape.get('y0')
                x1 = latest_shape.get('x1')
                y1 = latest_shape.get('y1')
                if all(coord is not None for coord in [x0, y0, x1, y1]):
                    length = math.hypot(x1 - x0, y1 - y0)
                    print(f"Line coordinates: ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f})")
                    print(f"Distance: {length:.1f} pixels")
                    line_data = {
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'distance': length
                    }
                    current_state = fft_state.get()
                    new_state = current_state.copy()
                    new_state['current_measurement'] = line_data
                    fft_state.set(new_state)
                    print(f"Stored measurement: {length:.1f} pixels")
                else:
                    print("Invalid line coordinates")
            else:
                print(f"Shape type is not 'line': {shape_type}")
        else:
            print("No shapes or shapes list is empty; clearing measurement")
            current_state = fft_state.get()
            new_state = current_state.copy()
            new_state['current_measurement'] = None
            fft_state.set(new_state)
        print(f"=== END SHINY RELAYOUT EVENT ===")

    @reactive.calc
    def fft_1d_data():
        """Calculate the data needed for the 1D FFT plot."""
        from shiny import req
        
        req(image_data.get() is not None)

        # Use stored FFT calculation state instead of current region/apix
        # This prevents replotting when regions are drawn or apix changes before "Calc FFT"
        calc_state = fft_calculation_state.get()
        if calc_state['region'] is None:
            return None

        return compute_fft_1d_data(
            region=calc_state['region'],
            apix=calc_state['apix'],
            use_mean_profile=input.use_mean_profile(),
            log_y=input.log_y(),
            smooth=input.smooth(),
            window_size=input.window_size(),
            detrend=input.detrend(),
            resolution_type=calc_state['resolution_type'],
            custom_resolution=calc_state['custom_resolution']
        )
    
    @render_plotly("fft_1d_plot")
    def fft_1d_plot():
        # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            # Return None to show nothing when no FFT has been calculated
            return None
        
        # Get the calculated plot data
        plot_data = fft_1d_data()
        if plot_data is None:
            return go.Figure()
        
        # Use stored FFT calculation state instead of current region/zoom/resolution
        # This prevents replotting when regions are drawn or parameters change
        calc_state = fft_calculation_state.get()
        if calc_state['region'] is None:
            return go.Figure()
        
        # Get stored region, zoom, and resolution from calculation state
        region = calc_state['region']
        zoom = plot_zoom.get()  # Keep zoom state for interactive zoom/pan
        resolution, _ = get_resolution_info(calc_state['resolution_type'], calc_state['custom_resolution'])
        
        # Create plotly figure using the original data (no filtering)
        fig = create_fft_1d_plotly_figure(
            plot_data=plot_data,
            resolution=resolution,
            region=region,
            size=size,
            zoom_state=zoom
        )
        
        # Create FigureWidget from the Figure and return it
        fw = FigureWidget(fig)
        
        # Store the widget for in-place updates
        fft_1d_widget.set(fw)
        
        return fw



    # @output
    # @render.text
    # def lattice_points_data():
    #     """Hidden output to expose lattice points data for persistence."""
    #     state = fft_state.get()
    #     if state['mode'] == 'Lattice Point':
    #         # Get points from separate storage
    #         points = lattice_points_storage.get()
    #         if points:
    #             # Return lattice points as JSON-like string for easy parsing
    #             points_str = ";".join([f"{x},{y}" for x, y in points])
    #             return f"Lattice Points: {points_str}"
    #     return "Lattice Points: None"

    # @output
    # @render.text
    # def lattice_points_count():
    #     """Hidden output to expose lattice points count for debugging."""
    #     state = fft_state.get()
    #     if state['mode'] == 'Lattice Point':
    #         points = lattice_points_storage.get()
    #         return f"Lattice Points Count: {len(points)}"
    #     return "Lattice Points Count: 0"

    @output
    @render.text
    def tilt_output():
        """Display tilt estimation results."""
        # Check dual tilt storages first
        green_tilt = tilt_info_green_storage.get()
        red_tilt = tilt_info_red_storage.get()
        
        results = []
        
        # Format green ellipse results
        if green_tilt is not None:
            small_axis, large_axis, tilt_angle, untilted_apix = green_tilt
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            apix_str = ""
            if untilted_apix is not None:
                apix_str = f", Apix: {untilted_apix:.3f} Å/px"
            
            green_result = (f"🟢 User-clicked: Minor axis: {small_axis:.2f}, "
                           f"Major axis: {large_axis:.2f}, "
                           f"Tilt angle: {tilt_angle_degrees:.2f}°"
                           f"{apix_str}")
            results.append(green_result)
        
        # Format red ellipse results
        if red_tilt is not None:
            small_axis, large_axis, tilt_angle, untilted_apix = red_tilt
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            apix_str = ""
            if untilted_apix is not None:
                apix_str = f", Apix: {untilted_apix:.3f} Å/px"
            
            red_result = (f"🔴 Fine-tuned: Minor axis: {small_axis:.2f}, "
                         f"Major axis: {large_axis:.2f}, "
                         f"Tilt angle: {tilt_angle_degrees:.2f}°"
                         f"{apix_str}")
            results.append(red_result)
        
        # If we have dual results, return them
        if results:
            return "\n".join(results)
        
        # Fallback to legacy single tilt info
        tilt_info = tilt_info_storage.get()
        if tilt_info is not None:
            # Check if we have the new format with untilted apix
            if len(tilt_info) >= 4:
                small_axis, large_axis, tilt_angle, untilted_apix = tilt_info
                tilt_angle_degrees = math.degrees(tilt_angle)
                
                apix_str = ""
                if untilted_apix is not None:
                    apix_str = f", Estimated untilted apix: {untilted_apix:.3f} Å/px"
                
                return (f"Minor axis: {small_axis:.2f}, "
                       f"Major axis: {large_axis:.2f}, "
                       f"Tilt angle: {tilt_angle_degrees:.2f}°"
                       f"{apix_str}")
            else:
                # Legacy format
                small_axis, large_axis, tilt_angle = tilt_info
                tilt_angle_degrees = math.degrees(tilt_angle)
                
                # Calculate apix from large axis
                resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
                apix_str = ""
                if resolution is not None and large_axis > 0:
                    calculated_apix = (large_axis * resolution) / size
                    if 0.01 <= calculated_apix <= 6.0:
                        apix_str = f", Apix: {calculated_apix:.3f} Å/px"
                
                return (f"Small axis: {small_axis:.2f}, "
                       f"Large axis: {large_axis:.2f}, "
                       f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°"
                       f"{apix_str}")
        
        # Fallback: check fft_state for legacy tilt info
        state = fft_state.get()
        if state['tilt_info'] is not None:
            # Legacy format from fft_state
            small_axis, large_axis, tilt_angle = state['tilt_info']
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            # Calculate apix from large axis
            resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
            apix_str = ""
            if resolution is not None and large_axis > 0:
                calculated_apix = (large_axis * resolution) / size
                if 0.01 <= calculated_apix <= 6.0:
                    apix_str = f", Apix: {calculated_apix:.3f} Å/px"
            
            return (f"Small axis: {small_axis:.2f}, "
                   f"Large axis: {large_axis:.2f}, "
                   f"Estimated Tilt Angle: {tilt_angle_degrees:.2f}°"
                   f"{apix_str}")
        elif state['ellipse_params'] is not None:
            # Show ellipse parameters when fitted but not yet estimated for tilt
            a, b, theta = state['ellipse_params']
            theta_degrees = math.degrees(theta)
            
            # Calculate apix from larger axis
            resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
            apix_str = ""
            if resolution is not None:
                large_axis = max(a, b)
                calculated_apix = (large_axis * resolution) / size
                if 0.01 <= calculated_apix <= 6.0:
                    apix_str = f", Estimated Apix: {calculated_apix:.3f} Å/px"
            
            return (f"Ellipse fitted: a={a:.1f}, b={b:.1f}, θ={theta_degrees:.1f}°"
                   f"{apix_str}")
        return ""

    @output
    @render.data_frame
    def region_table():
        """Render the region analysis table."""
        return render.DataGrid(
            region_table_data.get(),
            editable=False,
            selection_mode="row",
            width="100%",
            height="350px"
        )

    # Add reactive value to store the Apix-centered plot FigureWidget for in-place updates
    apix_centered_widget = reactive.Value(None)

    @output
    @render_widget
    def apix_centered_by_nominal_plot():
        """Live vertical scatter: Apix centered by nominal value, from region_table_data, using FigureWidget for in-place updates."""
        df = region_table_data.get().copy()
        if df is None or df.empty or 'Filename' not in df.columns or 'Apix' not in df.columns or 'Nominal' not in df.columns:
            #print("[DEBUG] DataFrame is empty or missing columns.")
            fw = FigureWidget()
            apix_centered_widget.set(fw)
            return fw
        # Ensure Apix is float
        try:
            df['Apix'] = pd.to_numeric(df['Apix'], errors='coerce')
        except Exception:
            df['Apix'] = None
        # Ensure Nominal is float, fill missing values by extracting from filename
        try:
            df['Nominal'] = pd.to_numeric(df['Nominal'], errors='coerce')
        except Exception:
            df['Nominal'] = None
        # Fill missing Nominal values with current textbox value
        missing_nominal = df['Nominal'].isna()
        if missing_nominal.any():
            textbox_nominal = float(input.nominal_apix())
            df.loc[missing_nominal, 'Nominal'] = textbox_nominal
        
        #print("[DEBUG] DataFrame after ensuring numeric types:\n", df[['Filename', 'Apix', 'Nominal']])
        # Drop rows with missing data
        df = df.dropna(subset=['Apix', 'Nominal'])
        if df.empty:
            #print("[DEBUG] DataFrame is empty after dropping missing Apix/Nominal.")
            fw = FigureWidget()
            apix_centered_widget.set(fw)
            return fw
        # Sort Nominal for plotting
        try:
            nominal_order = sorted(df['Nominal'].dropna().unique())
        except Exception:
            nominal_order = list(df['Nominal'].dropna().unique())
        # Apix - Nominal
        df['Apix_centered_by_nominal'] = df.apply(lambda row: row['Apix'] - row['Nominal'], axis=1)
        #print("[DEBUG] DataFrame before plotting:\n", df[['Nominal', 'Apix', 'Apix_centered_by_nominal']])
        fig = go.Figure()
        # Light blue vertical scatter for each group
        for nominal in nominal_order:
            group = df[df['Nominal'] == nominal]
            fig.add_trace(go.Scatter(
                x=[nominal]*len(group),
                y=group['Apix_centered_by_nominal'],
                mode='markers',
                marker=dict(color='lightblue', size=8),
                name='Apix values',
                showlegend=bool(nominal == nominal_order[0])
            ))
        # Red dot at y=0 for each group (Nominal - Nominal)
        fig.add_trace(go.Scatter(
            x=nominal_order,
            y=[0]*len(nominal_order),
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name='Nominal - Nominal',
            showlegend=True
        ))
        # Blue dot at (Mean - Nominal) for each group
        blue_dots = []
        for nominal in nominal_order:
            group = df[df['Nominal'] == nominal]
            if len(group) == 0:
                continue
            group_mean = group['Apix'].mean()
            blue_dots.append({'Nominal': nominal, 'y': group_mean - nominal})
        fig.add_trace(go.Scatter(
            x=[d['Nominal'] for d in blue_dots],
            y=[d['y'] for d in blue_dots],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            name='Actual Mean - Nominal',
            showlegend=True
        ))
        fig.update_layout(
            title='Apix Centered by Nominal Value',
            xaxis_title='Nominal Value (Å)',
            yaxis_title='Apix - Nominal (Å/px)',
            xaxis=dict(type='category', categoryorder='array', categoryarray=nominal_order),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                itemsizing='constant'
            ),
            margin=dict(l=20, r=20, t=40, b=90),
            autosize=True,
            height=None
        )
        fw = FigureWidget(fig)
        apix_centered_widget.set(fw)
        return fw

    # Reactive effect to update the Apix-centered plot in-place when the table changes
    @reactive.Effect
    @reactive.event(region_table_data)
    def _():
        widget = apix_centered_widget.get()
        if widget is not None:
            import re
            df = region_table_data.get().copy()
            if df is None or df.empty or 'Filename' not in df.columns or 'Apix' not in df.columns or 'Nominal' not in df.columns:
                #print("[DEBUG] DataFrame is empty or missing columns (effect update).")
                with widget.batch_update():
                    # Clear all existing traces
                    while len(widget.data) > 0:
                        widget.data = widget.data[:-1]
                    widget.layout.title = 'Vertical Scatter: Apix Centered by Nominal Value'
                    widget.layout.xaxis.title = 'Nominal Value (Å)'
                    widget.layout.yaxis.title = 'Apix - Nominal (Å/px)'
                return
            try:
                df['Apix'] = pd.to_numeric(df['Apix'], errors='coerce')
            except Exception:
                df['Apix'] = None
            # Ensure Nominal is float, fill missing values by extracting from filename
            try:
                df['Nominal'] = pd.to_numeric(df['Nominal'], errors='coerce')
            except Exception:
                df['Nominal'] = None
            # Fill missing Nominal values with current textbox value
            missing_nominal = df['Nominal'].isna()
            if missing_nominal.any():
                textbox_nominal = float(input.nominal_apix())
                df.loc[missing_nominal, 'Nominal'] = textbox_nominal
            
            #print("[DEBUG] DataFrame after ensuring numeric types (effect):\n", df[['Filename', 'Apix', 'Nominal']])
            df = df.dropna(subset=['Apix', 'Nominal'])
            if df.empty:
                #print("[DEBUG] DataFrame is empty after dropping missing Apix/Nominal (effect).")
                with widget.batch_update():
                    # Clear all existing traces
                    while len(widget.data) > 0:
                        widget.data = widget.data[:-1]
                    widget.layout.title = 'Vertical Scatter: Apix Centered by Nominal Value'
                    widget.layout.xaxis.title = 'Nominal Value (Å)'
                    widget.layout.yaxis.title = 'Apix - Nominal (Å/px)'
                return
            try:
                nominal_order = sorted(df['Nominal'].dropna().unique())
            except Exception:
                nominal_order = list(df['Nominal'].dropna().unique())
            df['Apix_centered_by_nominal'] = df.apply(lambda row: row['Apix'] - row['Nominal'], axis=1)
            #print("[DEBUG] DataFrame before plotting (effect):\n", df[['Nominal', 'Apix', 'Apix_centered_by_nominal']])
            traces = []
            for nominal in nominal_order:
                group = df[df['Nominal'] == nominal]
                traces.append(go.Scatter(
                    x=[nominal]*len(group),
                    y=group['Apix_centered_by_nominal'],
                    mode='markers',
                    marker=dict(color='lightblue', size=8),
                    name='Apix values',
                    showlegend=bool(nominal == nominal_order[0])
                ))
            traces.append(go.Scatter(
                x=nominal_order,
                y=[0]*len(nominal_order),
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Nominal - Nominal',
                showlegend=True
            ))
            blue_dots = []
            for nominal in nominal_order:
                group = df[df['Nominal'] == nominal]
                if len(group) == 0:
                    continue
                group_mean = group['Apix'].mean()
                blue_dots.append({'Nominal': nominal, 'y': group_mean - nominal})
            traces.append(go.Scatter(
                x=[d['Nominal'] for d in blue_dots],
                y=[d['y'] for d in blue_dots],
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle'),
                name='Actual Mean - Nominal',
                showlegend=True
            ))
            with widget.batch_update():
                # Remove all existing traces
                while len(widget.data) > 0:
                    widget.data = widget.data[:-1]
                # Add new traces one by one
                for trace in traces:
                    widget.add_trace(trace)
                widget.layout.title = 'Vertical Scatter: Apix Centered by Nominal Value'
                widget.layout.xaxis.title = 'Nominal Value (Å)'
                widget.layout.yaxis.title = 'Apix - Nominal (Å/px)'
                widget.layout.xaxis.type = 'category'
                widget.layout.xaxis.categoryorder = 'array'
                widget.layout.xaxis.categoryarray = nominal_order
                widget.layout.legend = dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    itemsizing='constant'
                )
                widget.layout.margin = dict(l=20, r=20, t=40, b=90)
                widget.layout.autosize = True
                widget.layout.height = None

    @reactive.Effect
    @reactive.event(input.random_generate)
    def _():
        """Generate random regions from the image and analyze them."""
        try:
            # Check if we have an image loaded
            original_data = original_image_data.get()
            if original_data is None:
                print("No image loaded for random generation")
                return
            
            filename = image_filename.get() or "Unknown"
            num_regions = input.random_count()
            
            if num_regions <= 0:
                print("Number of regions must be greater than 0")
                return
            
            # Calculate region size based on percentage of image
            img_height, img_width = original_data.shape
            region_size_percent = input.region_size_percent()
            region_size = int(min(img_width, img_height) * region_size_percent)
            
            print(f"=== GENERATING {num_regions} RANDOM {region_size}x{region_size} REGIONS ===")
            print(f"Original image shape: {original_data.shape}")
            print(f"Region size percentage: {region_size_percent:.1f} ({region_size}x{region_size} pixels)")
            print(f"Filename: {filename}")
            
            # Check if image is large enough for calculated region size
            if img_height < region_size or img_width < region_size:
                print(f"Image too small ({img_width}x{img_height}) for {region_size}x{region_size} regions")
                return
            
            # Calculate valid coordinate ranges (ensure regions stay within boundaries)
            # Convert to Python integers to avoid numpy.float64 issues
            max_x = int(img_width - region_size)
            max_y = int(img_height - region_size)
            
            print(f"Valid coordinate ranges: x=[0, {max_x}], y=[0, {max_y}]")
            
            # Generate random regions and analyze each one
            import random
            successful_regions = 0
            
            for i in range(num_regions):
                try:
                    # Generate random top-left coordinates (ensure integers)
                    x0 = int(random.randint(0, max_x))
                    y0 = int(random.randint(0, max_y))
                    x1 = int(x0 + region_size)
                    y1 = int(y0 + region_size)
                    
                    print(f"\nRegion {i+1}/{num_regions}: x=[{x0}, {x1}], y=[{y0}, {y1}]")
                    print(f"Types: x0={type(x0)}, y0={type(y0)}, x1={type(x1)}, y1={type(y1)}")
                    
                    # Extract the region from original image with explicit type checking
                    try:
                        region_data = original_data[y0:y1, x0:x1]
                        print(f"Region data shape: {region_data.shape}, dtype: {region_data.dtype}")
                        
                        # Ensure data is in correct format for PIL
                        region_data_clean = region_data.astype(np.uint8)
                        region_img = Image.fromarray(region_data_clean)
                        
                        print(f"Extracted region size: {region_img.size}")
                        
                    except Exception as e:
                        print(f"Error extracting region: {e}")
                        print(f"Coordinate types: x0={type(x0)}, y0={type(y0)}")
                        raise e
                    
                    # Compute 1D FFT radial profile with detrending
                    try:
                        # Convert current apix to float to avoid numpy type issues
                        current_apix = float(get_apix())
                        current_resolution_type = str(input.resolution_type())
                        current_custom_resolution = float(input.custom_resolution()) if input.custom_resolution() else None
                        
                        plot_data = compute_fft_1d_data(
                            region=region_img,
                            apix=current_apix,
                            use_mean_profile=False,  # Use standard radial average
                            log_y=False,  # Use linear scale
                            smooth=False,  # No smoothing
                            window_size=int(3),  # Ensure integer
                            detrend=True,  # Enable detrending as requested
                            resolution_type=current_resolution_type,
                            custom_resolution=current_custom_resolution
                        )
                        
                    except Exception as e:
                        print(f"Error in compute_fft_1d_data: {e}")
                        raise e
                    
                    if plot_data is None:
                        print(f"Failed to compute FFT for region {i+1}")
                        continue
                    
                    # Find the maximum in the detrended signal
                    x_data = plot_data['x_data']
                    y_data = plot_data['y_data']
                    
                    if len(y_data) == 0:
                        print(f"No data points for region {i+1}")
                        continue
                    
                    max_idx = np.argmax(y_data)
                    fft_max_x = x_data[max_idx]
                    fft_max_y = y_data[max_idx]
                    
                    print(f"Found maximum at x={fft_max_x:.3f}, y={fft_max_y:.3f}")
                    
                    # Calculate apix from the maximum position
                    resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
                    if resolution is not None and fft_max_x > 0:
                        calculated_apix = (fft_max_x * resolution) / region_size
                        
                        if 0.01 <= calculated_apix <= 6.0:
                            print(f"Calculated apix: {calculated_apix:.3f} Å/px")
                            
                            # Create table entry
                            region_location = f"x:{x0}–{x1}, y:{y0}–{y1}"
                            region_size_str = f"{region_size}×{region_size} px"
                            
                            # Get nominal value from textbox
                            nominal_value = float(input.nominal_apix())
                            
                            new_row = pd.DataFrame({
                                'Filename': [filename],
                                'Region Size': [region_size_str],
                                'Region Location': [region_location],
                                'Apix': [f"{calculated_apix:.3f}"],
                                'Nominal': [nominal_value]
                            })
                            
                            # Add to existing table data
                            current_data = region_table_data.get()
                            updated_data = pd.concat([current_data, new_row], ignore_index=True)
                            region_table_data.set(updated_data)
                            
                            successful_regions += 1
                            print(f"Added region {i+1} to table: {filename}, {region_size_str}, {region_location}, {calculated_apix:.3f}, {nominal_value}")
                            
                        else:
                            print(f"Calculated apix {calculated_apix:.3f} is outside valid range [0.01, 6.0]")
                    else:
                        print(f"Could not calculate apix for region {i+1}")
                        
                except Exception as e:
                    print(f"Error processing region {i+1}: {e}")
                    continue
            
            print(f"\n=== RANDOM GENERATION COMPLETE ===")
            print(f"Successfully analyzed {successful_regions}/{num_regions} regions")
            print(f"Total table entries: {len(region_table_data.get())}")
            
        except Exception as e:
            print(f"Error in random generation: {e}")
            import traceback
            traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.delete_selected)
    def _():
        """Delete selected rows from the region analysis table."""
        try:
            # Get the selected rows from the data grid
            selected_rows = input.region_table_selected_rows()
            
            if not selected_rows:
                print("No rows selected for deletion")
                return
            
            # Get current data
            current_data = region_table_data.get()
            
            if len(current_data) == 0:
                print("No data to delete")
                return
            
            # Convert selected rows to a list if it's not already
            if isinstance(selected_rows, int):
                selected_rows = [selected_rows]
            
            # Sort indices in descending order to avoid index shifting issues
            selected_indices = sorted(selected_rows, reverse=True)
            
            # Remove selected rows
            updated_data = current_data.copy()
            for idx in selected_indices:
                if 0 <= idx < len(updated_data):
                    updated_data = updated_data.drop(index=idx)
            
            # Reset index after deletion
            updated_data = updated_data.reset_index(drop=True)
            
            # Update the reactive value
            region_table_data.set(updated_data)
            
            print(f"Deleted {len(selected_indices)} row(s) from region analysis table")
            
        except Exception as e:
            print(f"Error deleting selected rows: {e}")
            import traceback
            traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.clear_table)
    def _():
        """Clear all entries from the region analysis table."""
        empty_df = pd.DataFrame({
            'Filename': [],
            'Region Size': [],
            'Region Location': [],
            'Apix': [],
            'Nominal': []
        })
        region_table_data.set(empty_df)
        print("Region analysis table cleared")

    @render.download(
        filename=lambda: generate_csv_filename()
    )
    def download_csv():
        """Download the region analysis table as CSV."""
        current_data = region_table_data.get()
        if len(current_data) == 0:
            # Yield empty CSV if no data
            yield "Filename,Region Size,Region Location,Apix,Nominal\n"
        else:
            # Generate and yield CSV content
            csv_content = current_data.to_csv(index=False)
            print(f"CSV download initiated: {len(current_data)} rows")
            yield csv_content

    def generate_csv_filename():
        """Generate a descriptive filename for CSV export."""
        from datetime import datetime
        import os
        
        # Get current image filename if available
        current_filename = image_filename.get()
        if current_filename:
            # Remove extension and use as base name
            base_name = os.path.splitext(current_filename)[0]
        else:
            base_name = "magnification_analysis"
        
        # Add timestamp and row count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_data = region_table_data.get()
        row_count = len(current_data)
        
        # Create descriptive filename
        return f"{base_name}_analysis_{row_count}regions_{timestamp}.csv"




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
        # Note: Skip clearing if resolution ring was just clicked (to preserve cyan circle)
        current_state = fft_state.get()
        if current_state['mode'] == 'Resolution Ring' and current_state.get('resolution_radius') is None:
            # Only clear if there's no active resolution ring (i.e., not just clicked)
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
            # Switching to Resolution Ring: clear lattice points, ellipse, and tilt info
            new_state['mode'] = 'Resolution Ring'
            new_state['lattice_points'] = []
            new_state['ellipse_params'] = None
            
            new_state['tilt_info'] = None
            # Also clear the separate lattice points storage
            lattice_points_storage.set([])
            # Also clear the separate tilt info storage
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the dual tilt info storages
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the separate ellipse params storage
            ellipse_params_storage.set(None)
            # Also clear the tuned markers storage
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            # Also clear the tuned resolution ring storage
            tuned_resolution_radius.set(None)
            #new_state['drawn_circles'] = []
            # Update the separate mode storage
            current_mode_storage.set('Resolution Ring')
        elif input.label_mode() == "Lattice Point":
            # Switching to Lattice Point: clear resolution radius, click coordinates, and tilt info
            new_state['mode'] = 'Lattice Point'
            new_state['resolution_radius'] = None
            new_state['resolution_click_x'] = None
            new_state['resolution_click_y'] = None
            new_state['ellipse_params'] = None
            new_state['tilt_info'] = None
            # Also clear the separate lattice points storage
            lattice_points_storage.set([])
            # Also clear the separate tilt info storage
            tilt_info_storage.set(None)
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the dual tilt info storages
            tilt_info_green_storage.set(None)
            tilt_info_red_storage.set(None)
            # Also clear the separate ellipse params storage
            ellipse_params_storage.set(None)
            # Also clear the tuned markers storage
            tuned_markers_storage.set([])
            tuned_resolution_radius.set(None)
            # Also clear the tuned resolution ring storage
            tuned_resolution_radius.set(None)
            # Update the separate mode storage
            current_mode_storage.set('Lattice Point')
        
        # Clear drawn circles when switching modes
        new_state['drawn_circles'] = []
        
        # Also clear current measurement when switching modes
        new_state['current_measurement'] = None
        
        fft_state.set(new_state)
        
        # Update Fit button state
        is_disabled = input.label_mode() != "Lattice Point"
        # Tune Markers now works in both modes, so don't disable it
        ui.update_action_button("fit_markers", disabled=is_disabled, session=session)
        ui.update_action_button("estimate_tilt", disabled=is_disabled, session=session)


    
    # Reactive effects to update 1D plot FigureWidget in-place
    @reactive.Effect
    @reactive.event(input.log_y, input.use_mean_profile, input.smooth, input.detrend, input.window_size)
    def _():
        """Update 1D plot in-place when plot parameters change."""
        widget = fft_1d_widget.get()
        if widget is not None and len(widget.data) > 0:
            # Get updated plot data
            plot_data = fft_1d_data()
            if plot_data is None:
                return
            
            # Update the trace data in-place
            with widget.batch_update():
                widget.data[0].x = plot_data['x_data']
                widget.data[0].y = plot_data['y_data']
                widget.data[0].name = plot_data['profile_label']
                
                # Update y-axis title based on log_y setting
                if input.log_y():
                    widget.layout.yaxis.title.text = "Log(FFT intensity)"
                else:
                    widget.layout.yaxis.title.text = "FFT intensity"

    @reactive.Effect
    @reactive.event(input.find_max)
    def _():
        """Handle Find Max button click to find and mark the global maximum in the current view."""
        # Get the 1D plot widget
        widget = fft_1d_widget.get()
        if widget is None or len(widget.data) == 0:
            return
        
        # Get the current plot data (already processed with smoothing/detrending)
        plot_data = fft_1d_data()
        if plot_data is None:
            return
        
        x_data = plot_data['x_data']
        y_data = plot_data['y_data']
        
        # Get the current zoom range from the widget
        x_range = widget.layout.xaxis.range
        
        # If zoomed, filter data to the visible range
        if x_range is not None and len(x_range) == 2:
            x_min, x_max = x_range
            # Find indices within the visible range
            mask = (x_data >= x_min) & (x_data <= x_max)
            if np.any(mask):
                visible_x = x_data[mask]
                visible_y = y_data[mask]
            else:
                # No data in range, use all data
                visible_x = x_data
                visible_y = y_data
        else:
            # Not zoomed, use all data
            visible_x = x_data
            visible_y = y_data
        
        # Find the global maximum in the visible range
        if len(visible_y) > 0:
            # Find integer maximum first
            max_idx = np.argmax(visible_y)
            max_x_int = visible_x[max_idx]
            max_y_int = visible_y[max_idx]
            
            print(f"Integer maximum found at x={max_x_int:.3f}, y={max_y_int:.3f}")
            
            # Check if Super Resolution mode is enabled
            if input.super_resolution():
                # Super Resolution mode: Fit Gaussian around the maximum for sub-pixel precision
                try:
                    # Get window size from slider (convert to half-window for ± range)
                    window_half_size = input.gaussian_window() / 2.0
                    
                    # Find indices within window around the maximum
                    mask = np.abs(visible_x - max_x_int) <= window_half_size
                    
                    if np.sum(mask) >= 5:  # Need at least 5 points for Gaussian fitting
                        fit_x = visible_x[mask]
                        fit_y = visible_y[mask]
                        
                        # Define Gaussian function: y = a * exp(-((x - mu)^2) / (2 * sigma^2)) + c
                        def gaussian(x, a, mu, sigma, c):
                            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                        
                        # Initial parameter guesses
                        a_guess = max_y_int  # amplitude
                        mu_guess = max_x_int  # center
                        sigma_guess = 1.0     # width
                        c_guess = np.min(fit_y)  # offset
                        
                        # Fit the Gaussian
                        from scipy.optimize import curve_fit
                        popt, _ = curve_fit(gaussian, fit_x, fit_y, 
                                          p0=[a_guess, mu_guess, sigma_guess, c_guess],
                                          maxfev=1000)
                        
                        # Extract fitted parameters
                        a_fit, mu_fit, sigma_fit, c_fit = popt
                        
                        # Use the fitted center as the refined maximum position
                        max_x = mu_fit
                        max_y = gaussian(mu_fit, *popt)
                        
                        print(f"Super Resolution: Gaussian fit successful with {input.gaussian_window()}-pixel window")
                        print(f"  - Center: {mu_fit:.5f}, Amplitude: {a_fit:.3f}, Sigma: {sigma_fit:.3f}")
                        print(f"  - Sub-pixel maximum: x={max_x:.5f}, y={max_y:.3f} (refined from {max_x_int:.3f})")
                        
                        # Store Gaussian parameters for overlay
                        gaussian_params = popt
                        gaussian_fit_range = (fit_x.min(), fit_x.max())
                        
                    else:
                        # Not enough points for fitting, use integer maximum
                        max_x = max_x_int
                        max_y = max_y_int
                        gaussian_params = None
                        gaussian_fit_range = None
                        print(f"Super Resolution: Not enough points for Gaussian fitting ({np.sum(mask)} points), using integer maximum")
                        
                except Exception as e:
                    # Gaussian fitting failed, use integer maximum
                    max_x = max_x_int
                    max_y = max_y_int
                    gaussian_params = None
                    gaussian_fit_range = None
                    print(f"Super Resolution: Gaussian fitting failed ({e}), using integer maximum")
            else:
                # Standard mode: Use integer maximum
                max_x = max_x_int
                max_y = max_y_int
                gaussian_params = None
                gaussian_fit_range = None
                print(f"Standard resolution mode: Using integer maximum")
            
            # Calculate apix value corresponding to the maximum position
            calculated_apix = None
            calc_state = fft_calculation_state.get()
            if calc_state['region'] is not None:
                resolution, _ = get_resolution_info(calc_state['resolution_type'], calc_state['custom_resolution'])
                if resolution is not None and max_x > 0:
                    # Convert from 1D plot coordinates to FFT image coordinates
                    region_size = calc_state['region'].size[0]
                    
                    # Calculate apix using the resolution and distance
                    # Formula: apix = (distance_in_pixels * resolution) / image_size
                    calculated_apix = (max_x * resolution) / region_size
                    
                    if 0.01 <= calculated_apix <= 6.0:
                        # Update the apix slider and text input with the calculated value
                        ui.update_slider("apix_slider", value=calculated_apix, session=session)
                        ui.update_text("apix_exact_str", value=f"{calculated_apix:.3f}", session=session)
                        if input.super_resolution():
                            print(f"Updated apix to {calculated_apix:.3f} Å/px based on Gaussian-fitted maximum at x={max_x:.5f}")
                        else:
                            print(f"Updated apix to {calculated_apix:.3f} Å/px based on integer maximum at x={max_x:.3f}")
                    else:
                        print(f"Calculated apix {calculated_apix:.3f} is outside valid range [0.01, 6.0]")
                        calculated_apix = None  # Mark as invalid
                else:
                    print("Could not calculate apix - no resolution or invalid max position")
            else:
                print("Could not calculate apix - no FFT calculation state available")
            
            # ALWAYS add vertical line at max position regardless of apix calculation success
            print(f"Adding vertical line at max position x={max_x:.5f}, y={max_y:.3f}")
            
            try:
                with widget.batch_update():
                    # Remove any existing max markers and Gaussian fits
                    traces_to_keep = []
                    removed_count = 0
                    for trace in widget.data:
                        if not (hasattr(trace, 'name') and (trace.name == 'max_marker' or trace.name == 'gaussian_fit')):
                            traces_to_keep.append(trace)
                        else:
                            removed_count += 1
                    widget.data = traces_to_keep
                    print(f"Removed {removed_count} existing max markers and Gaussian fits")
                    
                    # Get current visible range from widget (what user is actually seeing)
                    current_x_range = widget.layout.xaxis.range
                    current_y_range = widget.layout.yaxis.range
                    
                    print(f"Current x-axis range: {current_x_range}")
                    print(f"Current y-axis range: {current_y_range}")
                    print(f"Max position to mark: x={max_x:.3f}")
                    
                    # Determine y-axis range for the vertical line
                    if current_y_range is not None and len(current_y_range) == 2:
                        y_min, y_max_range = current_y_range
                        print(f"Using current y-axis range: [{y_min:.1f}, {y_max_range:.1f}]")
                    else:
                        # Fallback: use data range with padding
                        y_min = min(0, np.min(y_data) * 0.9)
                        y_max_range = np.max(y_data) * 1.1
                        print(f"Using calculated y-axis range: [{y_min:.1f}, {y_max_range:.1f}]")
                        # Update layout range
                        widget.layout.yaxis.range = [y_min, y_max_range]
                    
                    # Check if max_x is within visible range
                    if current_x_range is not None and len(current_x_range) == 2:
                        x_min_range, x_max_range = current_x_range
                        if not (x_min_range <= max_x <= x_max_range):
                            print(f"WARNING: max_x={max_x:.3f} is outside visible x-range [{x_min_range:.3f}, {x_max_range:.3f}]")
                    
                    # Create hover info with apix if available
                    if input.super_resolution() and gaussian_params is not None:
                        # Super resolution mode with successful Gaussian fit
                        if calculated_apix is not None:
                            hover_info = f'<b>Global Max (Super Resolution)</b><br>x: {max_x:.5f}<br>y: {max_y:.3f}<br>Apix: {calculated_apix:.3f} Å/px<extra></extra>'
                        else:
                            hover_info = f'<b>Global Max (Super Resolution)</b><br>x: {max_x:.5f}<br>y: {max_y:.3f}<extra></extra>'
                    else:
                        # Standard resolution mode
                        if calculated_apix is not None:
                            hover_info = f'<b>Global Max (Standard)</b><br>x: {max_x:.3f}<br>y: {max_y:.3f}<br>Apix: {calculated_apix:.3f} Å/px<extra></extra>'
                        else:
                            hover_info = f'<b>Global Max (Standard)</b><br>x: {max_x:.3f}<br>y: {max_y:.3f}<extra></extra>'
                    
                    # Add new vertical line at max position with enhanced visibility
                    line_trace = go.Scatter(
                        x=[max_x, max_x],
                        y=[y_min, y_max_range],
                        mode='lines',
                        line=dict(color='red', width=1, dash='solid'),  # Slim vertical line
                        name='max_marker',
                        showlegend=False,
                        hovertemplate=hover_info,
                        opacity=1.0  # Ensure full opacity
                    )
                    
                    widget.add_trace(line_trace)
                    
                    # Add Gaussian curve overlay if fitting was successful
                    if gaussian_params is not None and gaussian_fit_range is not None:
                        # Create high-resolution x points for smooth curve
                        x_smooth = np.linspace(gaussian_fit_range[0], gaussian_fit_range[1], 100)
                        
                        # Calculate Gaussian curve using fitted parameters
                        def gaussian(x, a, mu, sigma, c):
                            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                        
                        y_smooth = gaussian(x_smooth, *gaussian_params)
                        
                        # Create Gaussian curve trace
                        gaussian_trace = go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            line=dict(color='red', width=2, dash='dot'),
                            name='gaussian_fit',
                            showlegend=False,
                            hovertemplate='<b>Gaussian Fit</b><br>x: %{x:.5f}<br>y: %{y:.3f}<extra></extra>',
                            opacity=0.8
                        )
                        
                        widget.add_trace(gaussian_trace)
                        print(f"Gaussian curve overlay added: range [{gaussian_fit_range[0]:.3f}, {gaussian_fit_range[1]:.3f}]")
                    
                    print(f"Vertical line added successfully:")
                    print(f"  - Position: x={max_x:.3f}")
                    print(f"  - Y-range: [{y_min:.1f}, {y_max_range:.1f}]")
                    print(f"  - Line style: red, width=1, solid")
                    print(f"  - Total traces in widget: {len(widget.data)}")
                    
                    # Force a refresh of the widget display
                    widget.layout.uirevision = f"max_marker_{max_x:.3f}_{max_y:.3f}"
                    
            except Exception as e:
                print(f"Error adding vertical line: {e}")
                import traceback
                traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.add_to_table)
    def _():
        """Handle Add to Table button click to add current analysis to the table."""
        try:
            # Check if we have FFT calculation state (must have calculated FFT first)
            calc_state = fft_calculation_state.get()
            if calc_state['region'] is None:
                print("No FFT calculation available for table entry. Please click 'Calc FFT' first.")
                return
                
            # Get current analysis data from FFT calculation state
            filename = image_filename.get() or "Unknown"
            region_size = f"{calc_state['region'].size[0]}×{calc_state['region'].size[1]} px"
            
            # Get region location from zoom state (when FFT was calculated)
            zoom_state = image_zoom_state.get()
            if zoom_state.get('drawn_region') is not None:
                drawn_region = zoom_state['drawn_region']
                region_location = f"x:{int(drawn_region['x0'])}–{int(drawn_region['x1'])}, y:{int(drawn_region['y0'])}–{int(drawn_region['y1'])}"
            else:
                region_location = "Full image"
            
            # Use current apix value (allows user to adjust apix after FFT calculation)
            apix_value = get_apix()
            
            # Get nominal value from textbox
            nominal_value = float(input.nominal_apix())
            
            # Create new row
            new_row = pd.DataFrame({
                'Filename': [filename],
                'Region Size': [region_size],
                'Region Location': [region_location],
                'Apix': [f"{apix_value:.3f}"],
                'Nominal': [nominal_value]
            })
            
            # Add to existing table data
            current_data = region_table_data.get()
            updated_data = pd.concat([current_data, new_row], ignore_index=True)
            region_table_data.set(updated_data)
            
            print(f"Added row to region analysis table: {filename}, {region_size}, {region_location}, {apix_value:.3f}, {nominal_value}")
            
        except Exception as e:
            print(f"Error adding to table: {e}")
            import traceback
            traceback.print_exc()

    
    # Note: Selection handling is now done directly via FigureWidget callback
    
    # Note: Click events are handled by FigureWidget callback if needed

    
    # Note: Line drawing events are now handled directly in the FigureWidget's on_relayout callback
    # This provides better event capture for the drawline tool



    # Note: The FFT figure should only re-render when the base FFT image changes (cached_fft_image)
    # Overlay changes (lattice points, ellipse, mode) should not trigger base FFT re-renders
    
    # The overlays will be updated through click callbacks and shape interactions
    # without re-rendering the base FFT image

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
