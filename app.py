from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_plotly, render_widget
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import math
import requests
import os
from urllib.parse import urlparse
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
    # compute_fft_polar_heatmap_data,
    calibrateMag_process_one_region_advanced,
    calibrateMag_process_one_region_fast,
    resolution_to_radius,
    # create_fft_1d_plotly_figure,
    get_image,
    plot_image,
    get_image_with_binning,
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
- 1D radial plot
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
    

# Create the main UI using page_sidebar for proper Shiny styling
app_ui = ui.page_fillable(
    # ui.sidebar(
    #     # App title and description in sidebar
    #     ui.h1("Magnification Calibration", style="font-size: 24px; font-weight: bold; margin-bottom: 15px; color: #333;"),
    #     ui.p("This tool helps calibrate electron microscopes by analyzing test specimen images.", 
    #          style="font-size: 13px; color: #666; margin-bottom: 10px; line-height: 1.4;"),
    #     # Add some basic help text
    #     ui.div(
    #         ui.h4("Quick Start:", style="font-size: 16px; font-weight: bold; margin-bottom: 10px;"),
    #         ui.tags.ol(
    #             ui.tags.li("Select input method (URL or Upload)"),
    #             ui.tags.li("Choose region and calculate FFT"),
    #             ui.tags.li("Analyze resolution patterns"),
    #             style="font-size: 12px; line-height: 1.4; margin-left: 15px;"
    #         ),
    #         style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;"
    #     ),
    #     open="closed"
    # ),
    # Add custom CSS for enhanced styling
    ui.tags.head(
        ui.tags.style("""
            /* Image container styles */
            .image-output {
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: auto;
                #padding: 5px;
                #margin-bottom: 5px;
                width: 100%;
                min-height: 300px;
                flex: 1;
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
                border-radius: 2px;
                border: 2px solid transparent;
            }
            .image-output img {
                height: auto;
                width: auto;
                max-width: none;
                max-height: none;
                margin-bottom: 8px;
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
        """)
    ),
    # App title
    ui.h1("Magnification Calibration", 
          style="text-align: center; font-size: 28px; font-weight: bold; margin: 10px 0; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px;"),
    # Primary analysis section - always visible
    ui.div(
        {"style": "margin-bottom: 10px;"},
        ui.layout_columns(
            ui.card(
                ui.card_header("Original Image"),
                # Accordion panel with controls
                ui.accordion(
                    ui.accordion_panel(
                        "Input Image",
                        # Input method selection with conditional panels
                        ui.div(
                            {"style": "display: flex; gap: 10px; margin-bottom: 5px; width: 100%;"},
                            # Left side: Radio buttons and inputs
                            ui.div(
                                {"style": "flex: 4; display: flex; flex-direction: column; gap: 5px;"},
                                ui.input_radio_buttons(
                                    "input_method",
                                    "Method:",
                                    choices=["URL", "Upload (.mrc,.tiff,.png)"],
                                    selected="URL",
                                    inline=True
                                ),
                                # Conditional input for URL
                                ui.panel_conditional(
                                    "input.input_method === 'URL'",
                                    ui.input_text(
                                        "download_url",
                                        "Download URL:",
                                        value="https://raw.githubusercontent.com/jianglab/magCalApp/008ab91715945e3a52355ed2be64bb8bc027cc13/test_image/130k-Pixel0.75A.tiff",
                                        placeholder="Enter image URL",
                                        width="100%"
                                    )
                                ),
                                # Conditional input for Upload
                                ui.panel_conditional(
                                    "input.input_method === 'Upload (.mrc,.tiff,.png)'",
                                    ui.input_file("upload", None, accept=["image/*", ".mrc", ".tif", ".png"], multiple=True)
                                )
                            ),
                            # Right side: Upload files table (only visible when Upload is selected)
                            ui.panel_conditional(
                                "input.input_method === 'Upload (.mrc,.tiff,.png)'",
                                ui.div(
                                    {"style": "flex: 6; overflow-y: auto; padding: 2px; min-height: 0; max-height: 170px;"},
                                    ui.output_data_frame("upload_files_table"),
                                )
                            )
                        ),
                        ui.div(
                            {"style": "display: flex; justify-content: flex-start; align-items: flex-start; gap: 100px; width: 100%;"},
                            ui.div(
                                {"style": "display: flex; flex-direction: column; gap: 5px;"},
                                ui.tags.label("Nominal Pixel Size", {"for": "nominal_apix", "style": "margin-bottom: 0","width":"120px"}),
                                ui.input_numeric("nominal_apix", None, value=1.00, min=0.01, max=10.0, step=0.01, width="160px"),
                            ),
                            ui.div(
                                {"style": "display: flex; flex-direction: column; gap: 5px;"},
                                ui.tags.label("Test Specimen:", {"for": "resolution_type", "style": "margin-bottom: 0"}),
                                ui.div(
                                    {"style": "display: flex; align-items: flex-start; gap: 10px;"},
                                    ui.input_select("resolution_type", None,
                                        choices=["Graphene (2.13 Å)", "Gold (2.355 Å)", "Ice (3.661 Å)", "Custom"],
                                        selected="Graphene (2.13 Å)",
                                        width="180px"),
                                    ui.panel_conditional(
                                        "input.resolution_type == 'Custom'",
                                        ui.input_numeric("custom_resolution", "Custom Res (Å):", value=3.0, min=0.1, max=10.0, step=0.01, width="240px")
                                    )
                                )
                            )
                        ),
                        open="closed"
                    ),
                    open=False,
                    multiple=False
                ),
                output_widget("image_display"),
                ui.div(
                    {"style": "display: flex; gap: 5px; padding: 5px; justify-content: center;"},
                    #ui.input_action_button("clear_drawn_region", "Clear Selection", class_="btn-secondary"),
                    ui.input_action_button("calc_fft", "Calc FFT", class_="btn-primary"),
                ),
                # ui.div(
                #     {"class": "card-footer"},
                #     "Use box selection tool to drag and select regions (you'll see red dots), then click 'Calc FFT' to analyze.",
                # ),
                full_screen=True,
            ),
            # Right column: FFT and Result cards stacked vertically
            ui.div(
                {"style": "display: flex; flex-direction: column; gap: 8px; height: 100%;"},
                ui.card(
                    ui.card_header("FFT Analysis"),
                    ui.navset_tab(
                        ui.nav_panel(
                            ui.tooltip(
                                "1D Radial Profile",
                                "Radially averaged power spectrum analysis with NuFFT interpolation for enhanced resolution detection",
                                placement="top"
                            ),
                            ui.div(
                                {"style": "height: 100%; display: grid; grid-template-columns: 1fr 200px; gap: 8px;"},
                                # Left: Main content area with NuFFT power curve and heatmap  
                                ui.div(
                                    {"style": "height: 100%; display: flex; flex-direction: column;"},
                                    # Top: Power curve plot (60% height) - shows first for quick interaction
                                    ui.div(
                                        {"style": "flex: 6; display: flex; flex-direction: column; min-height: 0;"},
                                        ui.div(
                                            {"style": "padding: 5px 10px; font-size: 14px; font-weight: 500; border-bottom: 1px solid #dee2e6; background-color: #f8f9fa;"},
                                            ui.tooltip(
                                                "1D Power Curve",
                                                "Click on peaks to calculate tentative pixel size. For multiple peaks, select the leftmost one.",
                                                placement="top"
                                            )
                                        ),
                                        ui.div(
                                            {"style": "flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0;"},
                                            output_widget("nufft_power_curve")
                                        )
                                    ),
                                    # Bottom: NuFFT focused heatmap (40% height) - shows after click
                                    ui.div(
                                        {"style": "flex: 4; display: flex; align-items: center; justify-content: center; min-height: 0; border-top: 1px solid #dee2e6;"},
                                        output_widget("nufft_heatmap")
                                    )
                                ),
                                # Right: Controls spanning entire height
                                ui.div(
                                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; width: 100%; padding: 5px;"},
                                    ui.input_checkbox("nufft_log_y", "Log Scale", value=False),
                                    # ui.input_checkbox("nufft_use_mean_profile", "Use Average Profile", value=False),
                                    # ui.input_checkbox("nufft_smooth", "Smooth Signal", value=False),
                                    # ui.input_checkbox("nufft_detrend", "Detrend Signal", value=False),
                                    # ui.div(
                                    #     {"style": "margin-bottom: 5px;"},
                                    #     ui.panel_conditional(
                                    #         "input.nufft_smooth",
                                    #         ui.input_slider("nufft_window_size", "Window Size", min=1, max=11, value=3, step=2),
                                    #     ),
                                    # ),
                                    ui.div(
                                        {"style": "margin-bottom: 5px;"},
                                        ui.input_slider("nufft_r_sampling_freq", "Radial Sampling Frequency (per pixel)", min=0.1, max=10, value=5, step=0.1),
                                    ),
                                    ui.div(
                                        {"style": "margin-bottom: 5px;"},
                                        ui.input_slider("nufft_theta_sampling_freq", "Angular Sampling Frequency (per degree)", min=0.1, max=5, value=2, step=0.1),
                                    ),
                                    ui.div(
                                        {"style": "margin-bottom: 5px;"},
                                        ui.input_slider("nufft_display_range", "Display Range (%)", min=1, max=10, value=3, step=0.5),
                                    ),
                                    # ui.input_action_button("nufft_find_apix", "Find Apix", class_="btn-primary"),
                                )
                            )
                        ),
                        ui.nav_panel(
                            ui.tooltip(
                                "2D Spectrum",
                                "Click on the spots associated with the known resolution (e.g., 2.13 for GO) to place markers. Use 'Autocorrect' to refine positions",
                                placement="top"
                            ),
                            ui.div(
                                {"style": "height: 100%; display: grid; grid-template-columns: 1fr 250px; gap: 8px;"},
                                # Left side: FFT display
                                output_widget("fft_with_circle"),
                                # ui.div(
                                #     {"style": "width: 100%; height: 100%;"},
                                #     output_widget("fft_with_circle")
                                # ),
                                # Right side: Controls arranged with fixed positioning
                                ui.div(
                                    {"style": "position: relative; background-color: #f8f9fa; border-radius: 8px; height: 450px; width: 250px; flex-shrink: 1; overflow: hidden;"},
                                    # Label Type dropdown - fixed position
                                    ui.div(
                                        {"style": "position: absolute; top: 10px; left: 10px; right: 10px;"},
                                        ui.input_select("label_mode", "", 
                                            choices=["Resolution Ring", "Lattice Point"], 
                                            selected="Resolution Ring")
                                    ),
                                    # FFT Range slider - fixed position
                                    ui.div(
                                        {"style": "position: absolute; top: 57px; left: 10px; right: 10px;"},
                                        ui.input_slider("contrast", "Contrast", min=0.1, max=5.0, value=1.0, step=0.1),
                                    ),

                                    # Buttons with fixed positions - well spaced
                                    ui.input_action_button("clear_markers", "Clear Markers", class_="btn-secondary", style="position: absolute; top: 140px; left: 10px; right: 10px; padding: 8px;"),
                                    ui.input_action_button("tune_markers", "Autocorrect", class_="btn-secondary", style="position: absolute; top: 190px; left: 10px; right: 10px; padding: 8px;"),
                                    ui.input_action_button("fit_markers", "Fit Ellipse", class_="btn-secondary", style="position: absolute; top: 240px; left: 10px; right: 10px; padding: 8px;"),
                                    ui.input_action_button("estimate_tilt", "Estimate Tilt", class_="btn-secondary", style="position: absolute; top: 290px; left: 10px; right: 10px; padding: 8px;"),
                                    # Tilt output - fixed position
                                    ui.div(
                                        {"style": "position: absolute; top: 340px; left: 10px; right: 10px; font-size: 12px; color: #666; min-height: 20px;"},
                                        ui.output_text("tilt_output"),
                                    ),
                                )
                            )
                        ),

                        # ui.nav_panel(
                        #     "DFT",
                        #     ui.div(
                        #         {"style": "height: 100%; display: grid; grid-template-columns: 1fr 200px; gap: 15px;"},
                        #         # Left: Main content area with polar heatmap and radial profile
                        #         ui.div(
                        #             {"style": "height: 100%; display: flex; flex-direction: column;"},
                        #             # Top: Polar Heat Map (70% height)
                        #             ui.div(
                        #                 {"style": "flex: 7; display: flex; align-items: center; justify-content: center; min-height: 0;"},
                        #                 output_widget("fft_polar_heatmap")
                        #             ),
                        #             # Bottom: Radial Profile (30% height) 
                        #             ui.div(
                        #                 {"style": "flex: 3; display: flex; align-items: center; justify-content: center; min-height: 0; border-top: 1px solid #dee2e6;"},
                        #                 output_widget("fft_1d_plot")
                        #             )
                        #         ),
                        #         # Right: Controls spanning entire height
                        #         ui.div(
                        #             {"style": "display: flex; flex-direction: column; justify-content: flex-start; width: 100%; padding: 10px;"},
                        #             ui.input_checkbox("log_y", "Log Scale", value=False),
                        #             ui.input_checkbox("use_mean_profile", "Use Average Profile", value=False),
                        #             ui.input_checkbox("smooth", "Smooth Signal", value=False),
                        #             ui.input_checkbox("detrend", "Detrend Signal", value=False),
                        #             ui.div(
                        #                 {"style": "margin-bottom: 10px;"},
                        #                 ui.panel_conditional(
                        #                     "input.smooth",
                        #                     ui.input_slider("window_size", "Window Size", min=1, max=11, value=3, step=2),
                        #                 ),
                        #             ),
                        #             ui.div(
                        #                 {"style": "margin-bottom: 10px;"},
                        #                 ui.input_checkbox("super_resolution", "Super Resolution", value=True),
                        #                 ui.panel_conditional(
                        #                     "input.super_resolution",
                        #                     ui.input_slider("gaussian_window", "Gaussian Window (pixels)", min=1, max=21, value=5, step=2),
                        #                 ),
                        #             ),
                        #             ui.input_action_button("find_max", "Find Max", class_="btn-primary"),
                        #         )
                        #     )
                        # )
                    ),
                    full_screen=True,
                    style="flex: 1; min-height: 400px;"
                ),
                # Result card (bottom)
                ui.card(
                    ui.card_header("Result"),
                    ui.div(
                        {"style": "padding: 8px; display: flex; align-items: center; gap: 12px; min-height: 80px;"},
                        # Apix slider
                        #ui.div(
                        #{"style": "flex: 5; display: flex; align-items: flex-end; gap: 5px;"},
                        ui.tags.label("Pixel Size (Å/px):", {"for": "apix_slider", "style": "margin: 0; white-space: nowrap;"}),
                        ui.input_slider("apix_slider", None, min=0.01, max=2.0, value=1.0, step=0.0001, width="100%"),
                            # ui.div(
                            #     {"style": "flex: 1; display: flex; align-items: flex-end;"},
                            #     ui.input_slider("apix_slider", None, min=0.01, max=2.0, value=1.0, step=0.0001, width="100%")
                            # ),
                        #),
                        # Apix exact input and Set button
                        #ui.div(
                            #{"style": "display: flex; align-items: flex-end; gap: 5px; flex: 3;"},
                            #ui.tags.label("Exact Value:", {"for": "apix_exact_str", "style": "margin: 0; white-space: nowrap;"}),
                        ui.input_text("apix_exact_str", None, value="1.0", width="200px"),

                        ui.input_action_button("apix_set_btn", "Set", class_="btn-primary", style="height: 38px; min-width: 50px; display: flex; align-items: center; justify-content: center;"),
                        #),
                        # Add to Table button
                        ui.input_action_button("add_to_table", "Add to Table", class_="btn-success", style="height: 38px; width: 100%;max-width: 200px; display: flex; align-items: center; justify-content: center;"),

                    ),
                    style="flex: 1 2 2 1 2;min-height: 100px;"
                )
            ),
            col_widths=[5, 7],
        )
    ),
    # Secondary analysis section - scrollable below
    ui.div(
        {"style": "margin-top: 10px;"},
        ui.card(
            ui.card_header(
                ui.tooltip(
                    "Statistics",
                    "Table tracks all calibrated pixel size results across multiple regions and magnifications. Click Download CSV to save the result table",
                    placement="top"
                )
            ),
            # Use row layout: table+buttons on left (55%), plot on right (45%), both full height
            ui.layout_columns(
                # Left column: Table and controls (55% width, 100% height)
                ui.div(
                    {"style": "display: flex; flex-direction: column; height: 500px;"},
                    ui.div(
                        {"style": "flex: 1; overflow-y: auto; padding: 5px; min-height: 0;"},
                        ui.output_data_frame("region_table"),
                    ),
                    ui.div(
                        {"style": "flex-shrink: 0; display: flex; gap: 5px; padding: 5px; justify-content: center; align-items: center; flex-wrap: wrap; border-top: 1px solid #dee2e6;"},
                        # ui.div(
                        #     {"style": "display: flex; gap: 5px; align-items: center;"},
                        #     #ui.input_action_button("random_generate", "Random Generate", class_="btn-info"),
                        #     ui.div(
                        #         {"style": "display: flex; flex-direction: column; align-items: center;"},
                        #         ui.div(
                        #             {"style": "font-size: 10px; color: #666; margin-bottom: 2px;"},
                        #             "Count"
                        #         ),
                        #         ui.input_numeric("random_count", None, value=5, min=1, max=100, step=1, width="70px"),
                        #     ),
                        #     ui.div(
                        #         {"style": "display: flex; flex-direction: column; align-items: center;"},
                        #         ui.div(
                        #             {"style": "font-size: 10px; color: #666; margin-bottom: 2px;"},
                        #             "Size %"
                        #         ),
                        #         ui.input_numeric("region_size_percent", None, value=0.2, min=0.1, max=1.0, step=0.1, width="70px"),
                        #     ),
                        # ),
                        ui.input_action_button("delete_selected", "Delete Selected", class_="btn-danger"),
                        ui.input_action_button("clear_table", "Clear Table", class_="btn-secondary"),
                        ui.download_button("download_csv", "Download CSV", class_="btn-primary"),
                    ),
                ),
                # Right column: Plot (45% width, 100% height)
                ui.div(
                    {"style": "height: 500px; padding: 5px; display: flex; flex-direction: column;"},
                    ui.div(
                        {"style": "flex: 1; min-height: 0;"},
                        output_widget("apix_centered_by_nominal_plot"),
                    ),
                ),
                col_widths=[7, 5],  # 58.3%/41.7% split (closest to 55%/45% with integer grid)
            ),
            full_screen=True,
        ),
        fillable=True,
        )
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
    
    # Flag to prevent duplicate image downloads during initialization
    startup_download_completed = reactive.Value(False)
    
    # Handle file upload and populate upload table
    @reactive.Effect
    @reactive.event(input.upload)
    def _():
        """Handle multiple file upload and populate the upload table."""
        files = input.upload()
        if files is None or len(files) == 0:
            # Clear the uploaded files data if no files
            uploaded_files_data.set([])
            return
            
        # Process each uploaded file
        files_info = []
        for i, file_info in enumerate(files):
            # Extract nominal apix from filename, default to 1.0 if unclear
            filename = file_info['name']
            nominal_apix = extract_nominal(filename)
            
            files_info.append({
                'index': i,
                'name': filename,
                'nominal_apix': nominal_apix,
                'file_info': file_info  # Store the full file info for later use
            })
        
        # Update the uploaded files data
        uploaded_files_data.set(files_info)
        
        # Auto-select the first file
        if files_info:
            selected_file_index.set(0)
            # Set the nominal apix from the first file
            ui.update_numeric("nominal_apix", value=files_info[0]['nominal_apix'])
    
    # Handle table row selection for file switching  
    @reactive.Effect
    def _():
        """Handle file selection from upload table."""
        try:
            # Get the data frame selection from upload_files_table
            selected_rows = input.upload_files_table_selected_rows()
            if selected_rows and len(selected_rows) > 0:
                # Update selected file index based on table selection
                new_index = selected_rows[0]
                if new_index != selected_file_index.get():
                    selected_file_index.set(new_index)
                    
                    # Update nominal apix from selected file
                    files_data = uploaded_files_data.get()
                    if files_data and new_index < len(files_data):
                        selected_file = files_data[new_index]
                        ui.update_numeric("nominal_apix", value=selected_file['nominal_apix'])
        except Exception as e:
            print(f"Error handling table selection: {e}")
    
    # Remove fft_1d_data since we're no longer using static markers

    # Add reactive value to cache the base FFT image
    cached_fft_image = reactive.Value(None)
    
    # Add reactive values to cache NuFFT data
    cached_nufft_heatmap_data = reactive.Value(None)
    cached_nufft_power_data = reactive.Value(None)

    # Add plot zoom state
    plot_zoom = reactive.Value({
        'x_range': None,
        'y_range': None
    })
    
    # Shared x-axis range for heatmap and 1D profile
    shared_x_range = reactive.Value(None)
    
    # Track which plot triggered the range change to avoid loops
    range_update_source = reactive.Value(None)

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
    
    # Add reactive value to store the NuFFT power curve FigureWidget for click handling
    nufft_power_widget = reactive.Value(None)

    # Add reactive value to store the NuFFT heatmap FigureWidget for click handling
    nufft_heatmap_widget = reactive.Value(None)
    
    # Add reactive value to store the clicked position on NuFFT power curve for green line
    #nufft_click_position = reactive.Value(None)
    
    # Smart heatmap control: track clicked frequency for focused heatmap rendering
    nufft_clicked_frequency = reactive.Value(None)
    nufft_show_focused_heatmap = reactive.Value(False)
    
    # Flag to prevent NuFFT recalculation when apix is updated from power curve clicks
    apix_updating_from_nufft_click = reactive.Value(False)
    
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
    
    
    # Add reactive value to force complete FFT widget refresh
    fft_widget_refresh_trigger = reactive.Value(0)
    
    # Add reactive values for uploaded files management
    uploaded_files_data = reactive.Value([])  # List of file info dicts
    selected_file_index = reactive.Value(0)   # Currently selected file index
    
    # Helper function to extract nominal value from filename
    def extract_nominal(filename):
        """Extract nominal apix value from filename.
        
        Examples:
        - '390k-nominal0.36.tiff' -> 0.36
        - '150k-nominal0.97.tiff' -> 0.97  
        - '130k-Pixel0.75A.tiff' -> 0.75
        - 'test1.25image.mrc' -> 1.25
        
        Returns 1.0 if extraction fails or if ambiguous (multiple values found).
        """
        import re
        if not isinstance(filename, str):
            return 1.0
            
        # Try different patterns in order of specificity
        patterns = [
            r"nominal(\d+\.\d+)",        # nominal0.36, nominal1.25
            r"pixel(\d+\.\d+)a?",        # Pixel0.75A, pixel1.0
            r"apix(\d+\.\d+)",           # apix0.5, apix1.2
            r"(\d+\.\d+)a(?:ngstrom)?",  # 0.75A, 1.5angstrom
            r"(\d+\.\d+)(?=\D|$)"        # Any X.XX format (less specific, use last)
        ]
        
        found_values = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match)
                    # Only consider reasonable apix values (0.1 to 5.0)
                    if 0.1 <= value <= 5.0:
                        found_values.append(value)
                except ValueError:
                    continue
        
        # Return the first reasonable value found, or 1.0 if none/ambiguous
        if len(found_values) == 1:
            return found_values[0]
        elif len(found_values) > 1:
            # If multiple values, prefer the first one from a more specific pattern
            return found_values[0]
        else:
            return 1.0
    
    # Add reactive value to store the region analysis table data
    region_table_data = reactive.Value(pd.DataFrame({
        'Filename': [],
        'Region Size': [],
        'Region Location': [],
        'Pixel Size': [],
        'Nominal': [],
        'Average Pixel Size': []
    }))
    
    # Add reactive value to store the region and parameters used for current FFT calculation
    fft_calculation_state = reactive.Value({
        'region': None,
        'apix': None,
        'resolution_type': None,
        'custom_resolution': None
    })
    
    # Add separate reactive state for NuFFT calculations
    nufft_calculation_state = reactive.Value({
        'region': None,
        'apix': None,
        'resolution_type': None,
        'custom_resolution': None
    })

    # Track FFT widget creation to prevent duplicates
    fft_widget_last_created = reactive.Value(None)
    
    # Update 1D plot when cached FFT image changes (removed base_fft_trigger to prevent double render)
    @reactive.Effect
    @reactive.event(cached_fft_image)
    def _():
        """Update 1D plot when the base FFT image changes."""
        
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
        # Tune Markers now works in both modes
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
        click_flag = apix_updating_from_nufft_click.get()
        if click_flag:
            # Skip updates when apix is being set from NuFFT click
            return
        apix_master.set(input.apix_slider())
        # Clear 1D plot clicked position when apix changes from slider
        #plot_1d_click_pos.set({'x': None, 'y': None})

    @reactive.Effect
    @reactive.event(input.apix_set_btn)
    def _():
        try:
            val = float(input.apix_exact_str())
            if 0.001 <= val <= 6.0:
                click_flag = apix_updating_from_nufft_click.get()
                if click_flag:
                    # Skip updates when apix is being set from NuFFT click
                    return
                #apix_master.set(val)
                ui.update_slider("apix_slider", value=val, session=session)
                ui.update_text("apix_exact_str", value=str(round(val, 4)), session=session)
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
        
        # Also clear NuFFT calculation state
        nufft_calculation_state.set({
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
            
            # Clear all previous FFT data and trigger complete refresh
            cached_fft_image.set(None)
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request
            fft_widget.set(None)
            
            # Trigger FFT calculation
            base_fft_trigger.set(base_fft_trigger.get() + 1)
            
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
            # Resolution Ring mode: Local maximum search around clicked pixel
            current_state = fft_state.get()
            click_x = current_state.get('resolution_click_x')
            click_y = current_state.get('resolution_click_y')
            
            if click_x is None or click_y is None:
                print("No click location available for Resolution Ring autocorrect.")
                return
                
            # Get the cached FFT image data
            cached_fft = cached_fft_image.get()
            if cached_fft is None:
                print("No FFT image available for autocorrect.")
                return
                
            # Convert PIL image to numpy array for processing
            fft_array = np.array(cached_fft)
            if len(fft_array.shape) == 3:  # RGB image
                fft_array = np.mean(fft_array, axis=2)  # Convert to grayscale
                
            N = fft_array.shape[0]  # Assuming square image
            
            # Search for local maximum in 10x10 neighborhood around clicked location
            search_size = 10
            half_size = search_size // 2
            click_xi, click_yi = int(round(click_x)), int(round(click_y))
            
            
            # Define search bounds
            x_min = max(0, click_xi - half_size)
            x_max = min(N, click_xi + half_size + 1)
            y_min = max(0, click_yi - half_size) 
            y_max = min(N, click_yi + half_size + 1)
            
            # Extract neighborhood and find local maximum
            neighborhood = fft_array[y_min:y_max, x_min:x_max]
            max_idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
            
            # Convert back to full image coordinates
            refined_y = y_min + max_idx[0]
            refined_x = x_min + max_idx[1]
            
            
            # Calculate the refined radius from image center
            center_x, center_y = N / 2, N / 2
            refined_radius = np.sqrt((refined_x - center_x)**2 + (refined_y - center_y)**2)
            
            # Store tuned resolution ring and update FFT state with refined coordinates
            tuned_resolution_radius.set(refined_radius)
            
            # Update FFT state with refined click coordinates so the cyan ring moves
            updated_state = current_state.copy()
            updated_state['resolution_click_x'] = refined_x
            updated_state['resolution_click_y'] = refined_y
            updated_state['resolution_radius'] = refined_radius
            fft_state.set(updated_state)
            
            print(f"Local maximum search: radius {np.sqrt((click_x - center_x)**2 + (click_y - center_y)**2):.3f} -> {refined_radius:.3f}")
            print(f"Updated click coordinates: ({click_x:.1f}, {click_y:.1f}) -> ({refined_x}, {refined_y})")
            
            # Calculate and update apix based on refined radius
            calc_state = fft_calculation_state.get()
            try:
                resolution_type = input.resolution_type()
                custom_resolution = input.custom_resolution()
            except:
                # Fallback to calc_state values
                resolution_type = calc_state['resolution_type']
                custom_resolution = calc_state['custom_resolution']
            
            resolution, _ = get_resolution_info(resolution_type, custom_resolution)
            if resolution is not None:
                # Calculate apix based on refined radius
                fft_image_size = cached_fft.size[0]  # PIL image size
                calculated_apix = (refined_radius * resolution) / fft_image_size

                # Update the apix slider with the refined value
                ui.update_slider("apix_slider", value=calculated_apix, session=session)
                ui.update_text("apix_exact_str", value=f"{calculated_apix:.4f}", session=session)
            
            print(f"Autocorrect completed using local maximum search.")
                
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
                
                # Calculate the angle between major axis and x-axis
                # theta is the angle of the first axis (a) from x-axis
                # We need to determine if the first axis (a) is the minor or major axis
                if a >= b:
                    # a is the major axis, so theta is already the major axis angle
                    major_axis_angle = theta
                else:
                    # a is the minor axis, so major axis is perpendicular (add π/2)
                    major_axis_angle = theta + math.pi/2
                
                # Normalize angle to [-π/2, π/2] range
                while major_axis_angle > math.pi/2:
                    major_axis_angle -= math.pi
                while major_axis_angle < -math.pi/2:
                    major_axis_angle += math.pi
                
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
                
                # Store ellipse orientation (major axis angle) for display purposes
                return (small_axis, large_axis, tilt_angle, untilted_apix, major_axis_angle)
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
    @reactive.event(selected_file_index, uploaded_files_data)
    def _():
        """Update image data when a file is selected from uploaded files."""
        files_data = uploaded_files_data.get()
        selected_idx = selected_file_index.get()
        
        # Check if we have valid files and selection
        if not files_data or selected_idx >= len(files_data):
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            cached_fft_image.set(None)
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request
            fft_widget.set(None)  # Clear FFT widget
            base_fft_trigger.set(0)  # Also reset base trigger for clean state
            return
            
        # Get the selected file
        selected_file = files_data[selected_idx]
        file_info = selected_file['file_info']
        
        # Create a temporary path for the selected file
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_info['name'])
        
        # Write the file content to temp path  
        # Debug: Check the structure of file_info
        
        # Shiny file upload structure - use the correct key for content
        if 'contents' in file_info:
            file_content = file_info['contents']
        elif 'content' in file_info:
            file_content = file_info['content']  
        elif 'datapath' in file_info:
            # If there's a datapath, copy from there
            import shutil
            shutil.copy2(file_info['datapath'], temp_path)
            file_content = None
        else:
            print(f"ERROR: Could not find file content in file_info keys: {list(file_info.keys())}")
            return
            
        if file_content is not None:
            with open(temp_path, 'wb') as f:
                f.write(file_content)
        
        import pathlib
        path = pathlib.Path(temp_path)
        
        if not path.exists():
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)
            image_zoom_state.set({'x_range': None, 'y_range': None, 'is_zoomed': False, 'drawn_region': None})
            cached_fft_image.set(None)
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request
            fft_widget.set(None)  # Clear FFT widget
            base_fft_trigger.set(0)  # Also reset base trigger for clean state

            
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
            original_filename = file_info["name"]
            print(f"Original filename: {original_filename}")
            
            # Extract nominal apix from filename and update the textbox and slider
            nominal_value = extract_nominal(original_filename)
            ui.update_numeric("nominal_apix", value=nominal_value, session=session)
            ui.update_slider("apix_slider", value=nominal_value, session=session)
            ui.update_text("apix_exact_str", value=f"{nominal_value:.3f}", session=session)
            print(f"Extracted nominal apix from filename: {nominal_value:.2f}")
            print(f"Set apix slider and exact value to match nominal apix: {nominal_value:.3f}")
            
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
            base_fft_trigger.set(0)  # Also reset base trigger for clean state
            cached_fft_image.set(None)
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request
            fft_widget.set(None)  # Clear FFT widget to prevent appending to previous widget
            drawn_shapes.set([])
            
            # Clear FFT calculation state when a new image is loaded
            # This will make FFT displays empty until user calculates FFT for the new image
            fft_calculation_state.set({
                'region': None,
                'apix': None,
                'resolution_type': None,
                'custom_resolution': None
            })
            
            # Also clear NuFFT calculation state
            nufft_calculation_state.set({
                'region': None,
                'apix': None,
                'resolution_type': None,
                'custom_resolution': None
            })
            
            # Clear all overlay storage
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

    # Helper function for URL downloading logic
    def download_image_from_url(url):
        """Download and process image from URL."""
        if not url or not url.strip():
            return False
            
        url = url.strip()
        print(f"Attempting to download image from URL: {url}")
        
        try:
            # Download the image
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or content-type
            parsed_url = urlparse(url)
            url_path = parsed_url.path
            if '.' in url_path:
                file_ext = os.path.splitext(url_path)[1].lower()
            else:
                # Try to determine from content-type
                content_type = response.headers.get('content-type', '').lower()
                if 'png' in content_type:
                    file_ext = '.png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    file_ext = '.jpg'
                elif 'tiff' in content_type or 'tif' in content_type:
                    file_ext = '.tif'
                else:
                    file_ext = '.png'  # Default
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            # Load image using the same logic as file upload
            print(f"Loading downloaded image from: {temp_path}")
            original_data, target_apix, original_apix = get_image(temp_path)
            print(f"Image loaded successfully: original_shape={original_data.shape}, apix={target_apix}")
            
            # Create binned version for display
            binned_data = bin_image(original_data, target_size=1000)
            print(f"Created binned version for display: binned_shape={binned_data.shape}")
            
            # Extract filename from URL
            original_filename = os.path.basename(parsed_url.path) or "downloaded_image" + file_ext
            
            # Extract nominal apix from filename and update UI
            nominal_value = extract_nominal(original_filename)
            ui.update_numeric("nominal_apix", value=nominal_value, session=session)
            ui.update_slider("apix_slider", value=nominal_value, session=session)
            ui.update_text("apix_exact_str", value=f"{nominal_value:.3f}", session=session)
            print(f"Extracted nominal apix from filename: {nominal_value:.2f}")
            
            # Set the image data (same as upload handler)
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
            base_fft_trigger.set(0)  # Also reset base trigger for clean state
            cached_fft_image.set(None)
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request
            fft_widget.set(None)  # Clear FFT widget to prevent appending to previous widget
            drawn_shapes.set([])
            
            # Clear FFT calculation state when a new image is loaded
            # This will make FFT displays empty until user calculates FFT for the new image
            fft_calculation_state.set({
                'region': None,
                'apix': None,
                'resolution_type': None,
                'custom_resolution': None
            })
            
            # Also clear NuFFT calculation state
            nufft_calculation_state.set({
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
            
            print(f"Successfully loaded image from URL: {original_filename}")
            
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file: {cleanup_error}")
            
            return True
                
        except Exception as e:
            print(f"Error downloading image from URL: {e}")
            import traceback
            traceback.print_exc()
            # Clear image data on error
            raw_image_data.set({'img': None, 'data': None})
            image_data.set(None)
            image_filename.set(None)
            original_image_data.set(None)
            binned_image_data.set(None)
            return False

    @reactive.Effect
    @reactive.event(input.download_url)
    def _():
        """Handle URL download when download_url changes."""
        # Only download if URL mode is selected
        if input.input_method() != "URL":
            return
            
        url = input.download_url()
        download_image_from_url(url)
    
    @reactive.Effect
    def _():
        """Auto-download preset URL on app startup when URL mode is selected."""
        # This effect runs when the app starts and input_method is available
        try:
            if input.input_method() == "URL":
                url = input.download_url()
                # if url:  # Only download if URL is not empty
                    # print("Auto-downloading preset URL on app startup...")
                    # download_image_from_url(url)
        except Exception as e:
            print(f"Auto-download on startup failed: {e}")

    @reactive.Effect
    def _():
        """Set up initial region when image loads."""
        current_image = binned_image_data.get()
        current_zoom_state = image_zoom_state.get()
        
        # If we have image data but no region set up yet, initialize the pre-selected region
        if (current_image is not None and 
            current_zoom_state.get('drawn_region') is None):
            
            print("🔧 Setting up initial pre-selected region...")
            
            # Set up the initial region coordinates 
            initial_coords = {
                'x0': 300,
                'x1': 700,
                'y0': 300,
                'y1': 700
            }
            box_coordinates.set(initial_coords)
            
            # Update zoom state with the initial region
            new_zoom_state = current_zoom_state.copy()
            new_zoom_state['drawn_region'] = initial_coords
            new_zoom_state['is_zoomed'] = True
            image_zoom_state.set(new_zoom_state)
            
            print(f"✅ Pre-selected initial region: X[300,700] Y[300,700]")

    @reactive.Effect
    def _():
        """Auto-calculate FFT when image and region are ready."""
        current_image = binned_image_data.get()
        current_zoom_state = image_zoom_state.get()
        current_fft = cached_fft_image.get()
        current_calc_state = fft_calculation_state.get()
        
        # Only auto-calculate if we have image, region, but no FFT yet
        if (current_image is not None and 
            current_zoom_state.get('drawn_region') is not None and
            current_zoom_state.get('is_zoomed') == True and
            current_fft is None and 
            current_calc_state.get('region') is None):
            
            print("🚀 Auto-calculating FFT for pre-selected region...")
            print(f"   Current zoom state drawn_region: {current_zoom_state.get('drawn_region')}")
            
            # Trigger FFT calculation by incrementing the base trigger (same as manual Calc FFT button)
            current_trigger = base_fft_trigger.get()
            base_fft_trigger.set(current_trigger + 1)
            
            # Also trigger autoscale (same as manual Calc FFT button)
            autoscale_trigger.set(autoscale_trigger.get() + 1)
            
            print("✅ Automatic FFT calculation triggered")

    # FFT calculation is now done directly in fft_with_circle widget function

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
            selected=dict(marker=dict(opacity=0.0, color='blue', size=4)),
            unselected=dict(marker=dict(opacity=0.0)),
        )

        figw.add_trace(scatter)

 

        # figw.update_layout(dragmode='select',
        #           newselection=dict(line=dict(color='blue')))
        
        # Configure layout for box selection
        figw.update_layout(
            # Enable box selection mode (equivalent to R's dragmode = "select")
            dragmode='select',
            newselection=dict(line=dict(color='blue',width=4)),
            # Configure selection behavior
            selectdirection='any',
            # newselection=dict(
            #     mode='immediate'
            # ),
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
        
        figw.add_selection(x0=300, y0=300, x1=700, y1=700,line=dict(color='blue',width=4))
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
    @render.data_frame  
    def upload_files_table():
        """Render table for uploaded files with File Name and Nominal Apix columns."""
        import pandas as pd
        
        # Get uploaded files data
        files_data = uploaded_files_data.get()
        
        if not files_data:
            # Create empty DataFrame with the required columns
            empty_df = pd.DataFrame({
                'File Name': [],
                'Nominal Size': []
            })
            return empty_df
        
        # Create DataFrame from files data
        df = pd.DataFrame([
            {
                'File Name': file_info['name'], 
                'Nominal Size': file_info['nominal_apix']
            }
            for file_info in files_data
        ])
        
        # Return with selection enabled
        from shiny import render
        return render.DataGrid(
            df, 
            selection_mode="row",
            filters=False,
            width={"File Name": "70%", "Nomimal Size": "30%"}
        )

    # Create a reactive calc that only depends on essential FFT data
    @reactive.calc
    @reactive.event(cached_fft_image, image_data)
    def fft_widget_data():
        """Calculate FFT widget data only when essential data changes"""
        from shiny import req
        req(image_data.get() is not None)

        # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            return None

        # Use the cached FFT image (already has current contrast applied)
        fft_img = cached_fft.copy()
        # Convert PIL image to numpy array for Plotly
        fft_arr = np.array(fft_img.convert('L')).astype(np.uint8)

        # Get resolution parameters for hover text (included in data calculation)
        nominal_apix = float(input.nominal_apix()) if input.nominal_apix() else 1.0
        current_resolution_type = input.resolution_type()
        current_custom_resolution = input.custom_resolution()

        # Get resolution from resolution type or custom value
        if current_resolution_type and current_resolution_type != "Custom":
            resolution_map = {
                "Graphene (2.13 Å)": 2.13,
                "Graphene (100)": 2.13,
                "Graphene (110)": 1.23,
                "Gold (2.355 Å)": 2.355,
                "Gold (111)": 2.35,
                "Gold (200)": 2.04,
                "Gold (220)": 1.44,
                "Ice (3.661 Å)": 3.661
            }
            target_resolution = resolution_map.get(current_resolution_type, 2.13)
        else:
            target_resolution = current_custom_resolution if current_custom_resolution else 2.13

        return {
            'fft_arr': fft_arr,
            'nominal_apix': nominal_apix,
            'target_resolution': target_resolution
        }

    @output
    @render_widget
    def fft_with_circle():
        widget_data = fft_widget_data()
        if widget_data is None:
            return None

        fft_arr = widget_data['fft_arr']
        nominal_apix = widget_data['nominal_apix']
        target_resolution = widget_data['target_resolution']

        print("=== CREATING NEW FFT WIDGET (should only happen on Calc FFT) ===")

        # Create the FFT figure manually to ensure click events work
        # Add unique identifier to force complete recreation
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
            np.arange(0, fft_arr.shape[0], 2),  # Every 2 pixels for performance
            np.arange(0, fft_arr.shape[1], 2),  # Every 2 pixels for performance
            indexing='ij'
        )

        # Calculate tentative apix for each point in the grid
        # Use resolution parameters from widget data (no direct input dependencies)
            
        # Calculate tentative apix for each grid point
        # Distance from center to each point
        center_x, center_y = fft_arr.shape[1] / 2, fft_arr.shape[0] / 2
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Convert distance to spatial frequency (1/Å)
        # For FFT: spatial_freq = distance_in_pixels / (image_size_in_pixels * apix)
        fft_image_size = fft_arr.shape[0]  # Assuming square
        target_spatial_freq = 1.0 / target_resolution
        
        # Calculate tentative apix: what apix would make this distance correspond to target resolution
        # From: spatial_freq = distance / (size * apix)
        # Solve for apix: apix = distance / (size * spatial_freq)
        tentative_apix_array = distances / (fft_image_size * target_spatial_freq)
        
        # Avoid division by zero for center point
        tentative_apix_array = np.where(distances == 0, nominal_apix, tentative_apix_array)
        
        scatter_trace = go.Scatter(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            mode='markers',
            marker=dict(
                size=3,  # Small size for dense grid
                opacity=0,  # Completely transparent
                color='rgba(0,0,0,0)'  # Valid transparent color
            ),
            hoverinfo='skip',  # Enable hover for click events
            showlegend=False,
            hovertemplate='<b>Distance from center:</b> %{customdata[0]:.1f} px<br>' +
                         '<b>Tentative Apix:</b> %{customdata[1]:.4f} Å/px<br>' +
                         '<b>For resolution:</b> ' + f'{target_resolution:.3f} Å<extra></extra>',
            customdata=np.column_stack([distances.flatten(), tentative_apix_array.flatten()])
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
        
        # Force square display but allow arbitrary zoom box ratios
        fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
        fig.update_yaxes(constrain="domain")

        # Note: All shapes (circles, measurements, lattice points) are now handled by 
        # separate overlay effects to avoid re-rendering base FFT on state changes
        
        # Configure interactive layout
        fig.update_layout(
            height=None,  # Allow natural sizing like original image
            margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins
            autosize=True,
            dragmode='zoom',  # Keep zoom mode as default
            modebar=dict(
                add=[ 'zoom', 'pan', 'reset+autorange'],
                remove=['select2d', 'lasso2d'],
                bgcolor='rgba(255,255,255,0.8)',
                color='black',
                activecolor='red'
            ),
            # Use stable uirevision to prevent unnecessary re-renders
            uirevision="fft-widget-stable",
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
                    # Find local maximum around click point
                    N = fft_arr.shape[0]
                    cx = cy = N/2
                    
                    # Search for local maximum in 5x5 neighborhood around click
                    search_size = 5
                    half_size = search_size // 2
                    click_xi, click_yi = int(round(click_x)), int(round(click_y))
                    
                    # Define search bounds
                    x_min = max(0, click_xi - half_size)
                    x_max = min(N, click_xi + half_size + 1)
                    y_min = max(0, click_yi - half_size) 
                    y_max = min(N, click_yi + half_size + 1)
                    
                    # Extract neighborhood
                    neighborhood = fft_arr[y_min:y_max, x_min:x_max]
                    
                    # Find local maximum
                    max_idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
                    max_y_local, max_x_local = max_idx
                    
                    # Convert back to full image coordinates
                    max_x = x_min + max_x_local
                    max_y = y_min + max_y_local
                    
                    
                    # Calculate circle radius from image center to the local maximum
                    r = ((max_x-cx)**2 + (max_y-cy)**2)**0.5
                    
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
                        ui.update_text("apix_exact_str", value=f"{calculated_apix:.4f}", session=session)
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
                    new_state['resolution_click_x'] = max_x  # Store local maximum coordinates
                    new_state['resolution_click_y'] = max_y  # Store local maximum coordinates
                    fft_state.set(new_state)
                    
                    # Update the figure with the new shape but keep zoom mode
                    with fw.batch_update():
                        fw.layout.shapes = [new_circle]
                        fw.layout.dragmode = 'zoom'  # Keep zoom mode after adding circle
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
    @reactive.event(base_fft_trigger)
    def _():
        """Update cached FFT images when FFT is manually triggered (Calc FFT button)."""
        print("FFT calculation triggered - checking drawn region")
        
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
            
            # Also store the same state for NuFFT calculations - use nominal apix
            nufft_calculation_state.set({
                'region': region,
                'apix': float(input.nominal_apix()),
                'resolution_type': input.resolution_type(),
                'custom_resolution': input.custom_resolution()
            })
            
            print("✅ FFT image cached successfully")
        else:
            print("❌ No region available for FFT calculation")

    # Track if NuFFT calculation has been requested
    nufft_calculation_requested = reactive.Value(False)
    
    @reactive.Effect
    @reactive.event(input.nufft_r_sampling_freq, input.nufft_theta_sampling_freq, input.nufft_display_range, nufft_calculation_requested)
    def calculate_nufft_when_requested():
        """Calculate NuFFT data ONLY when sampling parameters change or explicitly requested."""
        if not nufft_calculation_requested.get():
            return
        
        # Get current slider values
        r_freq = input.nufft_r_sampling_freq()
        theta_freq = input.nufft_theta_sampling_freq()
        display_range = input.nufft_display_range()
        
        try:
            # Check if we have the required NuFFT calculation state
            calc_state = nufft_calculation_state.get()
            if calc_state['region'] is None:
                return
            
            # Get current parameters for NuFFT calculation - use NOMINAL apix only
            current_apix = float(input.nominal_apix())
            
            # Get target resolution and create ±10% range around it
            target_resolution, _ = get_resolution_info(calc_state['resolution_type'], calc_state['custom_resolution'])
            if target_resolution is None:
                target_resolution = 2.13  # Default fallback
            
            # Create ±display_range% range around target resolution
            res_margin = display_range / 100.0  # Convert percentage to fraction
            res_low = target_resolution * (1 - res_margin)   # e.g., 2.13 * (1-0.03) = 2.066 Å
            res_high = target_resolution * (1 + res_margin)  # e.g., 2.13 * (1+0.03) = 2.194 Å
            
            # Convert to spatial frequency (1/Å)
            freq_low = 1.0 / res_high   # 1/2.343 = 0.427 1/Å
            freq_high = 1.0 / res_low   # 1/1.917 = 0.522 1/Å
            
            # Add timing for tab switching
            import time
            tab_switch_time = time.time()
            print(f"🕒 TAB SWITCH TIMING: NuFFT calculation requested at {tab_switch_time:.3f}")
            
            print(f"🔄 Recalculating NuFFT data (r_freq={r_freq}, theta_freq={theta_freq}, range={display_range}%)...")
            print(f"🎯 NuFFT range: {res_low:.3f} - {res_high:.3f} Å (±{display_range:.1f}% around {target_resolution:.3f} Å)")
            
            # Calculate samples from current slider values 
            region_size = min(calc_state['region'].size)  # Get smaller dimension
            r_samples_uncapped = int((0.5 * region_size) * r_freq)
            theta_samples_uncapped = int(360 * theta_freq)
            
            r_samples = min(r_samples_uncapped, 10000)  # Cap at 3000 for better resolution
            theta_samples = min(theta_samples_uncapped, 1800)  # Cap at 1800 for better angular resolution
            
            # Show if capping occurred
            r_capped = r_samples_uncapped > 10000
            theta_capped = theta_samples_uncapped > 1800
            print(f"🔧 NuFFT calculation params:")
            print(f"   r_samples: {r_samples}{' (CAPPED from ' + str(r_samples_uncapped) + ')' if r_capped else ''} → affects power curve smoothness")
            print(f"   theta_samples: {theta_samples}{' (CAPPED from ' + str(theta_samples_uncapped) + ')' if theta_capped else ''} → affects heatmap angular resolution")
            print(f"   region_size: {region_size}")
            if r_capped:
                print(f"   ⚠️  Radial sampling is capped - reduce r_freq slider to see changes in power curve detail")
            if theta_capped:
                print(f"   ⚠️  Angular sampling is capped - reduce theta_freq slider to see changes in heatmap resolution")
            
            # Process the region using FAST NuFFT method
            core_calc_start = time.time()
            print(f"🚀 Starting FAST NuFFT core calculation at {core_calc_start:.3f}...")
            
            pwr_curve, pwr2d_raw = calibrateMag_process_one_region_fast(
                region_data=calc_state['region'],
                apix=current_apix,
                res_low=res_low,
                res_high=res_high,
                r_samples=r_samples,
                theta_samples=theta_samples
            )
            
            core_calc_end = time.time()
            core_calc_duration = core_calc_end - core_calc_start
            print(f"🏁 FAST NuFFT core calculation completed in {core_calc_duration:.3f} seconds")
            
            # Cache the NuFFT heatmap data
            cached_nufft_heatmap_data.set({
                'pwr2d_raw': pwr2d_raw,
                'r_samples': r_samples,
                'theta_samples': theta_samples,
                'res_low': res_low,
                'res_high': res_high,
                'apix': current_apix,
                'target_resolution': target_resolution,
                'freq_low': freq_low,
                'freq_high': freq_high,
                'display_range': display_range
            })
            
            # Cache the NuFFT power curve data  
            cached_nufft_power_data.set({
                'pwr_curve': pwr_curve,
                'pwr2d_raw': pwr2d_raw,
                'r_samples': r_samples,
                'theta_samples': theta_samples,
                'res_low': res_low,
                'res_high': res_high,
                'apix': current_apix,
                'target_resolution': target_resolution,
                'freq_low': freq_low,
                'freq_high': freq_high,
                'display_range': display_range
            })
            
            # Add complete timing measurement
            total_process_end = time.time()
            total_duration = total_process_end - tab_switch_time
            caching_duration = total_process_end - core_calc_end
            
            print(f"💾 Data caching completed in {caching_duration:.3f} seconds")
            print(f"⏱️  TOTAL TAB SWITCH TO DATA READY: {total_duration:.3f} seconds")
            print("✅ NuFFT data calculated and cached successfully")
            
            # Clear green vertical lines from existing power curve widget since we have new data
            existing_widget = nufft_power_widget.get()
            if existing_widget is not None:
                try:
                    current_shapes = list(existing_widget.layout.shapes) if existing_widget.layout.shapes else []
                    preserved_shapes = [shape for shape in current_shapes 
                                      if not (hasattr(shape, 'line') and 
                                             hasattr(shape.line, 'color') and 
                                             shape.line.color == 'green')]
                    existing_widget.layout.shapes = preserved_shapes
                except Exception as e:
                    print(f"Note: Could not clear green lines from existing widget: {e}")
            
        except Exception as e:
            print(f"❌ Error calculating NuFFT data: {e}")
            import traceback
            traceback.print_exc()
            # Clear cache on error
            cached_nufft_heatmap_data.set(None)
            cached_nufft_power_data.set(None)
            nufft_calculation_requested.set(False)  # Reset calculation request

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
            return
        
        lattice_points = lattice_points_storage.get()
        tuned_markers = tuned_markers_storage.get()
        current_mode = current_mode_storage.get()
        
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
                
            # Add tuned markers as red crosshairs
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
        # else:
        
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
                widget.layout.xaxis.constrain = "domain"
                widget.layout.yaxis.constrain = "domain"
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
    
    # @render_plotly("fft_1d_plot")
    # def fft_1d_plot():
    #     # Check if FFT has been calculated
    #     cached_fft = cached_fft_image.get()
    #     if cached_fft is None:
    #         # Return None to show nothing when no FFT has been calculated
    #         return None
    #     
    #     # Get the calculated plot data
    #     plot_data = fft_1d_data()
    #     if plot_data is None:
    #         return go.Figure()
    #     
    #     # Use stored FFT calculation state instead of current region/zoom/resolution
    #     # This prevents replotting when regions are drawn or parameters change
    #     calc_state = fft_calculation_state.get()
    #     if calc_state['region'] is None:
    #         return go.Figure()
    #     
    #     # Get stored region, zoom, and resolution from calculation state
    #     region = calc_state['region']
    #     zoom = plot_zoom.get()  # Keep zoom state for interactive zoom/pan
    #     resolution, _ = get_resolution_info(calc_state['resolution_type'], calc_state['custom_resolution'])
    #     
    #     # Get shared x-axis range
    #     shared_range = shared_x_range.get()
    #     
    #     # Create plotly figure using the original data (no filtering)
    #     fig = create_fft_1d_plotly_figure(
    #         plot_data=plot_data,
    #         resolution=resolution,
    #         region=region,
    #         size=size,
    #         zoom_state=zoom,
    #         shared_x_range=shared_range
    #     )
    #     
    #     # Create FigureWidget from the Figure and return it
    #     fw = FigureWidget(fig)
    #     
    #     # Add relayout callback to sync x-axis with heatmap
    #     def on_1d_relayout(layout_data, fig):
    #         if ('xaxis.range[0]' in layout_data and 'xaxis.range[1]' in layout_data and 
    #             range_update_source.get() != '1d'):
    #             new_range = [layout_data['xaxis.range[0]'], layout_data['xaxis.range[1]']]
    #             range_update_source.set('1d')
    #             shared_x_range.set(new_range)
    #             range_update_source.set(None)
    #     
    #     fw.layout.on_change(on_1d_relayout, 'xaxis.range')
    #     
    #     # Store the widget for in-place updates
    #     fft_1d_widget.set(fw)
    #     
    #     return fw
    
    # @render_plotly("fft_polar_heatmap")
    # def fft_polar_heatmap():
    #     pass
    #     # Check if FFT has been calculated
        cached_fft = cached_fft_image.get()
        if cached_fft is None:
            return None
        
        # Get the stored FFT calculation state
        calc_state = fft_calculation_state.get()
        if calc_state['region'] is None:
            return go.FigureWidget()
        
        try:
            # Check if we should use nominal apix (initial) or current apix (after slider changes)
            nominal_apix = float(input.nominal_apix())
            current_apix = get_apix()
            
            # Use nominal apix for initial range, but update with current apix when slider changes
            # If current apix is very close to nominal, use nominal (initial state)
            # Otherwise use current apix (user moved the slider)
            if abs(current_apix - nominal_apix) < 0.01:
                range_apix = nominal_apix
            else:
                range_apix = current_apix
            
            # Compute polar heatmap data using unbinned coordinates
            heatmap_data = compute_fft_polar_heatmap_data(
                region=calc_state['region'],
                apix=range_apix,
                resolution_type=calc_state['resolution_type'],
                custom_resolution=calc_state['custom_resolution']
            )
            
            # Apply log scale to heatmap data if enabled
            z_data = heatmap_data['heatmap_data']
            if input.log_y():
                z_data = np.log1p(z_data)  # log1p is safe for positive values
            
            # Create custom hover text with apix calculations
            region_size = calc_state['region'].size[0]
            resolution = heatmap_data['resolution']
            
            hover_text = []
            for i, angle in enumerate(heatmap_data['angles']):
                row_text = []
                for j, radius in enumerate(heatmap_data['radii']):
                    # Calculate apix for this radius
                    if radius > 0 and resolution > 0:
                        apix_value = (radius * resolution) / region_size
                        apix_str = f"{apix_value:.3f}"
                    else:
                        apix_str = "N/A"
                    
                    hover_info = f"Angle: {angle:.0f}°<br>Radius: {radius:.1f} px<br>Apix: {apix_str} Å/px"
                    row_text.append(hover_info)
                hover_text.append(row_text)
            
            # Create Plotly heatmap (swapped axes: angle on x, radius on y)
            # Need to transpose data: original is (angles x radii), but Plotly expects (radii x angles) for x=angles, y=radii
            fig = go.Figure(data=go.Heatmap(
                z=z_data.T,  # Transpose: (angles x radii) -> (radii x angles)
                x=heatmap_data['angles'],
                y=heatmap_data['radii'],
                colorscale='viridis',
                hoverongaps=False,
                hovertemplate='%{text}<extra></extra>',
                text=[[hover_text[j][i] for j in range(len(hover_text))] for i in range(len(hover_text[0]))]  # Transpose hover text too
            ))
            
            # Get shared x-axis range or use data range
            shared_range = shared_x_range.get()
            if shared_range is not None:
                x_range = shared_range
            else:
                x_range = [heatmap_data['radii'][0], heatmap_data['radii'][-1]]
            
            # Calculate target radius for auto-zoom
            # Get resolution from current input (not cached state) for real-time updates
            current_resolution_type = input.resolution_type()
            current_custom_resolution = input.custom_resolution()
            if current_resolution_type and current_resolution_type != "Custom":
                resolution_map = {
                    "Graphene (2.13 Å)": 2.13,
                    "Graphene (100)": 2.13,
                    "Graphene (110)": 1.23,
                    "Gold (2.355 Å)": 2.355,
                    "Gold (111)": 2.35,
                    "Gold (200)": 2.04,
                    "Gold (220)": 1.44,
                    "Ice (3.661 Å)": 3.661
                }
                target_resolution = resolution_map.get(current_resolution_type, 2.13)
            else:
                target_resolution = current_custom_resolution if current_custom_resolution else 2.13
            
            # Get nominal apix and region size
            nominal_apix = float(input.nominal_apix())
            region_size = calc_state['region'].size[0]  # Unbinned region size
            
            # Calculate target radius in pixels using the same formula as resolution_to_radius
            # radius = (image_size * apix) / resolution
            target_radius = (region_size * nominal_apix) / target_resolution
            
            print(f"Auto-zoom calculation: resolution={target_resolution:.2f} Å, apix={nominal_apix:.3f} Å/px, region_size={region_size}")
            print(f"Calculated target radius: {target_radius:.1f} px")
            
            # Set zoom range: ±10 pixels around target radius
            zoom_margin = 10
            y_min = max(heatmap_data['radii'][0], target_radius - zoom_margin)
            y_max = min(heatmap_data['radii'][-1], target_radius + zoom_margin)
            
            fig.update_layout(
                title=f'FFT Profile Near Frequency of Interest<br>Target: {target_radius:.1f} px ({target_resolution:.2f} Å @ {nominal_apix:.3f} Å/px)',
                xaxis_title='Angle (degrees)',
                yaxis_title='Radius (pixels)',
                height=300,
                width=650,  # Fixed width to match NuFFT plots
                margin=dict(l=60, r=20, t=80, b=40),
                autosize=False,
                xaxis=dict(
                    range=[0, 360],  # Full angle range
                    showgrid=True,
                    showticklabels=True
                ),
                yaxis=dict(
                    range=[y_min, y_max],  # Zoomed to ±15 around target
                    showgrid=True,
                    showticklabels=True
                )
            )
            
            # Check if we should add red circle overlay for Find Max
            max_pos = heatmap_max_position.get()
            if max_pos.get('show_overlay', False) and max_pos.get('radius') is not None:
                # Add small red circle at max position (~1 pixel radius)
                fig.add_shape(
                    type="circle",
                    x0=max_pos['angle'] - 2,  # 1-degree radius circle
                    y0=max_pos['radius'] - 2,  # 1-pixel radius circle  
                    x1=max_pos['angle'] + 2,
                    y1=max_pos['radius'] + 2,
                    line_color="red",
                    line_width=2,
                    fillcolor="rgba(255,0,0,0.3)"
                )
            
            # Create FigureWidget
            fw = FigureWidget(fig)
            
            return fw
            
        except Exception as e:
            print(f"Error creating polar heatmap: {e}")
            return go.FigureWidget()
    
    @output
    @render_widget
    @reactive.event(nufft_show_focused_heatmap, nufft_clicked_frequency, input.nufft_r_sampling_freq, input.nufft_theta_sampling_freq)
    def nufft_heatmap():
        from shiny import req
        import time
        plot_start = time.time()
        print(f"🎨 HEATMAP PLOT START: {plot_start:.3f}")
        
        # Check if we should show focused heatmap
        show_focused = nufft_show_focused_heatmap.get()
        clicked_freq = nufft_clicked_frequency.get()
        
        if not show_focused:
            # Initial state: Show placeholder message encouraging user to click
            print("🎯 INITIAL STATE: Showing click-to-generate-heatmap placeholder")
            
            fig = go.Figure()
            fig.add_annotation(
                text="<b>📊 Interactive Heatmap</b><br><br>" +
                     "👆 Click on the power curve above<br>" +
                     "to generate a focused heatmap<br>" +
                     "around your selected frequency<br><br>" +
                     "🎯 This saves time by showing<br>" +
                     "only the relevant data slice!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=14, color="gray"),
                bgcolor="rgba(240,240,240,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
            fig.update_layout(
                title="Click Power Curve to Generate Focused Heatmap",
                height=300,
                width=650,
                margin=dict(l=80, r=20, t=60, b=40),
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            plot_end = time.time()
            print(f"🏁 PLACEHOLDER HEATMAP completed in {plot_end - plot_start:.3f} seconds")

            # Clear widget for placeholder state
            nufft_heatmap_widget.set(None)

            return go.FigureWidget(fig)
        
        # Focused mode: Generate heatmap around clicked frequency
        print(f"🎯 FOCUSED MODE: Generating heatmap around frequency {clicked_freq:.6f} 1/Å")
        
        # Require NuFFT calculation state to exist
        calc_state = nufft_calculation_state.get()
        req(calc_state['region'] is not None)
        
        # Request NuFFT calculation if not already done
        nufft_calculation_requested.set(True)
        
        # Get cached data
        cached_heatmap_data = cached_nufft_heatmap_data.get()
        if cached_heatmap_data is None:
            print("⏳ Waiting for NuFFT heatmap data...")
            nufft_heatmap_widget.set(None)
            return go.FigureWidget()  # Return empty widget while calculating
        
        print("📊 Using NuFFT heatmap data for FOCUSED rendering")
        
        pwr = cached_heatmap_data['pwr2d_raw']
        r_samples = cached_heatmap_data['r_samples']
        theta_samples = cached_heatmap_data['theta_samples']
        res_low = cached_heatmap_data['res_low']
        res_high = cached_heatmap_data['res_high']
        current_apix = cached_heatmap_data['apix']
        # Get current resolution from dropdown instead of cached value for real-time updates
        current_resolution_type = input.resolution_type()
        current_custom_resolution = input.custom_resolution()
        if current_resolution_type and current_resolution_type != "Custom":
            resolution_map = {
                "Graphene (2.13 Å)": 2.13,
                "Graphene (100)": 2.13,
                "Graphene (110)": 1.23,
                "Gold (2.355 Å)": 2.355,
                "Gold (111)": 2.35,
                "Gold (200)": 2.04,
                "Gold (220)": 1.44,
                "Ice (3.661 Å)": 3.661
            }
            target_resolution = resolution_map.get(current_resolution_type, 2.13)
        else:
            target_resolution = current_custom_resolution if current_custom_resolution else 2.13
        freq_low = cached_heatmap_data['freq_low']
        freq_high = cached_heatmap_data['freq_high']
        display_range = cached_heatmap_data['display_range']
        apix_source = f"Nominal Apix: {current_apix:.3f}"
        
        print(f"   pwr shape: {pwr.shape if hasattr(pwr, 'shape') else type(pwr)}")
        print(f"   r_samples: {r_samples}, theta_samples: {theta_samples}")
        print(f"   res_low: {res_low:.3f}, res_high: {res_high:.3f}")
        
        try:
            data_prep_start = time.time()
            
            # NuFFT output shape is (theta_samples, r_samples) from meshgrid
            if len(pwr.shape) > 2:
                heatmap_data = pwr[0]  # Shape: (theta_samples, r_samples)
            else:
                heatmap_data = pwr     # Shape: (theta_samples, r_samples)
            
            print(f"🔍 Original NuFFT shape: {heatmap_data.shape} = (theta={theta_samples}, r={r_samples})")
            
            print(f"  Full heatmap data shape: {heatmap_data.shape}")
            print(f"  Sample from different theta values:")
            for theta_idx in [0, theta_samples//4, theta_samples//2, 3*theta_samples//4]:
                if theta_idx < heatmap_data.shape[0]:
                    sample = heatmap_data[theta_idx, :5]  # First 5 radial values
                    print(f"    θ={theta_idx}: {sample} (range: {heatmap_data[theta_idx, :].min():.3f}-{heatmap_data[theta_idx, :].max():.3f})")
            
            # Check if source data is 1D radial profile repeated for all angles
            first_radial_profile = heatmap_data[0, :]
            source_all_identical = True
            for theta_idx in range(1, min(10, heatmap_data.shape[0])):
                if not np.allclose(heatmap_data[theta_idx, :], first_radial_profile, rtol=1e-6):
                    source_all_identical = False
                    break
            print(f"  🚨 Source data - all angular columns identical: {source_all_identical}")
            
            if source_all_identical:
                print(f"  ⚠️ ROOT CAUSE: Source NuFFT data is 1D radial profile replicated!")
                print(f"  The NuFFT calculation should produce 2D (theta, r) data with variation in both dimensions")
                print(f"  Current data appears to be a 1D radial profile repeated for all angles")
            
            # FOCUSED SLICE: Extract ±2 pixels around clicked frequency  
            # Create frequency array to find the index corresponding to clicked frequency
            res_range_array = np.linspace(res_low, res_high, r_samples)
            spatial_freq_array = 1.0 / res_range_array
            
            # Find the index closest to clicked frequency
            freq_index = np.argmin(np.abs(spatial_freq_array - clicked_freq))
            
            print(f"🔍 Frequency matching:")
            print(f"  clicked_freq: {clicked_freq:.6f}")
            print(f"  freq_index: {freq_index}")
            print(f"  r_samples: {r_samples}")
            print(f"  actual_freq_at_index: {spatial_freq_array[freq_index]:.6f}")
            
            # Extract ±10 pixel slice (21 pixels total) - slice the RADIAL dimension (axis=1)
            slice_width = 10  # ±10 pixels = 21 total pixels
            start_idx = max(0, freq_index - slice_width)
            end_idx = min(r_samples, freq_index + slice_width + 1)
            
            print(f"🔍 Slice calculation:")
            print(f"  slice_width: {slice_width}")
            print(f"  start_idx: {start_idx}")
            print(f"  end_idx: {end_idx}")
            print(f"  slice_size: {end_idx - start_idx}")
            
            # Extract focused slice: all theta, focused radial range
            # Shape: (theta_samples, focused_r_samples) = (1800, 31)
            original_focused_data = heatmap_data[:, start_idx:end_idx]  # All theta, focused radial range
            original_focused_res_range = res_range_array[start_idx:end_idx]
            original_focused_freq_range = spatial_freq_array[start_idx:end_idx]
            
            print(f"  Original focused data shape: {original_focused_data.shape}")
            print(f"  Sample angular profiles (first 5 theta values):")
            for theta_idx in range(min(5, original_focused_data.shape[0])):
                profile = original_focused_data[theta_idx, :]
                print(f"    θ={theta_idx}: {profile[:3]}...{profile[-3:]} (range: {profile.min():.3f}-{profile.max():.3f})")
            
            # Check if all angular columns are identical (vertical stripe issue)
            first_column = original_focused_data[0, :]
            all_identical = True
            for theta_idx in range(1, min(10, original_focused_data.shape[0])):
                if not np.allclose(original_focused_data[theta_idx, :], first_column, rtol=1e-6):
                    all_identical = False
                    break
            print(f"  🚨 All angular columns identical: {all_identical}")
            
            if all_identical:
                print(f"  ⚠️ PROBLEM DETECTED: All angular columns are the same!")
                print(f"  This suggests the NuFFT data extraction is wrong.")
                print(f"  Expected: Each θ should have different radial profile")
                print(f"  Actual: All θ have same radial profile → vertical stripes")
            
            # Get current sampling slider values for enhanced resolution
            r_sampling_freq = input.nufft_r_sampling_freq()
            theta_sampling_freq = input.nufft_theta_sampling_freq()
            
            # Enhanced y-resolution: 21 * r_sampling_freq slider value for display (21 pixels = ±10)
            enhanced_y_resolution = int(21 * r_sampling_freq)
            
            # Enhanced x-resolution: theta_sampling_freq affects angular resolution  
            enhanced_x_resolution = int(theta_sampling_freq * theta_samples)
            
            print(f"🔧 Enhanced y-resolution: 21 × {r_sampling_freq} = {enhanced_y_resolution} display samples")
            print(f"🔧 Enhanced x-resolution: {theta_sampling_freq} × {theta_samples} = {enhanced_x_resolution} angular samples")
            
            # TEMPORARY: Skip interpolation to debug the source data issue
            # Use original focused data directly to see if the stripe issue is in source data or interpolation
            focused_data = original_focused_data
            focused_res_range = original_focused_res_range
            focused_freq_range = 1.0 / focused_res_range
            
            # Create simple angle array
            angles = np.linspace(0, 180, theta_samples, endpoint=False)  # Match NuFFT range 0 to π
            
            print(f"🔧 Original focused slice shape: {focused_data.shape} = ({focused_data.shape[0]}θ × {focused_data.shape[1]}r) [±15 pixels]")
            
            print(f"🎯 FOCUSED SLICE: freq {clicked_freq:.6f} → index {freq_index} → slice [{start_idx}:{end_idx}]")
            print(f"🔧 Original heatmap shape: {heatmap_data.shape}")
            print(f"🔧 Focused slice shape: {focused_data.shape}")
            print(f"🔧 Expected shape: ({end_idx - start_idx}, {heatmap_data.shape[1]})")
            print(f"📊 Frequency range: {focused_freq_range[0]:.6f} - {focused_freq_range[-1]:.6f} 1/Å")
            print(f"📊 Resolution range: {focused_res_range[0]:.3f} - {focused_res_range[-1]:.3f} Å")
            
            # Transpose focused data for display: (theta, r) → (r, theta) 
            heatmap_data = focused_data.T  # Transpose to (r_samples, theta_samples)
            r_samples = heatmap_data.shape[0]  # Number of radial samples in focused slice (21)
            theta_samples = heatmap_data.shape[1]  # Number of angular samples 
            print(f"🔧 Transposed for display: {heatmap_data.shape} = ({r_samples}r × {theta_samples}θ)")
            print(f"🔧 Original slice shape: 31r × {theta_samples}θ for ±15 pixels around clicked frequency")
            
            # Apply log scale if enabled
            z_data = heatmap_data
            if input.nufft_log_y():
                z_data = np.log1p(np.abs(z_data))
            
            # Use the enhanced frequency and angle arrays we already calculated above
            # (focused_res_range, focused_freq_range, and angles are already computed with enhanced resolution)
            
            data_prep_end = time.time()
            print(f"🔧 Data preparation: {data_prep_end - data_prep_start:.3f}s")
            
            # SMART hover text creation - only for top 5% intensity pixels
            hover_start = time.time()
            total_pixels = z_data.shape[0] * z_data.shape[1]
            
            # Calculate 95th percentile threshold for intensity
            intensity_threshold = np.percentile(z_data.flatten(), 95)
            high_intensity_mask = z_data >= intensity_threshold
            high_intensity_count = np.sum(high_intensity_mask)
            
            print(f"🧠 SMART HOVER: Creating detailed hover for {high_intensity_count:,} high-intensity pixels (top 5% of {total_pixels:,})")
            print(f"📊 Intensity threshold: {intensity_threshold:.3f}")
            
            # SMART FOCUSED HOVER - Much fewer pixels now, so we can afford detailed hover
            hover_text_creation_start = time.time()
            total_pixels = z_data.shape[0] * z_data.shape[1]
            
            print(f"✨ FOCUSED HOVER: Creating detailed hover for ALL {total_pixels:,} pixels in focused view")
            
            # Create detailed hover text for all pixels in focused view (it's small now!)
            # Calculate tentative apix for each frequency point (same formula as power curve)
            nominal_apix = float(input.nominal_apix()) if input.nominal_apix() else current_apix
            target_spatial_freq = 1.0 / target_resolution
            tentative_apix_array = nominal_apix * (focused_freq_range / target_spatial_freq)

            hover_text = []
            for i in range(z_data.shape[0]):  # r_samples (now just 3 or so)
                row_text = []
                for j in range(z_data.shape[1]):  # theta_samples
                    if i < len(focused_res_range) and j < len(angles):
                        # Use the same template as nufft_power_curve
                        tentative_apix = tentative_apix_array[i]
                        hover_info = f"🎯 FOCUSED VIEW 🎯 <br><b>Spatial freq: {focused_freq_range[i]:.4f} Å⁻¹<br><b>Tentative Apix:</b> {tentative_apix:.4f} Å/px<br><b>For resolution:</b> {target_resolution:.3f} Å<br>"
                    else:
                        hover_info = f"🎯 FOCUSED VIEW 🎯<br>R-idx: {i}<br>θ-idx: {j}<br>Intensity: {z_data[i,j]:.3f}"
                    row_text.append(hover_info)
                hover_text.append(row_text)
            
            hover_text_creation_end = time.time()
            print(f"✨ FOCUSED HOVER COMPLETED: {hover_text_creation_end - hover_text_creation_start:.3f}s for {total_pixels:,} pixels")
            
            hover_end = time.time()
            hover_duration = hover_end - hover_start
            print(f"🎯 SMART HOVER COMPLETED: {hover_duration:.3f}s for {high_intensity_count:,} pixels")
            
            # Create tick values and labels for focused spatial frequency y-axis
            n_ticks = min(6, len(focused_freq_range))  # Maximum 6 ticks
            if n_ticks > 0:
                tick_indices = np.linspace(0, len(focused_freq_range)-1, n_ticks, dtype=int)
                y_tickvals = tick_indices
                y_ticktext = []
                for idx in tick_indices:
                    if idx < len(focused_res_range):
                        res_val = focused_res_range[idx]
                        y_ticktext.append(f"1/{res_val:.3f}")  # More precision for focused view
                    else:
                        y_ticktext.append("")
            else:
                y_tickvals = []
                y_ticktext = []
            
            # Create Plotly heatmap
            plotly_start = time.time()
            heatmap_obj_start = time.time()
            print(f"📊 Creating Plotly heatmap figure...")
            print(f"🔍 Final z_data shape for heatmap: {z_data.shape}")
            print(f"🔍 Data type: {z_data.dtype}")
            print(f"🔍 Data size in MB: {z_data.nbytes / 1024 / 1024:.4f}")
            print(f"🔍 Shape breakdown: {z_data.shape[0]} radial × {z_data.shape[1]} angular samples")
            print(f"🔍 Data range: min={np.min(z_data):.3f}, max={np.max(z_data):.3f}, mean={np.mean(z_data):.3f}")
            print(f"🔍 Has NaN/Inf: nan={np.any(np.isnan(z_data))}, inf={np.any(np.isinf(z_data))}")
            print(f"🔍 Sample values: {z_data.flat[:5]}")  # First 5 values
            
            heatmap_trace = go.Heatmap(
                z=z_data,
                x=np.arange(z_data.shape[1]),  # Angular samples
                y=np.arange(z_data.shape[0]),  # Focused spatial frequency samples (small!)
                colorscale='viridis',
                showscale=False,  # Remove colorbar
                hoverongaps=False,
                # Rich hover text for focused view - we can afford it now with few pixels
                hovertemplate='%{text}<extra></extra>',
                text=hover_text  # Detailed hover for all pixels in focused view
            )
            
            heatmap_obj_end = time.time()
            print(f"🎨 Heatmap object creation: {heatmap_obj_end - heatmap_obj_start:.3f}s")
            
            figure_start = time.time()
            fig = go.Figure(data=heatmap_trace)
            figure_end = time.time()
            print(f"📋 Figure object creation: {figure_end - figure_start:.3f}s")
            
            plotly_mid = time.time()
            print(f"📈 Total heatmap + figure creation: {plotly_mid - plotly_start:.3f}s")
            
            # FIX: Calculate dimensions for BETTER VISUALIZATION
            # For focused view: 31 × theta_samples (e.g., 31 × 1800) 
            # INCREASE height significantly for better visualization - pixels don't need to be square
            min_visible_height = 350  # Increased for 31 radial samples
            
            # Scale height for better visualization of frequency patterns across more samples
            base_pixel_height = max(min_visible_height // z_data.shape[0], 10)  # At least 10px per radial sample for 31 samples
            max_reasonable_height = 600  # Increased to accommodate 31 samples
            display_height = min(max(z_data.shape[0] * base_pixel_height, min_visible_height), max_reasonable_height)
            
            # Use full available width in container
            display_width = 650  # Full width available in container
            
            print(f"🔧 Enhanced visibility heatmap: {z_data.shape[0]}r × {z_data.shape[1]}θ samples [±15 pixels]")
            print(f"🔧 Display dimensions: {display_width:.0f}w × {display_height}h (increased for better visualization)")
            print(f"🔧 Per-sample size: ~{display_width/z_data.shape[1]:.1f}w × {base_pixel_height}h pixels")
            print(f"🔧 Height range: {min_visible_height}px - {max_reasonable_height}px (no longer square pixels)")
            
            fig.update_layout(
                title=f'🎯 FOCUSED NuFFT: ±15px around {clicked_freq:.6f} 1/Å (Resolution: {1/clicked_freq:.3f} Å)',
                xaxis_title='Angular Samples',
                yaxis_title='Radial Samples (Frequency)', 
                height=display_height,
                width=display_width,
                margin=dict(l=80, r=20, t=50, b=40),
                autosize=False,
                yaxis=dict(
                    tickvals=y_tickvals,
                    ticktext=y_ticktext,
                    tickmode='array'
                )
            )
            
            plotly_layout_end = time.time()
            print(f"🎨 Layout update completed in {plotly_layout_end - plotly_mid:.3f}s")
            
            widget_start = time.time()
            
            # FIX: Ensure the heatmap trace is correctly configured for visibility
            print(f"🔧 Final heatmap trace validation before FigureWidget creation...")
            print(f"🔍 Trace z data: {np.array(heatmap_trace.z).shape}, type: {type(heatmap_trace.z)}")
            print(f"🔍 Z data sample: {np.array(heatmap_trace.z).flat[:5]}")
            print(f"🔍 Colorscale: {heatmap_trace.colorscale}")
            print(f"🔍 X array length: {len(heatmap_trace.x) if hasattr(heatmap_trace.x, '__len__') else 'scalar'}")
            print(f"🔍 Y array length: {len(heatmap_trace.y) if hasattr(heatmap_trace.y, '__len__') else 'scalar'}")
            
            # FIX: Explicitly set the visible property and ensure proper data format
            heatmap_trace.visible = True
            
            # FIX: Recreate figure with explicit data validation
            if np.any(np.isnan(z_data)) or np.any(np.isinf(z_data)):
                print("⚠️ WARNING: NaN/Inf values detected, cleaning data...")
                z_data = np.nan_to_num(z_data, nan=0.0, posinf=np.max(z_data[np.isfinite(z_data)]), 
                                     neginf=np.min(z_data[np.isfinite(z_data)]))
                heatmap_trace.z = z_data
            
            # FIX: Try creating the FigureWidget directly with data
            try:
                # Method 1: Direct FigureWidget creation
                widget = go.FigureWidget(data=[heatmap_trace])
                print(f"✅ FigureWidget created using Method 1 (direct data)")
            except Exception as e1:
                print(f"❌ Method 1 failed: {e1}")
                try:
                    # Method 2: Create Figure first, then convert
                    widget = go.FigureWidget(fig)
                    print(f"✅ FigureWidget created using Method 2 (via Figure)")
                except Exception as e2:
                    print(f"❌ Method 2 failed: {e2}")
                    # Method 3: Minimal fallback
                    widget = go.FigureWidget()
                    widget.add_heatmap(z=z_data, colorscale='viridis')
                    print(f"✅ FigureWidget created using Method 3 (add_heatmap)")
            
            widget_end = time.time()
            print(f"🔧 FigureWidget creation: {widget_end - widget_start:.3f}s")
            
            # FIX: Force update the layout on the widget directly with proper sizing
            layout_height = display_height  # Use calculated display height
            print(f"🔧 Final layout height: {layout_height}px (required minimum: {min_visible_height}px)")
            
            widget.update_layout(
                title=dict(
                    text=f'🎯 FOCUSED NuFFT: ±10px around {clicked_freq:.6f} 1/Å (Resolution: {1/clicked_freq:.3f} Å)',
                    font=dict(size=12)
                ),
                xaxis=dict(
                    title='Angular Samples',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                yaxis=dict(
                    title='Radial Samples (Frequency)',
                    tickvals=y_tickvals,
                    ticktext=y_ticktext,
                    tickmode='array',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                height=layout_height,
                width=display_width,
                margin=dict(l=100, r=80, t=60, b=60),  # Increased margins for better visibility
                autosize=False,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Debug the final widget contents
            print(f"🔍 Final widget validation:")
            print(f"  - Data traces: {len(widget.data)}")
            print(f"  - Layout width: {widget.layout.width}")
            print(f"  - Layout height: {widget.layout.height}")
            print(f"  - Layout title: {widget.layout.title.text if hasattr(widget.layout.title, 'text') else widget.layout.title}")
            if len(widget.data) > 0:
                trace = widget.data[0]
                print(f"  - First trace type: {type(trace).__name__}")
                print(f"  - First trace visible: {getattr(trace, 'visible', 'undefined')}")
                print(f"  - First trace z shape: {np.array(trace.z).shape if hasattr(trace, 'z') and trace.z is not None else 'None'}")
            
            plot_end = time.time()
            plot_duration = plot_end - plot_start
            plotly_total = plotly_layout_end - plotly_start
            print(f"📊 Total Plotly operations: {plotly_total:.3f}s")
            print(f"🏁 HEATMAP PLOT COMPLETED in {plot_duration:.3f} seconds")

            # Store widget for click handling
            nufft_heatmap_widget.set(widget)

            return widget
            
        except Exception as e:
            plot_end = time.time()
            plot_duration = plot_end - plot_start
            print(f"❌ HEATMAP PLOT FAILED after {plot_duration:.3f} seconds: {e}")
            import traceback
            traceback.print_exc()
            nufft_heatmap_widget.set(None)
            return go.FigureWidget()
    
    @output
    @render_widget
    def nufft_power_curve():
        from shiny import req
        import time
        power_plot_start = time.time()
        print(f"📈 POWER CURVE PLOT START: {power_plot_start:.3f}")
        
        # Require NuFFT calculation state to exist
        calc_state = nufft_calculation_state.get()
        req(calc_state['region'] is not None)
        
        # Request NuFFT calculation if not already done
        nufft_calculation_requested.set(True)
        
        # Get cached data
        cached_power_data = cached_nufft_power_data.get()
        if cached_power_data is None:
            print("⏳ Waiting for NuFFT power curve data...")
            return go.FigureWidget()  # Return empty widget while calculating
        
        print("📈 Using NuFFT power curve data")
        
        pwr_curve = cached_power_data['pwr_curve']
        pwr = cached_power_data['pwr2d_raw']
        r_samples = cached_power_data['r_samples']
        theta_samples = cached_power_data['theta_samples']
        res_low = cached_power_data['res_low']
        res_high = cached_power_data['res_high']
        current_apix = cached_power_data['apix']
        # Get current resolution from dropdown instead of cached value for real-time updates
        current_resolution_type = input.resolution_type()
        current_custom_resolution = input.custom_resolution()
        if current_resolution_type and current_resolution_type != "Custom":
            resolution_map = {
                "Graphene (2.13 Å)": 2.13,
                "Graphene (100)": 2.13,
                "Graphene (110)": 1.23,
                "Gold (2.355 Å)": 2.355,
                "Gold (111)": 2.35,
                "Gold (200)": 2.04,
                "Gold (220)": 1.44,
                "Ice (3.661 Å)": 3.661
            }
            target_resolution = resolution_map.get(current_resolution_type, 2.13)
        else:
            target_resolution = current_custom_resolution if current_custom_resolution else 2.13
        freq_low = cached_power_data['freq_low']
        freq_high = cached_power_data['freq_high']
        display_range = cached_power_data['display_range']
        apix_source = f"Nominal Apix: {current_apix:.3f}"
        
        print(f"   pwr_curve shape: {pwr_curve.shape if hasattr(pwr_curve, 'shape') else type(pwr_curve)}")
        print(f"   pwr shape: {pwr.shape if hasattr(pwr, 'shape') else type(pwr)}")
        print(f"   r_samples: {r_samples}, theta_samples: {theta_samples}")
        
        try:
            
            # Create resolution range and spatial frequency arrays for x-axis
            res_range_array = np.linspace(res_low, res_high, len(pwr_curve))
            # Convert to spatial frequency (1/Å)
            spatial_freq_array = 1.0 / res_range_array
            
            # Apply log scale if enabled
            y_data = pwr_curve
            
            # # Apply smoothing if enabled
            # if input.nufft_smooth() and len(y_data) > input.nufft_window_size():
            #     window_size = input.nufft_window_size()
            #     kernel = np.ones(window_size) / window_size
            #     pad_amount = (len(kernel) - 1) // 2
            #     padded_y_data = np.pad(y_data, pad_width=pad_amount, mode='reflect')
            #     y_data = np.convolve(padded_y_data, kernel, mode='valid')
            #     y_data = y_data - y_data.min()
            #     # Adjust spatial frequency array if length changed due to smoothing
            #     if len(y_data) != len(spatial_freq_array):
            #         spatial_freq_array = spatial_freq_array[:len(y_data)]
            #         res_range_array = res_range_array[:len(y_data)]
            
            # # Apply detrending if enabled
            # if input.nufft_detrend() and len(y_data) > 2:
            #     m, b = np.polyfit(spatial_freq_array, y_data, 1)
            #     baseline = m * spatial_freq_array + b
            #     y_data = y_data - baseline
            #     y_data = y_data - y_data.min()
            
            if input.nufft_log_y():
                y_data = np.log1p(np.abs(y_data))
                y_title = "Log(Intensity)"
            else:
                y_title = "Intensity"
            
            # Calculate tentative apix for each spatial frequency point
            # Formula: tentative_apix = nominal_apix * (spatial_freq / target_spatial_freq)
            # This shows what the apix would be if the highest peak were at the target resolution
            nominal_apix = float(input.nominal_apix()) if input.nominal_apix() else current_apix
            target_spatial_freq = 1.0 / target_resolution
            tentative_apix_array = nominal_apix * (spatial_freq_array / target_spatial_freq)
            
            # Create the plot with hoverable vertical dash line
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spatial_freq_array,
                y=y_data,
                mode='lines',
                name='Power Curve',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Tentative Apix:</b> %{customdata[1]:.4f} Å/px<br>' +
                             '<b>For resolution:</b> ' + f'{target_resolution:.3f} Å<extra></extra>',
                customdata=np.column_stack([res_range_array, tentative_apix_array])
            ))
            
            # Removed target resolution line - no longer needed
            # Note: Green vertical line for clicks is now added directly in click handler
            
            # Create tick values and labels for spatial frequency x-axis
            n_ticks = min(6, len(spatial_freq_array))  # Maximum 6 ticks
            if n_ticks > 0:
                tick_indices = np.linspace(0, len(spatial_freq_array)-1, n_ticks, dtype=int)
                x_tickvals = spatial_freq_array[tick_indices]
                x_ticktext = []
                for idx in tick_indices:
                    if idx < len(res_range_array):
                        res_val = res_range_array[idx]
                        x_ticktext.append(f"1/{res_val:.2f}")
                    else:
                        x_ticktext.append("")
            else:
                x_tickvals = []
                x_ticktext = []
            
            fig.update_layout(
                title=f'NuFFT Power Curve: ±{display_range:.1f}% around {target_resolution:.2f}Å ({freq_low:.3f}-{freq_high:.3f} 1/Å) ({apix_source})',
                xaxis_title='Spatial Resolution (1/Å)',
                yaxis_title=y_title,
                height=300,
                width=650,  # Fixed width to match heatmap
                margin=dict(l=60, r=20, t=40, b=60),
                autosize=False,
                showlegend=False,
                hovermode='x unified',  # Show unified hover with vertical line
                xaxis=dict(
                    tickvals=x_tickvals,
                    ticktext=x_ticktext,
                    tickmode='array',
                    showspikes=True,  # Enable vertical spike line on hover
                    spikecolor="rgba(0,0,0,0.5)",  # Dark gray with transparency
                    spikesnap="cursor",  # Snap spike to cursor position
                    spikemode="across",  # Show spike across entire plot
                    spikethickness=1,  # Thickness of spike line
                    spikedash="dash"  # Dashed spike line
                ),
                yaxis=dict(
                    showspikes=False  # Disable horizontal spikes
                )
            )
            
            # Create FigureWidget for click handling
            fw = go.FigureWidget(fig)
            
            # Clear any existing green vertical lines from previous power curves
            # This ensures green lines are reset when NuFFT is recalculated
            current_shapes = list(fw.layout.shapes) if fw.layout.shapes else []
            preserved_shapes = [shape for shape in current_shapes 
                              if not (hasattr(shape, 'line') and 
                                     hasattr(shape.line, 'color') and 
                                     shape.line.color == 'green')]
            fw.layout.shapes = preserved_shapes
            
            # Store widget for click event access
            nufft_power_widget.set(fw)
            
            power_plot_end = time.time()
            power_plot_duration = power_plot_end - power_plot_start
            print(f"🏁 POWER CURVE PLOT COMPLETED in {power_plot_duration:.3f} seconds")
            print(f"🎯 TOTAL END-TO-END TAB SWITCH TO PLOTS VISIBLE: Check previous timing logs")
            
            return fw
            
        except Exception as e:
            power_plot_end = time.time()
            power_plot_duration = power_plot_end - power_plot_start
            print(f"❌ POWER CURVE PLOT FAILED after {power_plot_duration:.3f} seconds: {e}")
            import traceback
            traceback.print_exc()
            return go.FigureWidget()

    # Synchronize x-axis ranges between heatmap and 1D plot
    @reactive.Effect
    def sync_plot_ranges():
        """Update plot ranges when shared x-range changes."""
        shared_range = shared_x_range.get()
        if shared_range is None:
            return
            
        # Update 1D plot range
        widget_1d = fft_1d_widget.get()
        if widget_1d is not None and range_update_source.get() != '1d':
            range_update_source.set('sync')
            try:
                widget_1d.layout.xaxis.range = shared_range
            except Exception as e:
                print(f"Error syncing 1D plot range: {e}")
            finally:
                range_update_source.set(None)
                
        # Note: Heatmap will update automatically through its render function
        # which uses shared_x_range.get() in its layout

    @reactive.Effect
    def setup_nufft_power_click_handler():
        """Set up click event handler for NuFFT power curve."""
        widget = nufft_power_widget.get()
        if widget is None:
            return
        
        # Get the current data to enable click info extraction
        calc_state = nufft_calculation_state.get()
        if calc_state['region'] is None:
            return
            
        def on_click(trace, points, selector):
            """Handle click events on NuFFT power curve."""
            import time
            click_start = time.time()
            print(f"🖱️  CLICK EVENT START: {click_start:.3f}")
            if points.point_inds:
                try:
                    # Get clicked point information
                    point_idx = points.point_inds[0]
                    
                    # Get the tentative apix from the hover data (customdata)
                    tentative_apix = None
                    if hasattr(trace, 'customdata') and trace.customdata is not None and point_idx < len(trace.customdata):
                        tentative_apix = trace.customdata[point_idx][1]  # Second column is tentative apix
                    
                    if tentative_apix is not None:
                        # Get the x-value (spatial frequency) for the vertical line
                        x_val = points.xs[0] if points.xs else None
                        
                        # Add vertical line at click position using add_vline method
                        if x_val is not None:
                            
                            # First clear any existing green lines
                            with widget.batch_update():
                                # Clear existing shapes by filtering out green lines
                                current_shapes = list(widget.layout.shapes) if widget.layout.shapes else []
                                
                                # More robust filtering - check for green color in different ways
                                preserved_shapes = []
                                for shape in current_shapes:
                                    is_green_line = False
                                    if hasattr(shape, 'line'):
                                        if hasattr(shape.line, 'color') and shape.line.color == 'green':
                                            is_green_line = True
                                        elif isinstance(shape.line, dict) and shape.line.get('color') == 'green':
                                            is_green_line = True
                                    elif isinstance(shape, dict) and shape.get('line', {}).get('color') == 'green':
                                        is_green_line = True
                                    
                                    if not is_green_line:
                                        preserved_shapes.append(shape)
                                
                                
                                # Set the filtered shapes first
                                widget.layout.shapes = preserved_shapes
                            
                            # Add the green line using add_shape method
                            try:
                                # Get the current y-axis range from the widget for proper line positioning
                                if len(widget.data) > 0 and len(widget.data[0].y) > 0:
                                    y_min = min(widget.data[0].y)
                                    y_max = max(widget.data[0].y)
                                else:
                                    y_min, y_max = 0, 1
                                
                                
                                # Use add_shape method which should force a redraw
                                widget.add_shape(
                                    type="line",
                                    x0=x_val, x1=x_val,
                                    y0=y_min-1, y1=y_max+1,
                                    line=dict(color="green", width=2)
                                )
                                
                            except Exception as e:
                                
                                # Fallback to direct layout manipulation
                                with widget.batch_update():
                                    current_shapes = list(widget.layout.shapes) if widget.layout.shapes else []
                                    green_line_shape = {
                                        'type': 'line',
                                        'x0': x_val, 'x1': x_val,
                                        'y0': y_min-1, 'y1': y_max+1,
                                        'line': {'color': 'green', 'width': 2, 'dash': 'solid'}
                                    }
                                    current_shapes.append(green_line_shape)
                                    widget.layout.shapes = current_shapes
                        
                        # Update the apix slider directly to the tentative apix value
                        ui.update_text("apix_exact_str", value=f"{tentative_apix:.4f}")
                        ui.update_slider("apix_slider", value=tentative_apix)
                        
                        # Enable focused heatmap around clicked frequency
                        if x_val is not None:
                            nufft_clicked_frequency.set(x_val)
                            nufft_show_focused_heatmap.set(True)
                            print(f"🎯 FOCUSED HEATMAP enabled around frequency {x_val:.6f} 1/Å")
                        
                        click_end = time.time()
                        click_duration = click_end - click_start
                        print(f"🎯 CLICK EVENT COMPLETED in {click_duration:.3f} seconds")
                        
                    # If tentative_apix is None, do nothing
                        
                except Exception as e:
                    click_end = time.time()
                    click_duration = click_end - click_start
                    print(f"❌ CLICK EVENT FAILED after {click_duration:.3f} seconds: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Add click handler to the first trace (the power curve line)
        if len(widget.data) > 0:
            widget.data[0].on_click(on_click)
            
        # Add general click handler to clear green line when clicking elsewhere
        def on_general_click(trace, points, selector):
            """Clear green line when clicking outside the power curve."""
            # If no points were clicked or this is a different trace, clear green lines
            if not points.point_inds or trace != widget.data[0]:
                with widget.batch_update():
                    # Remove all green vertical lines
                    current_shapes = list(widget.layout.shapes) if widget.layout.shapes else []
                    preserved_shapes = [shape for shape in current_shapes 
                                      if not (hasattr(shape, 'line') and 
                                             hasattr(shape.line, 'color') and 
                                             shape.line.color == 'green')]
                    widget.layout.shapes = preserved_shapes
                    
        # Add click handler to other traces if they exist (like background)
        for i, trace in enumerate(widget.data):
            if i != 0:  # Skip the first trace (power curve) as it has its own handler
                trace.on_click(on_general_click)

    @reactive.Effect
    def setup_nufft_heatmap_click_handler():
        """Set up click event handler for NuFFT heatmap."""
        widget = nufft_heatmap_widget.get()
        if widget is None:
            return

        # Get the current data to enable click info extraction
        calc_state = nufft_calculation_state.get()
        if calc_state['region'] is None:
            return

        show_focused = nufft_show_focused_heatmap.get()
        if not show_focused:
            return

        def on_heatmap_click(trace, points, selector):
            """Handle click events on NuFFT heatmap."""
            import time
            click_start = time.time()
            print(f"🖱️  HEATMAP CLICK EVENT START: {click_start:.3f}")

            if points.point_inds:
                try:
                    # Get clicked point information (y-coordinate represents frequency)
                    point_idx = points.point_inds[0]
                    y_val = points.ys[0] if points.ys else None

                    if y_val is not None:
                        # Get cached heatmap data to access frequency range
                        cached_heatmap_data = cached_nufft_heatmap_data.get()
                        if cached_heatmap_data is None:
                            print("No cached heatmap data available for click handling")
                            return

                        # Reconstruct frequency range from cached data
                        clicked_freq = nufft_clicked_frequency.get()
                        res_low = cached_heatmap_data['res_low']
                        res_high = cached_heatmap_data['res_high']
                        r_samples = cached_heatmap_data['r_samples']

                        # Find the focused frequency range around clicked frequency
                        res_range_array = np.linspace(res_low, res_high, r_samples)
                        spatial_freq_array = 1.0 / res_range_array
                        freq_index = np.argmin(np.abs(spatial_freq_array - clicked_freq))

                        # Calculate slice bounds (±15 pixels)
                        slice_width = 15
                        start_idx = max(0, freq_index - slice_width)
                        end_idx = min(r_samples, freq_index + slice_width + 1)

                        # Get the focused frequency range
                        focused_res_range = res_range_array[start_idx:end_idx]
                        focused_freq_range = 1.0 / focused_res_range

                        # Extract apix value directly from hover text to ensure exact match
                        try:
                            # Get the widget to access hover text
                            widget = nufft_heatmap_widget.get()
                            if widget and hasattr(widget.data[0], 'text'):
                                # Access the hover text array
                                hover_text_array = widget.data[0].text
                                if hover_text_array is not None and len(hover_text_array) > 0:
                                    # Convert coordinates to hover text indices
                                    x_idx = int(round(points.xs[0])) if points.xs else 0
                                    y_idx = int(round(y_val))

                                    # Safely access the hover text
                                    if (0 <= y_idx < len(hover_text_array) and
                                        0 <= x_idx < len(hover_text_array[y_idx])):
                                        hover_text = hover_text_array[y_idx][x_idx]

                                        # Extract apix value from hover text using regex
                                        import re
                                        apix_match = re.search(r'<b>Tentative Apix:</b>\s*([\d.]+)\s*Å/px', hover_text)
                                        if apix_match:
                                            tentative_apix = float(apix_match.group(1))
                                        else:
                                            raise ValueError("Could not extract apix from hover text")
                                    else:
                                        raise IndexError("Click coordinates out of hover text bounds")
                                else:
                                    raise ValueError("No hover text available")
                            else:
                                raise ValueError("Widget or hover text not accessible")

                        except Exception as e:
                            # Fallback to calculation if hover text extraction fails
                            y_idx = int(round(y_val))
                            if 0 <= y_idx < len(focused_freq_range):
                                clicked_spatial_freq = focused_freq_range[y_idx]
                                nominal_apix = float(input.nominal_apix()) if input.nominal_apix() else cached_heatmap_data['apix']
                                current_resolution_type = input.resolution_type()
                                current_custom_resolution = input.custom_resolution()
                                if current_resolution_type and current_resolution_type != "Custom":
                                    resolution_map = {
                                        "Graphene (2.13 Å)": 2.13,
                                        "Graphene (100)": 2.13,
                                        "Graphene (110)": 1.23,
                                        "Gold (2.355 Å)": 2.355,
                                        "Gold (111)": 2.35,
                                        "Gold (200)": 2.04,
                                        "Gold (220)": 1.44,
                                        "Ice (3.661 Å)": 3.661
                                    }
                                    target_resolution = resolution_map.get(current_resolution_type, 2.13)
                                else:
                                    target_resolution = current_custom_resolution if current_custom_resolution else 2.13
                                target_spatial_freq = 1.0 / target_resolution
                                tentative_apix = nominal_apix * (clicked_spatial_freq / target_spatial_freq)

                        # Update apix slider and text (regardless of extraction method)
                        if 0.01 <= tentative_apix <= 6.0:
                            # Set flag to prevent NuFFT recalculation during UI update
                            apix_updating_from_nufft_click.set(True)

                            ui.update_text("apix_exact_str", value=f"{tentative_apix:.4f}")
                            ui.update_slider("apix_slider", value=tentative_apix)

                            # Clear flag after a short delay
                            import threading
                            def clear_flag():
                                time.sleep(0.1)
                                apix_updating_from_nufft_click.set(False)
                            threading.Thread(target=clear_flag, daemon=True).start()

                            pass
                        else:
                            pass

                    click_end = time.time()
                    click_duration = click_end - click_start
                    print(f"🏁 HEATMAP CLICK EVENT completed in {click_duration:.3f} seconds")

                except Exception as e:
                    print(f"Error handling heatmap click: {e}")
                    import traceback
                    traceback.print_exc()

        # Add click handler to the heatmap trace (first trace)
        if len(widget.data) > 0:
            widget.data[0].on_click(on_heatmap_click)

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
            # Handle both old (4-element) and new (5-element) formats
            if len(green_tilt) >= 5:
                small_axis, large_axis, tilt_angle, untilted_apix, orientation_theta = green_tilt
                orientation_degrees = math.degrees(orientation_theta)
            else:
                small_axis, large_axis, tilt_angle, untilted_apix = green_tilt
                orientation_degrees = None
                
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            apix_str = ""
            if untilted_apix is not None:
                apix_str = f", Apix: {untilted_apix:.3f} Å/px"
            
            orientation_str = ""
            if orientation_degrees is not None:
                orientation_str = f", Orientation: {orientation_degrees:.1f}°"
            
            green_result = (f"🟢 User-clicked: Minor axis: {small_axis:.2f}, "
                           f"Major axis: {large_axis:.2f}, "
                           f"Tilt angle: {tilt_angle_degrees:.2f}°"
                           f"{orientation_str}"
                           f"{apix_str}")
            results.append(green_result)
        
        # Format red ellipse results
        if red_tilt is not None:
            # Handle both old (4-element) and new (5-element) formats
            if len(red_tilt) >= 5:
                small_axis, large_axis, tilt_angle, untilted_apix, orientation_theta = red_tilt
                orientation_degrees = math.degrees(orientation_theta)
            else:
                small_axis, large_axis, tilt_angle, untilted_apix = red_tilt
                orientation_degrees = None
                
            tilt_angle_degrees = math.degrees(tilt_angle)
            
            apix_str = ""
            if untilted_apix is not None:
                apix_str = f", Apix: {untilted_apix:.3f} Å/px"
            
            orientation_str = ""
            if orientation_degrees is not None:
                orientation_str = f", Orientation: {orientation_degrees:.1f}°"
            
            red_result = (f"🔴 Fine-tuned: Minor axis: {small_axis:.2f}, "
                         f"Major axis: {large_axis:.2f}, "
                         f"Tilt angle: {tilt_angle_degrees:.2f}°"
                         f"{orientation_str}"
                         f"{apix_str}")
            results.append(red_result)
        
        # If we have dual results, return them
        if results:
            return "\n".join(results)
        
        # Fallback to legacy single tilt info
        tilt_info = tilt_info_storage.get()
        if tilt_info is not None:
            # Check if we have the new format with orientation (5 elements)
            if len(tilt_info) >= 5:
                small_axis, large_axis, tilt_angle, untilted_apix, orientation_theta = tilt_info
                tilt_angle_degrees = math.degrees(tilt_angle)
                orientation_degrees = math.degrees(orientation_theta)
                
                apix_str = ""
                if untilted_apix is not None:
                    apix_str = f", Estimated untilted apix: {untilted_apix:.3f} Å/px"
                
                return (f"Minor axis: {small_axis:.2f}, "
                       f"Major axis: {large_axis:.2f}, "
                       f"Tilt angle: {tilt_angle_degrees:.2f}°, "
                       f"Orientation: {orientation_degrees:.1f}°"
                       f"{apix_str}")
            # Check if we have the format with untilted apix (4 elements)
            elif len(tilt_info) >= 4:
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
    
    # Store heatmap maximum position for overlay
    heatmap_max_position = reactive.Value({
        'radius': None,
        'angle': None,
        'show_overlay': False
    })
    
    # Store NuFFT data and estimated apix for Find Apix functionality
    nufft_data_cache = reactive.Value({
        'pwr_curve': None,
        'pwr2d_raw': None,
        'resolution': None,
        'res_low': None,
        'res_high': None,
        'r_samples': None,
        'theta_samples': None
    })
    

    @output
    @render_widget
    def apix_centered_by_nominal_plot():
        """Live vertical scatter: Pixel size centered by nominal value, from region_table_data, using FigureWidget for in-place updates."""
        df = region_table_data.get().copy()
        if df is None or df.empty or 'Filename' not in df.columns or 'Pixel Size' not in df.columns or 'Nominal' not in df.columns:
            fw = FigureWidget()
            apix_centered_widget.set(fw)
            return fw
        # Ensure Pixel Size is float
        try:
            df['Pixel Size'] = pd.to_numeric(df['Pixel Size'], errors='coerce')
        except Exception:
            df['Pixel Size'] = None
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
        
        # Drop rows with missing data
        df = df.dropna(subset=['Pixel Size', 'Nominal'])
        if df.empty:
            fw = FigureWidget()
            apix_centered_widget.set(fw)
            return fw
        # Sort Nominal for plotting
        try:
            nominal_order = sorted(df['Nominal'].dropna().unique())
        except Exception:
            nominal_order = list(df['Nominal'].dropna().unique())
        # Apix - Nominal
        df['Apix_centered_by_nominal'] = df.apply(lambda row: row['Pixel Size'] - row['Nominal'], axis=1)
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
            group_mean = group['Pixel Size'].mean()
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
            if df is None or df.empty or 'Filename' not in df.columns or 'Pixel Size' not in df.columns or 'Nominal' not in df.columns:
                with widget.batch_update():
                    # Clear all existing traces
                    while len(widget.data) > 0:
                        widget.data = widget.data[:-1]
                    widget.layout.title = 'Vertical Scatter: Apix Centered by Nominal Value'
                    widget.layout.xaxis.title = 'Nominal Value (Å)'
                    widget.layout.yaxis.title = 'Apix - Nominal (Å/px)'
                return
            try:
                df['Pixel Size'] = pd.to_numeric(df['Pixel Size'], errors='coerce')
            except Exception:
                df['Pixel Size'] = None
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
            
            df = df.dropna(subset=['Pixel Size', 'Nominal'])
            if df.empty:
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
            df['Apix_centered_by_nominal'] = df.apply(lambda row: row['Pixel Size'] - row['Nominal'], axis=1)
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
                group_mean = group['Pixel Size'].mean()
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

    # @reactive.Effect
    # @reactive.event(input.random_generate)
    # def _():
    #     """Generate random regions from the image and analyze them."""
    #     try:
    #         # Check if we have an image loaded
    #         original_data = original_image_data.get()
    #         if original_data is None:
    #             print("No image loaded for random generation")
    #             return
            
    #         filename = image_filename.get() or "Unknown"
    #         num_regions = input.random_count()
            
    #         if num_regions <= 0:
    #             print("Number of regions must be greater than 0")
    #             return
            
    #         # Calculate region size based on percentage of image
    #         img_height, img_width = original_data.shape
    #         region_size_percent = input.region_size_percent()
    #         region_size = int(min(img_width, img_height) * region_size_percent)
            
    #         print(f"=== GENERATING {num_regions} RANDOM {region_size}x{region_size} REGIONS ===")
    #         print(f"Original image shape: {original_data.shape}")
    #         print(f"Region size percentage: {region_size_percent:.1f} ({region_size}x{region_size} pixels)")
    #         print(f"Filename: {filename}")
            
    #         # Check if image is large enough for calculated region size
    #         if img_height < region_size or img_width < region_size:
    #             print(f"Image too small ({img_width}x{img_height}) for {region_size}x{region_size} regions")
    #             return
            
    #         # Calculate valid coordinate ranges (ensure regions stay within boundaries)
    #         # Convert to Python integers to avoid numpy.float64 issues
    #         max_x = int(img_width - region_size)
    #         max_y = int(img_height - region_size)
            
    #         print(f"Valid coordinate ranges: x=[0, {max_x}], y=[0, {max_y}]")
            
    #         # Generate random regions and analyze each one
    #         import random
    #         successful_regions = 0
            
    #         for i in range(num_regions):
    #             try:
    #                 # Generate random top-left coordinates (ensure integers)
    #                 x0 = int(random.randint(0, max_x))
    #                 y0 = int(random.randint(0, max_y))
    #                 x1 = int(x0 + region_size)
    #                 y1 = int(y0 + region_size)
                    
    #                 print(f"\nRegion {i+1}/{num_regions}: x=[{x0}, {x1}], y=[{y0}, {y1}]")
    #                 print(f"Types: x0={type(x0)}, y0={type(y0)}, x1={type(x1)}, y1={type(y1)}")
                    
    #                 # Extract the region from original image with explicit type checking
    #                 try:
    #                     region_data = original_data[y0:y1, x0:x1]
    #                     print(f"Region data shape: {region_data.shape}, dtype: {region_data.dtype}")
                        
    #                     # Ensure data is in correct format for PIL
    #                     region_data_clean = region_data.astype(np.uint8)
    #                     region_img = Image.fromarray(region_data_clean)
                        
    #                     print(f"Extracted region size: {region_img.size}")
                        
    #                 except Exception as e:
    #                     print(f"Error extracting region: {e}")
    #                     print(f"Coordinate types: x0={type(x0)}, y0={type(y0)}")
    #                     raise e
                    
    #                 # Compute 1D FFT radial profile with detrending
    #                 try:
    #                     # Convert current apix to float to avoid numpy type issues
    #                     current_apix = float(get_apix())
    #                     current_resolution_type = str(input.resolution_type())
    #                     current_custom_resolution = float(input.custom_resolution()) if input.custom_resolution() else None
                        
    #                     plot_data = compute_fft_1d_data(
    #                         region=region_img,
    #                         apix=current_apix,
    #                         use_mean_profile=False,  # Use standard radial average
    #                         log_y=False,  # Use linear scale
    #                         smooth=False,  # No smoothing
    #                         window_size=int(3),  # Ensure integer
    #                         detrend=True,  # Enable detrending as requested
    #                         resolution_type=current_resolution_type,
    #                         custom_resolution=current_custom_resolution
    #                     )
                        
    #                 except Exception as e:
    #                     print(f"Error in compute_fft_1d_data: {e}")
    #                     raise e
                    
    #                 if plot_data is None:
    #                     print(f"Failed to compute FFT for region {i+1}")
    #                     continue
                    
    #                 # Find the maximum in the detrended signal
    #                 x_data = plot_data['x_data']
    #                 y_data = plot_data['y_data']
                    
    #                 if len(y_data) == 0:
    #                     print(f"No data points for region {i+1}")
    #                     continue
                    
    #                 max_idx = np.argmax(y_data)
    #                 fft_max_x = x_data[max_idx]
    #                 fft_max_y = y_data[max_idx]
                    
    #                 print(f"Found maximum at x={fft_max_x:.3f}, y={fft_max_y:.3f}")
                    
    #                 # Calculate apix from the maximum position
    #                 resolution, _ = get_resolution_info(input.resolution_type(), input.custom_resolution())
    #                 if resolution is not None and fft_max_x > 0:
    #                     calculated_apix = (fft_max_x * resolution) / region_size
                        
    #                     if 0.01 <= calculated_apix <= 6.0:
    #                         print(f"Calculated apix: {calculated_apix:.3f} Å/px")
                            
    #                         # Create table entry
    #                         region_location = f"x:{x0}–{x1}, y:{y0}–{y1}"
    #                         region_size_str = f"{region_size}×{region_size} px"
                            
    #                         # Get nominal value from textbox
    #                         nominal_value = float(input.nominal_apix())
                            
    #                         new_row = pd.DataFrame({
    #                             'Filename': [filename],
    #                             'Region Size': [region_size_str],
    #                             'Region Location': [region_location],
    #                             'Pixel Size': [f"{calculated_apix:.3f}"],
    #                             'Nominal': [nominal_value]
    #                         })
                            
    #                         # Add to existing table data
    #                         current_data = region_table_data.get()
    #                         updated_data = pd.concat([current_data, new_row], ignore_index=True)
    #                         region_table_data.set(updated_data)
                            
    #                         successful_regions += 1
    #                         print(f"Added region {i+1} to table: {filename}, {region_size_str}, {region_location}, {calculated_apix:.3f}, {nominal_value}")
                            
    #                     else:
    #                         print(f"Calculated apix {calculated_apix:.3f} is outside valid range [0.01, 6.0]")
    #                 else:
    #                     print(f"Could not calculate apix for region {i+1}")
                        
    #             except Exception as e:
    #                 print(f"Error processing region {i+1}: {e}")
    #                 continue
            
    #         print(f"\n=== RANDOM GENERATION COMPLETE ===")
    #         print(f"Successfully analyzed {successful_regions}/{num_regions} regions")
    #         print(f"Total table entries: {len(region_table_data.get())}")
            
    #     except Exception as e:
    #         print(f"Error in random generation: {e}")
    #         import traceback
    #         traceback.print_exc()

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
            'Pixel Size': [],
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
            yield "Filename,Region Size,Region Location,Pixel Size,Nominal,Average Pixel Size\n"
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


    # @reactive.Effect
    # @reactive.event(input.find_max)
    # def _():
    #     pass
    #     """Handle Find Max button click to find and mark the global maximum in the current view."""
        # Get the 1D plot widget
        # widget = fft_1d_widget.get()
        # if widget is None or len(widget.data) == 0:
        #     return
        
        # # Get the current plot data (already processed with smoothing/detrending)
        # plot_data = fft_1d_data()
        # if plot_data is None:
        #     return
        
        # x_data = plot_data['x_data']
        # y_data = plot_data['y_data']
        
        # # Get the current zoom range from the widget
        # x_range = widget.layout.xaxis.range
        
        # # If zoomed, filter data to the visible range
        # if x_range is not None and len(x_range) == 2:
        #     x_min, x_max = x_range
        #     # Find indices within the visible range
        #     mask = (x_data >= x_min) & (x_data <= x_max)
        #     if np.any(mask):
        #         visible_x = x_data[mask]
        #         visible_y = y_data[mask]
        #     else:
        #         # No data in range, use all data
        #         visible_x = x_data
        #         visible_y = y_data
        # else:
        #     # Not zoomed, use all data
        #     visible_x = x_data
        #     visible_y = y_data
        
        # # Find the global maximum in the visible range
        # if len(visible_y) > 0:
        #     # Find integer maximum first
        #     max_idx = np.argmax(visible_y)
        #     max_x_int = visible_x[max_idx]
        #     max_y_int = visible_y[max_idx]
            
        #     print(f"Integer maximum found at x={max_x_int:.3f}, y={max_y_int:.3f}")
            
        #     # Check if Super Resolution mode is enabled
        #     if input.super_resolution():
        #         # Super Resolution mode: Fit Gaussian around the maximum for sub-pixel precision
        #         try:
        #             # Get window size from slider (convert to half-window for ± range)
        #             window_half_size = input.gaussian_window() / 2.0
                    
        #             # Find indices within window around the maximum
        #             mask = np.abs(visible_x - max_x_int) <= window_half_size
                    
        #             if np.sum(mask) >= 5:  # Need at least 5 points for Gaussian fitting
        #                 fit_x = visible_x[mask]
        #                 fit_y = visible_y[mask]
                        
        #                 # Define Gaussian function: y = a * exp(-((x - mu)^2) / (2 * sigma^2)) + c
        #                 def gaussian(x, a, mu, sigma, c):
        #                     return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                        
        #                 # Initial parameter guesses
        #                 a_guess = max_y_int  # amplitude
        #                 mu_guess = max_x_int  # center
        #                 sigma_guess = 1.0     # width
        #                 c_guess = np.min(fit_y)  # offset
                        
        #                 # Fit the Gaussian
        #                 from scipy.optimize import curve_fit
        #                 popt, _ = curve_fit(gaussian, fit_x, fit_y, 
        #                                   p0=[a_guess, mu_guess, sigma_guess, c_guess],
        #                                   maxfev=1000)
                        
        #                 # Extract fitted parameters
        #                 a_fit, mu_fit, sigma_fit, c_fit = popt
                        
        #                 # Use the fitted center as the refined maximum position
        #                 max_x = mu_fit
        #                 max_y = gaussian(mu_fit, *popt)
                        
        #                 print(f"Super Resolution: Gaussian fit successful with {input.gaussian_window()}-pixel window")
        #                 print(f"  - Center: {mu_fit:.5f}, Amplitude: {a_fit:.3f}, Sigma: {sigma_fit:.3f}")
        #                 print(f"  - Sub-pixel maximum: x={max_x:.5f}, y={max_y:.3f} (refined from {max_x_int:.3f})")
                        
        #                 # Store Gaussian parameters for overlay
        #                 gaussian_params = popt
        #                 gaussian_fit_range = (fit_x.min(), fit_x.max())
                        
        #             else:
        #                 # Not enough points for fitting, use integer maximum
        #                 max_x = max_x_int
        #                 max_y = max_y_int
        #                 gaussian_params = None
        #                 gaussian_fit_range = None
        #                 print(f"Super Resolution: Not enough points for Gaussian fitting ({np.sum(mask)} points), using integer maximum")
                        
        #         except Exception as e:
        #             # Gaussian fitting failed, use integer maximum
        #             max_x = max_x_int
        #             max_y = max_y_int
        #             gaussian_params = None
        #             gaussian_fit_range = None
        #             print(f"Super Resolution: Gaussian fitting failed ({e}), using integer maximum")
        #     else:
        #         # Standard mode: Use integer maximum
        #         max_x = max_x_int
        #         max_y = max_y_int
        #         gaussian_params = None
        #         gaussian_fit_range = None
        #         print(f"Standard resolution mode: Using integer maximum")
            
        #     # Calculate apix value corresponding to the maximum position
        #     calculated_apix = None
        #     calc_state = fft_calculation_state.get()
        #     if calc_state['region'] is not None:
        #         resolution, _ = get_resolution_info(calc_state['resolution_type'], calc_state['custom_resolution'])
        #         if resolution is not None and max_x > 0:
        #             # Convert from 1D plot coordinates to FFT image coordinates
        #             region_size = calc_state['region'].size[0]
                    
        #             # Calculate apix using the resolution and distance
        #             # Formula: apix = (distance_in_pixels * resolution) / image_size
        #             calculated_apix = (max_x * resolution) / region_size
                    
        #             if 0.01 <= calculated_apix <= 6.0:
        #                 # Update the apix slider and text input with the calculated value
        #                 ui.update_slider("apix_slider", value=calculated_apix, session=session)
        #                 ui.update_text("apix_exact_str", value=f"{calculated_apix:.3f}", session=session)
        #                 if input.super_resolution():
        #                     print(f"Updated apix to {calculated_apix:.3f} Å/px based on Gaussian-fitted maximum at x={max_x:.5f}")
        #                 else:
        #                     print(f"Updated apix to {calculated_apix:.3f} Å/px based on integer maximum at x={max_x:.3f}")
        #             else:
        #                 print(f"Calculated apix {calculated_apix:.3f} is outside valid range [0.01, 6.0]")
        #                 calculated_apix = None  # Mark as invalid
        #         else:
        #             print("Could not calculate apix - no resolution or invalid max position")
        #     else:
        #         print("Could not calculate apix - no FFT calculation state available")
            
        #     # ALWAYS add vertical line at max position regardless of apix calculation success
        #     print(f"Adding vertical line at max position x={max_x:.5f}, y={max_y:.3f}")
            
        #     try:
        #         with widget.batch_update():
        #             # Remove any existing max markers and Gaussian fits
        #             traces_to_keep = []
        #             removed_count = 0
        #             for trace in widget.data:
        #                 if not (hasattr(trace, 'name') and (trace.name == 'max_marker' or trace.name == 'gaussian_fit')):
        #                     traces_to_keep.append(trace)
        #                 else:
        #                     removed_count += 1
        #             widget.data = traces_to_keep
        #             print(f"Removed {removed_count} existing max markers and Gaussian fits")
                    
        #             # Get current visible range from widget (what user is actually seeing)
        #             current_x_range = widget.layout.xaxis.range
        #             current_y_range = widget.layout.yaxis.range
                    
        #             print(f"Current x-axis range: {current_x_range}")
        #             print(f"Current y-axis range: {current_y_range}")
        #             print(f"Max position to mark: x={max_x:.3f}")
                    
        #             # Determine y-axis range for the vertical line
        #             if current_y_range is not None and len(current_y_range) == 2:
        #                 y_min, y_max_range = current_y_range
        #                 print(f"Using current y-axis range: [{y_min:.1f}, {y_max_range:.1f}]")
        #             else:
        #                 # Fallback: use data range with padding
        #                 y_min = min(0, np.min(y_data) * 0.9)
        #                 y_max_range = np.max(y_data) * 1.1
        #                 print(f"Using calculated y-axis range: [{y_min:.1f}, {y_max_range:.1f}]")
        #                 # Update layout range
        #                 widget.layout.yaxis.range = [y_min, y_max_range]
                    
        #             # Check if max_x is within visible range
        #             if current_x_range is not None and len(current_x_range) == 2:
        #                 x_min_range, x_max_range = current_x_range
        #                 if not (x_min_range <= max_x <= x_max_range):
        #                     print(f"WARNING: max_x={max_x:.3f} is outside visible x-range [{x_min_range:.3f}, {x_max_range:.3f}]")
                    
        #             # Create hover info with apix if available
        #             if input.super_resolution() and gaussian_params is not None:
        #                 # Super resolution mode with successful Gaussian fit
        #                 if calculated_apix is not None:
        #                     hover_info = f'<b>Global Max (Super Resolution)</b><br>x: {max_x:.5f}<br>y: {max_y:.3f}<br>Apix: {calculated_apix:.3f} Å/px<extra></extra>'
        #                 else:
        #                     hover_info = f'<b>Global Max (Super Resolution)</b><br>x: {max_x:.5f}<br>y: {max_y:.3f}<extra></extra>'
        #             else:
        #                 # Standard resolution mode
        #                 if calculated_apix is not None:
        #                     hover_info = f'<b>Global Max (Standard)</b><br>x: {max_x:.3f}<br>y: {max_y:.3f}<br>Apix: {calculated_apix:.3f} Å/px<extra></extra>'
        #                 else:
        #                     hover_info = f'<b>Global Max (Standard)</b><br>x: {max_x:.3f}<br>y: {max_y:.3f}<extra></extra>'
                    
        #             # Add new vertical line at max position with enhanced visibility
        #             line_trace = go.Scatter(
        #                 x=[max_x, max_x],
        #                 y=[y_min, y_max_range],
        #                 mode='lines',
        #                 line=dict(color='red', width=1, dash='solid'),  # Slim vertical line
        #                 name='max_marker',
        #                 showlegend=False,
        #                 hovertemplate=hover_info,
        #                 opacity=1.0  # Ensure full opacity
        #             )
                    
        #             widget.add_trace(line_trace)
                    
        #             # Add Gaussian curve overlay if fitting was successful
        #             if gaussian_params is not None and gaussian_fit_range is not None:
        #                 # Create high-resolution x points for smooth curve
        #                 x_smooth = np.linspace(gaussian_fit_range[0], gaussian_fit_range[1], 100)
                        
        #                 # Calculate Gaussian curve using fitted parameters
        #                 def gaussian(x, a, mu, sigma, c):
        #                     return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                        
        #                 y_smooth = gaussian(x_smooth, *gaussian_params)
                        
        #                 # Create Gaussian curve trace
        #                 gaussian_trace = go.Scatter(
        #                     x=x_smooth,
        #                     y=y_smooth,
        #                     mode='lines',
        #                     line=dict(color='red', width=2, dash='dot'),
        #                     name='gaussian_fit',
        #                     showlegend=False,
        #                     hovertemplate='<b>Gaussian Fit</b><br>x: %{x:.5f}<br>y: %{y:.3f}<extra></extra>',
        #                     opacity=0.8
        #                 )
                        
        #                 widget.add_trace(gaussian_trace)
        #                 print(f"Gaussian curve overlay added: range [{gaussian_fit_range[0]:.3f}, {gaussian_fit_range[1]:.3f}]")
                    
        #             print(f"Vertical line added successfully:")
        #             print(f"  - Position: x={max_x:.3f}")
        #             print(f"  - Y-range: [{y_min:.1f}, {y_max_range:.1f}]")
        #             print(f"  - Line style: red, width=1, solid")
        #             print(f"  - Total traces in widget: {len(widget.data)}")
                    
        #             # Force a refresh of the widget display
        #             widget.layout.uirevision = f"max_marker_{max_x:.3f}_{max_y:.3f}"
                    
        #     except Exception as e:
        #         print(f"Error adding vertical line: {e}")
        #         import traceback
        #         traceback.print_exc()
                
        # # Also analyze the heatmap for maximum intensity
        # try:
        #     # Get heatmap data
        #     calc_state = fft_calculation_state.get()
        #     if calc_state['region'] is not None:
        #         # Check if we should use nominal apix (initial) or current apix (after slider changes)
        #         nominal_apix = float(input.nominal_apix())
        #         current_apix = get_apix()
                
        #         if abs(current_apix - nominal_apix) < 0.01:
        #             range_apix = nominal_apix
        #         else:
        #             range_apix = current_apix
                
        #         heatmap_data = compute_fft_polar_heatmap_data(
        #             region=calc_state['region'],
        #             apix=range_apix,
        #             resolution_type=calc_state['resolution_type'],
        #             custom_resolution=calc_state['custom_resolution']
        #         )
                
        #         z_data = heatmap_data['heatmap_data']
        #         if input.log_y():
        #             z_data = np.log1p(z_data)
                
        #         # Calculate zoom range to limit search area (same as heatmap display)
        #         # Get resolution and calculate target radius
        #         if calc_state['resolution_type'] and calc_state['resolution_type'] != "Custom":
        #             resolution_map = {
        #                 "Graphene (2.13 Å)": 2.13,
        #                 "Graphene (100)": 2.13,
        #                 "Graphene (110)": 1.23,
        #                 "Gold (2.355 Å)": 2.355,
        #                 "Gold (111)": 2.35,
        #                 "Gold (200)": 2.04,
        #                 "Gold (220)": 1.44,
        #                 "Ice (3.661 Å)": 3.661
        #             }
        #             target_resolution = resolution_map.get(calc_state['resolution_type'], 2.13)
        #         else:
        #             target_resolution = calc_state['custom_resolution'] if calc_state['custom_resolution'] else 2.13
                
        #         nominal_apix = float(input.nominal_apix())
        #         region_size = calc_state['region'].size[0]
        #         target_radius = (region_size * nominal_apix) / target_resolution
                
        #         # Define zoom range: ±15 pixels around target radius
        #         zoom_margin = 15
        #         y_min = max(heatmap_data['radii'][0], target_radius - zoom_margin)
        #         y_max = min(heatmap_data['radii'][-1], target_radius + zoom_margin)
                
        #         # Filter radii indices to only search within zoom range
        #         radii_mask = (heatmap_data['radii'] >= y_min) & (heatmap_data['radii'] <= y_max)
        #         valid_radius_indices = np.where(radii_mask)[0]
                
        #         if len(valid_radius_indices) == 0:
        #             print("No radii in zoom range for Find Max search")
        #             return
                
        #         # Create masked data for search (only zoomed region)
        #         # z_data has shape (angles x radii), we want to mask the radii dimension
        #         masked_z_data = z_data[:, radii_mask]
                
        #         # Find maximum intensity in the masked (zoomed) region
        #         max_intensity_idx = np.unravel_index(np.argmax(masked_z_data), masked_z_data.shape)
        #         max_angle_idx, masked_radius_idx = max_intensity_idx
                
        #         # Convert masked radius index back to original radius index
        #         max_radius_idx = valid_radius_indices[masked_radius_idx]
                
        #         max_radius = heatmap_data['radii'][max_radius_idx]
        #         max_angle = heatmap_data['angles'][max_angle_idx]
        #         max_intensity = masked_z_data[max_intensity_idx]
                
        #         print(f"Find Max search limited to zoomed region:")
        #         print(f"  - Zoom range: {y_min:.1f} - {y_max:.1f} px")
        #         print(f"  - Searched {len(valid_radius_indices)} of {len(heatmap_data['radii'])} radii")
                
        #         print(f"Heatmap maximum found:")
        #         print(f"  - Radius: {max_radius:.1f} px")
        #         print(f"  - Angle: {max_angle:.0f}°")
        #         print(f"  - Intensity: {max_intensity:.3f}")
                
        #         # Add red circle overlay to heatmap at max position
        #         try:
        #             # Get the heatmap widget (we need to access it through a stored reference)
        #             # Since we don't have a direct widget reference for the heatmap, 
        #             # we'll need to trigger a heatmap update that includes the overlay
        #             # Store the max position for the heatmap to use
        #             heatmap_max_position.set({
        #                 'radius': max_radius,
        #                 'angle': max_angle,
        #                 'show_overlay': True
        #             })
        #             print(f"Added red circle overlay to heatmap at radius={max_radius:.1f}, angle={max_angle:.0f}°")
                    
        #         except Exception as overlay_error:
        #             print(f"Error adding heatmap overlay: {overlay_error}")
                
        #         # Super resolution analysis for heatmap if enabled
        #         if input.super_resolution():
        #             # Extract the column (all radii) at the max angle, but limit to zoom region
        #             angle_row = z_data[max_angle_idx, radii_mask]  # Use masked data for consistency
        #             radii = heatmap_data['radii'][radii_mask]  # Use radii in zoom region
                    
        #             # Find window around max radius
        #             gaussian_window = input.gaussian_window()
        #             radius_window_half = gaussian_window / 2.0
                    
        #             # Create mask for radii within window
        #             radius_mask = np.abs(radii - max_radius) <= radius_window_half
        #             if np.sum(radius_mask) >= 5:  # Need at least 5 points
        #                 fit_radii = radii[radius_mask]
        #                 fit_intensities = angle_row[radius_mask]
                        
        #                 try:
        #                     from scipy.optimize import curve_fit
                            
        #                     def gaussian(x, a, mu, sigma, c):
        #                         return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c
                            
        #                     # Initial guesses
        #                     a_guess = max_intensity
        #                     mu_guess = max_radius
        #                     sigma_guess = radius_window_half / 3
        #                     c_guess = np.min(fit_intensities)
                            
        #                     # Fit Gaussian
        #                     popt, _ = curve_fit(gaussian, fit_radii, fit_intensities,
        #                                       p0=[a_guess, mu_guess, sigma_guess, c_guess],
        #                                       maxfev=1000)
                            
        #                     fitted_radius = popt[1]
        #                     fitted_intensity = popt[0] + popt[3]  # peak + baseline
                            
        #                     print(f"Heatmap Gaussian fit (super resolution):")
        #                     print(f"  - Fitted radius: {fitted_radius:.2f} px")
        #                     print(f"  - Fitted intensity: {fitted_intensity:.3f}")
                            
        #                 except Exception as fit_error:
        #                     print(f"Heatmap Gaussian fitting failed: {fit_error}")
        #             else:
        #                 print("Heatmap super resolution: Not enough points for Gaussian fitting")
        # except Exception as e:
        #     print(f"Error analyzing heatmap: {e}")
        #     import traceback
        #     traceback.print_exc()

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
                'Pixel Size': [f"{apix_value:.4f}"],
                'Nominal': [nominal_value],
                'Average Pixel Size': ['']  # Will be calculated after adding
            })
            
            # Add to existing table data
            current_data = region_table_data.get()
            updated_data = pd.concat([current_data, new_row], ignore_index=True)
            
            # Calculate statistics for each nominal value and update the Average Pixel Size column
            def update_apix_stats(df):
                df_copy = df.copy()
                # Convert Pixel Size column to float for calculations
                df_copy['Apix_numeric'] = pd.to_numeric(df_copy['Pixel Size'], errors='coerce')

                # Group by nominal value and calculate stats
                for nominal in df_copy['Nominal'].unique():
                    mask = df_copy['Nominal'] == nominal
                    apix_values = df_copy.loc[mask, 'Apix_numeric'].dropna()

                    if len(apix_values) > 0:
                        mean_val = apix_values.mean()
                        std_val = apix_values.std() if len(apix_values) > 1 else 0.0
                        stats_str = f"{mean_val:.4f} ± {std_val:.4f}"
                        df_copy.loc[mask, 'Average Pixel Size'] = stats_str

                return df_copy.drop('Apix_numeric', axis=1)
            
            updated_data = update_apix_stats(updated_data)
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
