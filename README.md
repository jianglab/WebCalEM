# WebCalEM: TEM Magnification Calibration

WebCalEM is a browser-based TEM magnification calibration tool. It loads test specimen images, lets you select image regions, computes FFT/radial profiles in the browser, and exports measurement and statistics results.

## Launch

The client app is contained in `index.html`.

For best results, serve the folder with a local web server:

```bash
python -m http.server 8766
```

Then open:

```text
http://127.0.0.1:8766/
```

You can also open `index.html` directly in a browser, but local file restrictions may limit some loading/export behavior.

## Basic Workflow

1. Load an image from the URL box or switch to Upload and click Browse.
2. Choose the specimen lattice spacing from Specimen, or use Custom.
3. Set the nominal pixel size in Nominal (A/px).
4. Drag a region on the Original Image panel. FFT and radial analysis update after the drag finishes.
5. Use 1D Radial Profile for radial peak picking, or 2D Spectrum for FFT peak/ellipse correction.
6. Review the Pixel Size result and click Add to Table.
7. Use the Measurements and Statistics section to review rows, group summaries, and scatter statistics.
8. Click Download Results to export the measurements CSV, grouped statistics CSV, and statistics plot PNG.

## Navigation Tips

- Hover over controls, images, plots, and table cells for contextual hints and preview snapshots.
- Drag a box on Plotly plots to zoom. Use the modebar, wheel zoom, pan, and reset controls as needed.
- Click a row in the Measurements table to reload its saved image region for manual correction, then click Update Row.
- Use Autosample with Upload mode to sample multiple regions per image. Set Regions and Region size, then click Start.
- The Autocorrect checkbox snaps 1D clicks to nearby local maxima and 2D clicks to the brightest pixel in a small window.
- Resize splitters between panels to allocate more space to the image, FFT analysis, or statistics areas.

## Test Data

The `test_image` folder includes sample TIFF images for quick validation. The default URL in the app points to a graphene test image with a nominal pixel size around 0.75 A/px.

## Notes

- Computation runs client-side in the browser; large regions and large FFT grids can be memory intensive.
- TIFF and MRC parsing are intentionally lightweight and may not cover every microscopy file variant.
- A true client-side FINUFFT/NUFFT implementation is not bundled; the radial calculation uses the current browser FFT/polar sampling approximation.
