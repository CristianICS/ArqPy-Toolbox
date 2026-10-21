# ArqPy tool guides

These guides explain what each graphical tool does, what to enter in every field, what files it creates, and how to interpret the result. They describe the current implementation of the toolbox.

## Recommended workflow

1. [Atmospheric correction](atmospheric-correction.md) — convert raw digital numbers to surface reflectance.
2. [Pansharpening](pansharpening.md) — combine multispectral information with panchromatic spatial detail.
3. [Principal component analysis (PCA)](pca.md) — expose correlated and subtle spectral variation.
4. [Spectral indices](spectral-indices.md) — calculate vegetation, soil, and related band ratios.
5. [High-pass filters](high-pass-filters.md) — emphasize edges and local texture.
6. [MAE saliency](mae-saliency.md) — rank visually unusual regions using a pretrained image model.
7. [Deep-learning pansharpening](deep-learning-pansharpening.md) — run the bundled Z-PNN methods.

Optional external tool:

- [SAM 3 crop-mark detection](sam3-crop-marks.md) — segment candidate features from a text prompt.

The order is a recommendation, not a requirement. Keep the original data and place outputs from each stage in separate folders. A visible anomaly is not automatically archaeological: compare it with field boundaries, drainage, geology, recent land use, image seams, and evidence from other dates or methods.

## Controls shared by several tools

| Control | Behaviour |
| --- | --- |
| **Run** | Starts processing. It is disabled while a job is running. Progress and errors appear in the text panel. |
| **Exit** | Closes the window. Do not use it to interrupt a job unless necessary. |
| **Output folder** | Where offered, leaving it blank uses the automatic location described in that tool's guide. A manually selected folder usually must already exist. |
| **Vector layer to perform a clip operation** | Optional cutline. The output is cropped to the vector geometry. Use a valid `.shp`, `.gpkg`, or `.geojson` and ensure its CRS is defined. KML is currently unsupported because the packaged GDAL environment does not include the `LIBKML` driver. |
| **Output compression** | `Lossless` (default) or `Near-lossless`. See [Output compression](#output-compression). |

## Output compression

Every tool with a floating-point output has an **Output compression** option:

- **Lossless** (default): DEFLATE compression. The stored values are exactly the computed ones.
- **Near-lossless**: LERC compression. Each pixel may differ from the computed value by at most a fixed tolerance, and the file is roughly 2–3× smaller. The data type stays Float32, and NoData pixels are kept exactly.

The tolerance depends on the product:

| Product | Maximum error per pixel |
| --- | --- |
| Reflectance (atmospheric correction, pansharpening) | 5e-5 |
| Spectral indices | 5e-4 |
| PCA components | 1e-3 |
| SAM 3 confidence | 1e-3 |
| MAE saliency, high-pass filters | 0.1 % of the output's value range (99th minus 1st percentile) |

Integer outputs are always lossless, whatever the option says: reflectance scaled to integers (atmospheric correction with **Scale** above 0, and pansharpening of two integer inputs), the SAM 3 mask, and deep-learning pansharpening. Overviews of near-lossless Cloud Optimized GeoTIFFs use a tolerance ten times larger, because they are only used for display. Overviews are built with average resampling.

### Chaining tools: keep intermediate outputs lossless

The tolerance only holds for one tool: a near-lossless output is within its tolerance of what the same tool would write losslessly. When that output becomes the input of another tool, the error enters the next calculation, where it can grow:

- **Pansharpening**: weighted Brovey divides by a weighted sum of the multispectral bands, so an input error is multiplied by roughly `PAN / Σ wⱼ·MSⱼ`. This factor is large in dark areas.
- **Spectral indices**: ratio formulas such as `(N−R)/(N+R)` divide by sums of reflectance. In dark pixels, an input error of 5e-5 can become an index error well above 5e-4.
- **PCA**: each band is divided by its standard deviation, so the input error is multiplied by `1/σ`.

Each tool then adds its own tolerance on top. The end product of a chain where every step is near-lossless therefore has no guaranteed maximum error compared with a fully lossless chain.

**Recommendation:** keep every output that another tool will read lossless, and use near-lossless only for the final product you keep or view.

### Scaled integer reflectance

Atmospheric correction with **Scale** `10000` writes 16-bit integers. These files are lossless and about as small as the near-lossless Float32 ones, but they are only suitable for some tools:

- Use them for Bayesian and deep-learning pansharpening.
- Do not use them for spectral indices: several formulas contain constants that assume reflectance in the 0–1 range (for example SAVI and OSAVI), so the results are wrong.
- Scaled outputs only store reflectance of 0 or more.

## Sensor band names

Band numbers are one-based: the first raster band is band 1.

| Position | WV3 | LEGION-06 |
| ---: | --- | --- |
| 1 | Coastal (C) | Coastal (C) |
| 2 | Blue (B) | Blue (B) |
| 3 | Green (G) | Green (G) |
| 4 | Yellow (Y) | Yellow (Y) |
| 5 | Red (R) | Red (R) |
| 6 | Red edge (RE1) | Red edge 1 (RE1) |
| 7 | Near infrared 1 (N) | Red edge 2 (RE2) |
| 8 | Near infrared 2 (N2) | Near infrared (N) |

Selecting the wrong sensor does not relabel the source raster; it makes the software interpret its band positions incorrectly.
