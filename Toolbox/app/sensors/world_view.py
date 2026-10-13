"""Classes to handle the World View constellation images.

`osgeo.gdal` is imported lazily, inside the two methods that actually call
it (`_create_vrt`, `_extract_dn_min`), rather than at module level:
`gui.framework` imports `sensors` unconditionally (for the `SENSORS`
registry used by `sensor_combo_row()`), including for tools such as
`sam3_cropmarks` whose conda env has no GDAL at all and never instantiates
`WV3`/`LG06`.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Union
from importlib.resources import files

from pathlib import Path
from xml.dom import minidom

from .utils import load_radiometric_csv, get_properties
from atmo_correction.utils import BandParams, CtxParams
from atmo_correction.utils import earth_sun_distance_au

import numpy as np
import re

class WorldView:
    """Handle the main sensor from the World View constellation."""

    def _create_vrt(self, folder):
        """Transform TIL files in VRT"""
        from osgeo import gdal

        # Translate the TIL mosaics to VRT in order to update the nodata value
        # This is mandatory to compute the stats
        tile_paths = [i for i in folder.glob("*.TIF")]
        # The name of the VRT file will be the same name as the TIL file
        til_path = [i for i in folder.glob("*.TIL")][0]
        vrt_path = Path(til_path.parent, til_path.name.replace(".TIL", ".vrt"))

        # Create the VRT
        if not vrt_path.exists():
            # Build an in-memory VRT mosaic
            # - resolution='highest' keeps the finest pixel size among inputs
            # - srcNodata/VRTNodata set the nodata value
            build_opts = gdal.BuildVRTOptions(
                resolution="highest",
                srcNodata=0,
                VRTNodata=0,
                separate=False # False = mosaic; True = stack as separate bands
            )
            # gdal.BuildVRT() requires plain str for both the destination
            # path and the source file list (this build's binding has no
            # PathLike support), the same limitation as gdal.Open().
            vrt = gdal.BuildVRT(
                str(vrt_path),
                [str(p) for p in tile_paths],
                options=build_opts,
            )
            # Clean up
            vrt = None

        return vrt_path

    def _find_cloud_cover(self, file_path, pattern="cloudCover"):
        # Load entire file into memory
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Search for the pattern in the text blob
        pos = text.find(pattern)
        if pos == -1:
            return None  # Not found

        # Extract cloudCover numeric value using regex
        match_re = re.search(r"cloudCover\s*=\s*([\d\.]+)", text)
        if match_re:
            cloud_cover = float(match_re.group(1))
            return cloud_cover
        else:
            return None

    def _extract_meta(self, img: Path):
        """
        Each image tile is stored inside a folder where its metadata is stored
        """
        # The image subtiles aren't named as the metadata files. Extract the
        # original image name, i.e., the TIL filename
        til_paths = list(img.parent.glob("*.TIL"))
        if len(til_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .TIL metadata file in {img.parent}, "
                f"found {len(til_paths)}."
            )
        original_img_til = str(til_paths[0])
        imd_path = original_img_til.replace(".TIL", '.IMD')
        # XML metadata file is required in 6S correction.
        xml_path = original_img_til.replace(".TIL", '.XML')

        self.imd = self._parse_imd(Path(imd_path))
        self.xml = minidom.parse(xml_path)

        # The README (and the cloud cover it carries) is optional:
        # _compute_rayleigh_coeffs() already falls back to a default
        # scattering power via `hasattr(self, "cloud_cover")` when it's
        # missing, so don't hard-fail here if no README is found.
        txt_paths = list(img.parent.parent.glob("*README.TXT"))
        if txt_paths:
            self.cloud_cover = self._find_cloud_cover(txt_paths[0])

    def _extract_dn_min(self, img: Path, band_keys):
        """
        Open image and compute the min ND value in start band.
        """
        from osgeo import gdal

        start_bidx = [
            i for i, b in enumerate(band_keys) 
            if b == self.START_BAND
        ][0]

        # gdal.Open() requires a plain str (this build's binding has no
        # PathLike support), and this GDAL build's Dataset has no
        # __enter__/__exit__, so a `with` block isn't usable here.
        ds = gdal.Open(str(img))
        try:
            for bidx in range(1, ds.RasterCount + 1):  # GDAL bands are 1-based

                if bidx != start_bidx + 1:
                    continue

                # Extract GDAL band object to avoid write the whole raster
                band = ds.GetRasterBand(bidx)
                # Avoid counting no data values inside the histogram
                band.SetNoDataValue(band.GetNoDataValue())

                # GDAL-based tool "statsitics" knows that 0 is nodata
                # stats = ds.statistics(bidx, approx=False)  # rasterio>=1.3
                vmin, vmax, *_ = band.GetStatistics(False, True)

                # Extract the 2nd percentile, better approach that min value
                # Get a list of counts
                counts = band.GetHistogram(
                    vmin,
                    vmax,
                    buckets=4096,
                    include_out_of_range=int(False),
                    approx_ok=int(False))

                # Cumulative counts and target rank
                cum_counts = np.cumsum(counts)
                total = cum_counts[-1]

                # target position in sorted data (2nd percentile)
                # bin-level approximation
                if self.START_BAND == "P":
                    # Extract the minimum DN to avoid nodata in dark pixels
                    target = 0.1 / 100.0 * total
                else:
                    target = 2 / 100.0 * total

                # find first bin where cumulative count >= target
                haze_idx = np.searchsorted(cum_counts, target)

                # bin width and lower edge
                bin_width = (vmax - vmin) / len(counts)
                # bin edges are min + i * bin_width
                return vmin + haze_idx * bin_width
        finally:
            ds = None

    def _parse_imd(self, imd_path: Path) -> Dict[str, Any]:
        """
        Parse an IMD (key=value) metadata file that may contain 1-level groups:

            BEGIN_GROUP = GROUP_NAME
            KEY = VALUE
            ...
            END_GROUP = GROUP_NAME

        Returns a dictionary with top-level key/value pairs plus nested dicts
        for each group. Also adds `band_keys` with the suffixes of any 
        top-level keys that start with "BAND_".
        """
        metadata: Dict[str, Any] = {}

        current_group: Optional[Dict[str, str]] = None
        group_name: Optional[str] = None

        with imd_path.open(encoding="utf-8", errors="ignore") as src:
            for lineno, raw in enumerate(src, 1):
                # Trim whitespace and ignore blank/comment-ish lines
                line = raw.strip()
                if not line or line.startswith(("#", "//", "<!--")):
                    continue

                # IMD lines sometimes end with a semicolon; strip it safely.
                if line.endswith(";"):
                    line = line[:-1].rstrip()

                # Skip lines that aren't simple KEY=VALUE statements.
                if "=" not in line:
                    # Also skip XML-ish headers if present (e.g., <?xml ...?>)
                    if line.startswith("<") and line.endswith(">"):
                        continue
                    continue

                # Split on the first '=' only (values may contain '=').
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Handle group boundaries.
                if key == "BEGIN_GROUP":
                    current_group = {}
                    group_name = value
                    continue

                if key == "END_GROUP":
                    # Commit any open group
                    # (be tolerant if END appears without BEGIN).
                    if group_name is not None:
                        metadata[group_name] = current_group or {}
                    current_group = None
                    group_name = None
                    continue

                # Regular key=value: route into current group or top level.
                if current_group is not None:
                    current_group[key] = value
                else:
                    metadata[key] = value

        # If the file ended without a matching END_GROUP,
        # commit the open group.
        if current_group is not None and group_name is not None:
            metadata[group_name] = current_group

        # Extract band suffixes from top-level keys like "BAND_X"
        metadata["band_keys"] = [
            k.split("_", 1)[1]
            for k in metadata.keys()
            if k.startswith("BAND_") and "_" in k
        ]

        return metadata

    def _get_rad_use_prop(self, band_key, prop):
        rad_meta = self.rad_use[band_key]
        return rad_meta[prop]
    
    def _compute_rayleigh_coeffs(self):
        """
        Compute the scaterring coefficients following Chavez 1988
        
        Important: Wavelengths must be in micrometers (e.g. 0.550)
        """
        # Get the effective center wavelengths in micrometers
        wav_c = {}
        cols = ["lowerBandEdge", "upperBandEdge"]
        for b in self.imd["band_keys"]:
            try:
                w_min, w_max = get_properties(self.rad_use, b, *cols)
            except:
                raise ValueError(f"Current band {b} is not in index.")
            wav_c[b] = np.mean([w_min, w_max])
        
        if hasattr(self, "cloud_cover"):
            if self.cloud_cover < 5:
                r_power = 4
            elif self.cloud_cover >= 5 and self.cloud_cover < 10:
                r_power = 2
            elif self.cloud_cover >= 10 and self.cloud_cover < 30:
                r_power = 1
            elif self.cloud_cover >= 30 and self.cloud_cover < 50:
                r_power = 0.7
            elif self.cloud_cover >= 50:
                r_power = 0.5
            
        else:
            r_power = 4
        
        s = {b: (wav_c[self.START_BAND] / wav_c[b])**r_power for b in wav_c}
        return s

    def get_images(self):
        return {'mul': self.mul_path, 'pan': self.pan_path}

    def extract_ctx_params(self, img: Path):
        
        self._extract_meta(img)

        # Retrieve the scene BBOX
        # Its coordinates are in EPSG:4326
        coords = []
        for i in ['ULLON','ULLAT','LRLON','LRLAT']:
            coord = self.xml.getElementsByTagName(i)[0].firstChild.data
            coords.append(float(coord))

        capture_date = self.imd['MAP_PROJECTED_PRODUCT']['earliestAcqTime']
        
        # Complement of the solar elevation angle
        sun_zen_angle = round(90 - float(self.imd['IMAGE_1']['meanSunEl']), 2)
        return CtxParams(
            d=earth_sun_distance_au(capture_date),
            sol_zen_ang=sun_zen_angle,
            mean_sat_az=float(self.imd['IMAGE_1']['meanSatAz']),
            mean_sun_az=float(self.imd['IMAGE_1']['meanSunAz']),
            bbox=coords
        )

class WV3(WorldView):

    # Band names/positions shared by tools that need to enumerate this
    # sensor's bands without opening an image (e.g. PCA combinations,
    # spectral index band mapping).
    BAND_KEYS = ["C", "B", "G", "Y", "R", "RE1", "N", "N2"]
    BAND_POS = [1, 2, 3, 4, 5, 6, 7, 8]
    # Default PCA combinations; ids match the full sweep (`generate_combis`).
    PCA_COMBIS_CSV = files("sensors.data") / "wv3_pca_combinations.csv"

    def __init__(self, image_dir: Path):
        """
        Handle the WV3 images.
        """
        # Choose the starting band for haze computation
        # (for WV3 "Blue" or "Coastal" are recommended)
        self.START_BAND = "C"

        # Store image folders (MUL and PAN)
        target_mul_folders = list(image_dir.glob("*_MUL"))
        target_pan_folders = list(image_dir.glob("*_PAN"))
        
        if (len(target_mul_folders) == 0) or (len(target_pan_folders) == 0):
            raise FileNotFoundError(f"WV3 folder is not valid: {image_dir}")
        
        self.mul_folder = target_mul_folders[0]
        self.pan_folder = target_pan_folders[0]

        # Create VRT from the above folders
        self.mul_path = self._create_vrt(self.mul_folder)
        self.pan_path = self._create_vrt(self.pan_folder)

        # Info about the wavelengths parameters
        # WV3 Spectral Response:
        rad_use_path = files("sensors.data") / "wv3_radiometric_use.csv"
        self.rad_use = load_radiometric_csv(rad_use_path, "band")

    def extract_params_per_band(self, img: Path):
        
        self._extract_meta(img)
        lamb_scatter = self._compute_rayleigh_coeffs()
        # Get the dark object DN value (start band)
        min_dn_value = self._extract_dn_min(img, self.imd["band_keys"])

        band_params = []
        
        for i, band in enumerate(self.imd["band_keys"]):
            # Return image metadata by band
            band_meta = self.imd['BAND_'+band]

            # Construct the dict with the band metadata
            params = BandParams(
                name=band,
                gain=self._get_rad_use_prop(band, 'GAIN_2015v2'),
                offset=self._get_rad_use_prop(band, 'OFFSET_2015v2'),
                e_thuillier=self._get_rad_use_prop(band, 'E_Thuillier'),
                abs_factor=float(band_meta['absCalFactor']),
                effective_band_width=float(band_meta["effectiveBandwidth"]),
                lower_band_edge=self._get_rad_use_prop(band, "lowerBandEdge"),
                upper_band_edge=self._get_rad_use_prop(band, "upperBandEdge"),
                lamb_scatter=lamb_scatter[band],
                dark_object_dn=min_dn_value
            )
            band_params.append(params)
        
        return band_params

    @staticmethod
    def radiometric_calibration(dn: Union[np.ndarray, float], band: BandParams):
        """Bound method to perform the ARC operation before atm correction."""
        if type(dn) == np.ndarray:
            dn = dn.astype("float32", copy=False)
        return band.gain * dn * (band.abs_factor / band.effective_band_width) + band.offset

class LG06(WorldView):

    # Band names/positions shared by tools that need to enumerate this
    # sensor's bands without opening an image (e.g. PCA combinations,
    # spectral index band mapping).
    BAND_KEYS = ["C", "B", "G", "Y", "R", "RE1", "RE2", "N"]
    BAND_POS = [1, 2, 3, 4, 5, 6, 7, 8]
    # Default PCA combinations; ids match the full sweep (`generate_combis`).
    PCA_COMBIS_CSV = files("sensors.data") / "legion6_pca_combinations.csv"

    def __init__(self, image_dir: Path):
        """
        Handle the WV Legion 06 images.
        """
        # Choose the starting band for haze computation
        # (for WV3 "Blue" or "Coastal" are recommended)
        self.START_BAND = "C"

        # Store image folders (MUL and PAN)
        target_mul_folders = list(image_dir.glob("*_MUL"))
        target_pan_folders = list(image_dir.glob("*_PAN"))
        
        if (len(target_mul_folders) == 0) or (len(target_pan_folders) == 0):
            raise FileNotFoundError(f"LEGION folder is not valid: {image_dir}")
        
        self.mul_folder = target_mul_folders[0]
        self.pan_folder = target_pan_folders[0]

        # Create VRT from the above folders
        self.mul_path = self._create_vrt(self.mul_folder)
        self.pan_path = self._create_vrt(self.pan_folder)

        rad_use_path = files("sensors.data") / "legion6_radiometric_use.csv"
        self.rad_use = load_radiometric_csv(rad_use_path, "band")

    def extract_params_per_band(self, img: Path):
        
        self._extract_meta(img)
        lamb_scatter = self._compute_rayleigh_coeffs()
        # Get the dark object DN value (start band)
        min_dn_value = self._extract_dn_min(img, self.imd["band_keys"])

        band_params = []
        
        for i, band in enumerate(self.imd["band_keys"]):
            # Return image metadata by band
            band_meta = self.imd['BAND_'+band]

            # Construct the dict with the band metadata
            params = BandParams(
                name=band,
                gain=self._get_rad_use_prop(band, 'GAIN'),
                offset=self._get_rad_use_prop(band, 'OFFSET'),
                e_thuillier=self._get_rad_use_prop(band, 'ESUN'),
                abs_factor=float(band_meta['absCalFactor']),
                effective_band_width=float(band_meta["effectiveBandwidth"]),
                lower_band_edge=self._get_rad_use_prop(band, "lowerBandEdge"),
                upper_band_edge=self._get_rad_use_prop(band, "upperBandEdge"),
                lamb_scatter=lamb_scatter[band],
                dark_object_dn=min_dn_value
            )
            band_params.append(params)
        
        return band_params

    @staticmethod
    def radiometric_calibration(dn: Union[np.ndarray, float], band: BandParams):
        """Bound method to perform the ARC operation before atm correction."""
        if type(dn) == np.ndarray:
            dn = dn.astype("float32", copy=False)
        return dn * (band.abs_factor / band.effective_band_width)
