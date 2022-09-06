"""Implements data loading for the spectromicroscopy beamline at Elettra."""
import os
import h5py
import numpy as np
import xarray as xr
from pathlib import Path
import typing

import arpes.config

from typing import Tuple

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, EndstationBase
from arpes.utilities import unwrap_xarray_item

__all__ = ("SLSUltraEndstation",)

class SLSUltraEndstation(HemisphericalEndstation, SynchrotronEndstation):
    """Implements loading h5 files from the SIS beamline."""

    PRINCIPAL_NAME = "SLS_ULTRA"
    ALIASES = [
        "ULTRA",
        "SIS_SLS"
    ]
    _TOLERATED_EXTENSIONS = {
        ".h5",".zip"
    }


    RENAME_KEYS = {
        "X": "x",
        "Y": "y",
        "Z": "z",
        "phi": "chi",
        "Theta": "theta",
        "Tilt": "beta"
    }

    COORDINATES = {
        # "X",
        # "Y",
        # "Z",
        "phi",
        # "Theta",
        # "Tilt",
        # "hv"
    }

    def print_m(self, *messages) :
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.PRINCIPLE_NAME)
        print(s, *messages)

    def resolve_frame_locations(self, scan_desc: dict = None):
        """There is only a single h5 file for SLS data without Deflector mode, so this is simple."""
        return [scan_desc.get("path", scan_desc.get("file"))]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):

        f = h5py.File(frame_path, 'r')

        dataset_contents = dict()

        # Extract the actual dataset and some metadata
        h5_data = f['Electron Analyzer/Image Data']
        h5_info = f['Other Instruments']
        # ToDo initialize atrributes
        attributes = h5_data.attrs.items()

        # ToDo get all coordinates from h5 file
        coordinates = {}
        for c in h5_info.keys():
            coord_data = h5_info.get(c)
            if c in self.COORDINATES:
                v = coord_data[()]
                coord = {c: v}
                coordinates.update(coord)

        # Convert to array and make 3 dimensional if necessary
        shape = h5_data.shape
        # Access data chunk-wise, which is much faster.
        # This improvement has been contributed by Wojtek Pudelko and makes data
        # loading from SIS Ultra orders of magnitude faster!
        if len(shape) == 3:
            data = np.zeros(shape)
            for i in range(shape[2]):
                data[:, :, i] = h5_data[:, :, i]
        else:
            data = np.array(h5_data)

        # Build axis
        e_attr = h5_data.attrs['Axis0.Scale']
        e_lims = [e_attr[0], e_attr[0] + e_attr[1] * shape[0]]
        e_axis = np.linspace(*e_lims, shape[0])
        y_attr = h5_data.attrs['Axis1.Scale']
        y_lims = [y_attr[0], y_attr[0] + y_attr[1] * shape[1]]
        y_axis = np.linspace(*y_lims, shape[1])
        # Add Tilt axis for FS scans
        if len(shape) == 3:
            tilt_attr = h5_data.attrs['Axis2.Scale']
            tilt_lims = [tilt_attr[0], tilt_attr[0] + tilt_attr[1] * shape[2]]
            tilt_axis = np.linspace(*tilt_lims, shape[2])

        # Build xrArray of EDC or 2D data
        if len(shape) == 2:
            coords = {"eV": e_axis, "phi": y_axis}
            dims = ["eV", "phi"]
        # Create xrArray of maps etc or 3D scans in general
        else:
            coords = {"energies": e_axis, "angles": y_axis, "tilt": tilt_axis}
            dims = ["energies", "angles", "tilt"]

        coordinates.update(coords)
        # print(coordinates)

        return xr.DataArray(data, coords=coordinates, dims=dims, attrs=attributes)

    def postprocess(self, frame: xr.Dataset):
        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        return  data