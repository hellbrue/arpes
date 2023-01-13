"""Implements data loading for the spectromicroscopy beamline at Elettra."""
import h5py
import numpy as np
import xarray as xr
from arpes.endstations import HemisphericalEndstation, SingleFileEndstation

__all__ = ("LACUSEndstation",)


# ToDo add much more metadata as attributes and coordinates, these should be saved in the file stored from Labview

# To use this script the .tsv files generated have to be converted with the specsanalyzer scripts
# These are then turned into xarray Datasets which are saved as a h5 from which they are input into the PyARPES env

class LACUSEndstation(HemisphericalEndstation, SingleFileEndstation):
    """Implements loading h5 files from the SIS beamline."""

    PRINCIPAL_NAME = "LACUS"
    ALIASES = [
        "LACUS",
        "Harmonium_Beamline",
        "Hyperion_Beamline"
    ]
    _TOLERATED_EXTENSIONS = {
        ".h5"
    }

    RENAME_COORDS = {
        "Angle" : "phi",
        "Ekin" : "eV"
        # "X": "x",
        # "Y": "y",
        # "Z": "z",
        # "Phi": "chi",
        # "Theta": "theta",
    }

    COORDINATES = {
        "Angle",
        "Ekin"
        # "X",
        # "Y",
        # "Z",
        # "Phi",
        # "Theta"
    }

    def print_m(self, *messages):
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.PRINCIPAL_NAME)
        print(s, *messages)

    def resolve_frame_locations(self, scan_desc: dict = None):
        """There is only a single h5 file for data taken at LACUS."""
        return [scan_desc.get("path", scan_desc.get("file"))]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        data_set = xr.open_dataset(frame_path, engine="h5netcdf")
        # Doing conversion to Dataset in lacus specsanalyzer script so below is redundant now
        # data_set = data_array.to_dataset(name="spectrum")

        return data_set

    def postprocess(self, frame: xr.Dataset):
        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """Perform final changes to convention and conversions for data taken at LACUS
        """

        data = data.rename({k: v for k, v in self.RENAME_COORDS.items() if k in data.coords.keys()})

        # Add manipulator coordinates (no access yet)
        data.coords["beta"] = np.nan
        data.coords["alpha"] = np.nan
        data.coords["psi"] = np.nan
        data.coords["x"] = np.nan
        data.coords["y"] = np.nan
        data.coords["z"] = np.nan
        data.coords["chi"] = np.nan
        data.coords["theta"] = np.nan
        data.coords["hv"] = np.nan

        data = super().postprocess_final(data, scan_desc)

        return data