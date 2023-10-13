"""Implements data loading for the spectromicroscopy beamline at Elettra."""
import h5py
import numpy as np
import xarray as xr

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation

__all__ = ("SLSUltraEndstation",)


class SLSUltraEndstation(HemisphericalEndstation, SynchrotronEndstation):
    """Implements loading h5 files from the SIS beamline."""

    PRINCIPAL_NAME = "SLS_ULTRA"
    ALIASES = [
        "ULTRA",
        "SIS_SLS"
    ]
    _TOLERATED_EXTENSIONS = {
        ".h5", ".zip"
    }

    RENAME_COORDS = {
        "X": "x",
        "Y": "y",
        "Z": "z",
        "Phi": "chi",
        "Theta": "theta"
    }

    COORDINATES = {
        "X",
        "Y",
        "Z",
        "Phi",
        "Theta",
        "hv"
    }

    def print_m(self, *messages):
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.PRINCIPAL_NAME)
        print(s, *messages)

    def resolve_frame_locations(self, scan_desc: dict = None):
        """There is only a single h5 file for SLS data without Deflector mode, so this is simple."""
        return [scan_desc.get("path", scan_desc.get("file"))]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """ Load data from a single h5 file. This is a modified dataloader script from the arpys
        package to be compatible with pyArpes library. """

        f = h5py.File(frame_path, 'r')

        # Extract the actual dataset and some metadata
        h5_data = f['Electron Analyzer/Image Data']
        h5_info = f['Other Instruments']

        # Get all attributes from h5 file
        attributes = {}
        # Add Image data attributes
        attributes.update(h5_data.attrs.items())
        # Add attributes from metadata in 'Other Instruments'
        for a in h5_info.keys():
            attr_data = h5_info.get(a)
            # Exclude fields that are are coordinates, but still add the attached attributes later
            if a not in self.COORDINATES:
                v = attr_data[0]
                attr = {a: v}
                attributes.update(attr)
            # Iterate and add Groups in 'Other Instruments'
            h5_sub_info = f['Other Instruments/' + a]
            attr = list(h5_sub_info.attrs.items())
            # Problem of same naming convention, rewriting the name and make it consistent with corr. group
            for attr_item in attr:
                attr_list = list(attr_item)
                attr_list[0] = a + ' ' + attr_list[0]
                mod_attr = {attr_list[0]: attr_list[1]}
                attributes.update(mod_attr)

        # Initialize coordinate dictionary
        coordinates = {}

        # Chunkwise dataloader from arpys package
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

        # Build xArray of EDC or 2D data
        if len(shape) == 2:
            axis_coords = {"eV": e_axis, "phi": y_axis}
            dims = ["eV", "phi"]
        # Create xArray of maps etc or 3D scans in general
        elif len(shape) == 3:
            # Add Tilt or theta in pyarpes conv axis for FS (3D data) scans
            beta_attr = h5_data.attrs['Axis2.Scale']
            beta_lims = [beta_attr[0], beta_attr[0] + beta_attr[1] * shape[2]]
            beta_axis = np.linspace(*beta_lims, shape[2])
            axis_coords = {"eV": e_axis, "phi": y_axis, "beta": beta_axis}
            dims = ["eV", "phi", "beta"]
        else:
            axis_coords = {}
            dims = []

        coordinates.update(axis_coords)

        # Get above defined coordinates from Dataset
        for c in h5_info.keys():
            coord_data = h5_info.get(c)
            if c in self.COORDINATES:
                v = coord_data[0]
                coord = {c: v}
                coordinates.update(coord)

        return xr.Dataset(
            {"spectrum": xr.DataArray(data, coords=coordinates, dims=dims, attrs=attributes)}, attrs=attributes
        )

    def postprocess(self, frame: xr.Dataset):
        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """Performs final postprocessing of the data.

        This mostly amounts to:
        1. Adjusting for the work function and converting kinetic to binding energy if necessary
        2. Adjusting angular coordinates to standard conventions
        """

        # Rename the coordinates as defined above in class
        data = data.rename({k: v for k, v in self.RENAME_COORDS.items() if k in data.coords.keys()})
        
        # Add manipulator coordinates (not changeable at SLS)
        data.coords["alpha"] = 0.0
        data.coords["psi"] = 0.0
        if "beta" not in data.coords:
            data.coords["beta"] = data.coords["theta"]
        data.attrs['sample_workfunction'] = data.attrs['Work Function (eV)']
        data.spectrum.attrs['sample_workfunction'] = data.attrs['Work Function (eV)']

        # Check if scan was done in E_kin or E_bin reference and transform to E_bin if necessary
        if "eV" in data.coords:
            workfunction = data.attrs['sample_workfunction']
            if data.attrs['Energy Scale'] == "Kinetic":
                photon_energy = data.coords["hv"]
                data.coords["eV"] = data.eV + workfunction - photon_energy

        for deg_to_rad_coord in {
            "theta",
            "phi",
            "beta",
            "psi"
        }:
            for l in [data]:
                l.coords[deg_to_rad_coord] = l.coords[deg_to_rad_coord] * np.pi / 180
                if deg_to_rad_coord in l.attrs:
                    l.attrs[deg_to_rad_coord] = l.attrs[deg_to_rad_coord] * np.pi / 180

        if "theta" in data.coords:
            data.spectrum.attrs["theta_offset"] = data.coords["theta"].values.ravel()[0]

        data = super().postprocess_final(data, scan_desc)
        return data
