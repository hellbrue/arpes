"""Extends lmfit to support curve fitting on xarray instances."""
import operator
import warnings
from lmfit.models import GaussianModel
import xarray as xr
import lmfit as lf
import numpy as np
import scipy

__all__ = ["XModelMixin", "gaussian_convolve"]


def dict_to_parameters(dict_of_parameters) -> lf.Parameters:
    params = lf.Parameters()

    for param_name, param in dict_of_parameters.items():
        params[param_name] = lf.Parameter(param_name, **param)

    return params


class XModelMixin(lf.Model):
    """A mixin providing curve fitting for ``xarray.DataArray`` instances.

    This amounts mostly to making `lmfit` coordinate aware, and providing
    a translation layer between xarray and raw np.ndarray instances.

    Subclassing this mixin as well as an lmfit Model class should bootstrap
    an lmfit Model to one that works transparently on xarray data.

    Alternatively, you can use this as a model base in order to build new models.

    The core method here is `guess_fit` which is a convenient utility that performs both
    a `lmfit.Model.guess`, if available, before populating parameters and
    performing a curve fit.

    __add__ and __mul__ are also implemented, to ensure that the composite model
    remains an instance of a subclass of this mixin.
    """

    n_dims = 1
    dimension_order = None

    def guess_fit(
        self,
        data,
        params=None,
        weights=None,
        guess=True,
        debug=False,
        prefix_params=True,
        transpose=False,
        **kwargs
    ):
        """Performs a fit on xarray data after guessing parameters.

        Params allows you to pass in hints as to what the values and bounds on parameters
        should be. Look at the lmfit docs to get hints about structure
        """
        if params is not None and not isinstance(params, lf.Parameters):
            params = dict_to_parameters(params)

        if transpose:
            assert (
                len(data.dims) == 1
                and "You cannot transpose (invert) a multidimensional array (scalar field)."
            )
        coord_values = {}
        if "x" in kwargs:
            coord_values["x"] = kwargs.pop("x")

        real_data, flat_data = data, data

        new_dim_order = None
        if isinstance(data, xr.DataArray):
            real_data, flat_data = data.values, data.values
            assert len(real_data.shape) == self.n_dims

            if self.n_dims == 1:
                coord_values["x"] = data.coords[list(data.indexes)[0]].values
            else:

                def find_appropriate_dimension(dim_or_dim_list):
                    if isinstance(dim_or_dim_list, str):
                        assert dim_or_dim_list in data.dims
                        return dim_or_dim_list

                    else:
                        intersect = set(dim_or_dim_list).intersection(data.dims)
                        assert len(intersect) == 1
                        return list(intersect)[0]

                # resolve multidimensional parameters
                if self.dimension_order is None or all(d is None for d in self.dimension_order):
                    new_dim_order = data.dims
                else:
                    new_dim_order = [
                        find_appropriate_dimension(dim_options)
                        for dim_options in self.dimension_order
                    ]

                if list(new_dim_order) != list(data.dims):
                    warnings.warn("Transposing data for multidimensional fit.")
                    data = data.transpose(*new_dim_order)

                coord_values = {k: v.values for k, v in data.coords.items() if k in new_dim_order}
                real_data, flat_data = data.values, data.values.ravel()
        real_weights = weights
        if isinstance(weights, xr.DataArray):
            if self.n_dims == 1:
                real_weights = real_weights.values
            else:
                if new_dim_order is not None:
                    real_weights = weights.transpose(*new_dim_order).values.ravel()
                else:
                    real_weights = weights.values.ravel()
        if transpose:
            cached_coordinate = list(coord_values.values())[0]
            coord_values[list(coord_values.keys())[0]] = real_data
            real_data = cached_coordinate
            flat_data = real_data
        if guess:
            guessed_params = self.guess(real_data, **coord_values)
        else:
            guessed_params = self.make_params()

        if params is not None:
            for k, v in params.items():
                if isinstance(v, dict):
                    if prefix_params:
                        guessed_params[self.prefix + k].set(**v)
                    else:
                        guessed_params[k].set(**v)

            guessed_params.update(params)
        result = None
        try:
            result = super().fit(
                flat_data, guessed_params, **coord_values, weights=real_weights, **kwargs
            )
            result.independent = coord_values
            result.independent_order = new_dim_order
        except Exception as e:
            print(e)
            if debug:
                import pdb

                pdb.post_mortem(e.__traceback__)
        finally:
            return result

    def xguess(self, data, **kwargs):
        """Tries to determine a guess for the parameters."""
        x = kwargs.pop("x", None)

        real_data = data
        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert len(real_data.shape) == 1
            x = data.coords[list(data.indexes)[0]].values

        return self.guess(real_data, x=x, **kwargs)

    def __add__(self, other):
        """Implements `+`."""
        comp = XAdditiveCompositeModel(self, other, operator.add)

        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def __mul__(self, other):
        """Implements `*`."""
        comp = XMultiplicativeCompositeModel(self, other, operator.mul)

        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp


class XAdditiveCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models."""

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XMultiplicativeCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models.

    Currently this just copies ``+``, might want to adjust things!
    """

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XConvolutionCompositeModel(lf.CompositeModel, XModelMixin):
    """Work in progress for convolving two ``Model``."""

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        guessed = {}

        for c in self.components:
            if c.prefix == "conv_":
                # don't guess on the convolution term
                continue

            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars
    
    def convolve(arr, kernel, arr_model=None, kernel_model=None, params=None, **kwargs):
        """Simple convolution of two arrays."""
        # Build interpolation from x values, min to max with stepsize equal to min stepsize
        low_lim , high_lim = kwargs['x'][0], kwargs['x'][-1]
        min_step = min(kwargs['x'][i+1] - kwargs['x'][i] for i in range(len(kwargs['x'])-1))
        min_step = min(10, min_step)
        max_val = max(abs(low_lim), abs(high_lim))
        # interp_values = np.arange(low_lim, high_lim+min_step, min_step)
        interp_values = np.arange(-max_val, max_val+min_step, min_step)

        # Build interpolated models, make sure that conv_x contains elements of x, remove double entries and sort
        conv_x = np.unique(np.sort(np.concatenate((interp_values, kwargs['x']))))
        # conv_arr = np.interp(conv_x, kwargs['x'], arr)
        # conv_kernel = np.interp(conv_x, kwargs['x'], kernel)
        interp_arr = arr_model.eval(params, x=conv_x)
        interp_kernel = kernel_model.eval(params, x=conv_x)
        norm_kernel = interp_kernel / np.sum(interp_kernel)

        # Create padding for convolution
        npts = 2*max(arr.size, kernel.size)
        pad = np.ones(npts)
        # tmp = np.concatenate((pad*conv_arr[0], conv_arr, pad*conv_arr[-1]))
        tmp = np.concatenate((pad*interp_arr[0], interp_arr, pad*interp_arr[-1]))

        # Convolve and remove padding, use mode same to keep same length as input
        # out = scipy.signal.fftconvolve(tmp, conv_kernel, mode='same')
        out = scipy.signal.fftconvolve(tmp, norm_kernel, mode='same')
        out = out[npts:-npts]

        # Filter output to only contain values at x (remove interpolated values, not sure if necessary)
        indices = np.where(np.isin(conv_x, kwargs['x']))
        filtered_out = out[indices]
        return filtered_out


def gaussian_convolve(model_instance):
    """Produces a model that consists of convolution with a Gaussian kernel."""
    return XConvolutionCompositeModel(model_instance, GaussianModel(prefix="conv_"), np.convolve)
