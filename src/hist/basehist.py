from __future__ import annotations

__lazy_modules__ = {
    "fnmatch",
    "functools",
    "histoprint",
    "itertools",
    "numpy",
    "operator",
    "warnings",
    f"{__spec__.parent}",
    f"{__spec__.parent}.axestuple",
    f"{__spec__.parent}.storage",
    f"{__spec__.parent}.svgplots",
}

import fnmatch
import functools
import importlib.metadata
import itertools
import operator
import typing
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, Generic, Protocol, SupportsIndex, TypeVar, Union

import boost_histogram as bh
import histoprint
import numpy as np
import packaging.version

import hist

from . import interop
from .axestuple import NamedAxesTuple
from .axis import AxisProtocol
from .quick_construct import MetaConstructor
from .storage import Storage
from .svgplots import html_hist, svg_hist_1d, svg_hist_1d_c, svg_hist_2d

if typing.TYPE_CHECKING:
    from builtins import ellipsis

    import matplotlib.axes
    from mplhep.plot import Hist1DArtists, Hist2DArtists

    from ._compat.typing import ArrayLike, Self
    from .plot import FitResultArtists, MainAxisArtists, RatiolikeArtists


class SupportsLessThan(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


InnerIndexing = Union[
    SupportsIndex, str, Callable[[bh.axis.Axis], int], slice, "ellipsis"
]
IndexingWithMapping = InnerIndexing | Mapping[int | str, InnerIndexing]
IndexingExpr = IndexingWithMapping | tuple[IndexingWithMapping, ...]
AxisTypes = AxisProtocol | tuple[int, float, float]


# Workaround for bug in mplhep
def _proc_kw_for_lw(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{k[:-3]}_linestyle"
        if k.endswith("_ls")
        else "linestyle"
        if k == "ls"
        else k: v
        for k, v in kwargs.items()
    }


def process_mistaken_quick_construct(
    axes: Sequence[AxisTypes | hist.quick_construct.ConstructProxy[Any]],
) -> Generator[AxisTypes, None, None]:
    for ax in axes:
        if isinstance(ax, hist.quick_construct.ConstructProxy):
            yield from ax.axes
        else:
            yield ax


NO_METADATA = object()

# Relative tolerance for matching requested rebin edges onto existing axis
# edges. Only edges that coincide with an original edge within this tolerance
# are accepted, so a lossless merge never silently splits a bin.
_EDGE_REL_TOLERANCE = 1e-12
# Absolute floor for axes whose edges lie near zero, where a relative
# tolerance becomes meaningless.
_EDGE_ABS_TOLERANCE = 1e-12


def _edge_tolerance(old_edges: np.typing.NDArray[Any]) -> float:
    return float(
        _EDGE_ABS_TOLERANCE
        + _EDGE_REL_TOLERANCE * max(abs(float(old_edges[0])), abs(float(old_edges[-1])))
    )


def _match_rebin_edges(
    edges: Sequence[float], old_edges: np.typing.NDArray[Any]
) -> np.typing.NDArray[np.intp]:
    """
    Map requested coarse bin edges onto indices of the original axis edges.

    Every requested edge must coincide with an original edge (within a
    scale-aware tolerance); otherwise the request would cut through an
    original bin, which cannot be done losslessly from binned data alone.
    The first and last requested edges must equal the axis outer edges so
    the rebin covers exactly the same in-range region.
    """
    requested = np.asarray(edges, dtype=float)
    old = np.asarray(old_edges, dtype=float)
    if requested.ndim != 1:
        msg = "Rebin edges must be a one-dimensional sequence"
        raise ValueError(msg)
    if requested.size < 2:
        msg = "At least two edges are required to define a binning"
        raise ValueError(msg)
    if np.any(np.diff(requested) <= 0):
        msg = "Rebin edges must be strictly increasing"
        raise ValueError(msg)
    tolerance = _edge_tolerance(old)
    if not np.isclose(requested[0], old[0], rtol=0.0, atol=tolerance):
        msg = (
            f"First rebin edge {requested[0]} does not match the axis lower "
            f"edge {old[0]}; a lossless rebin must cover the whole axis"
        )
        raise ValueError(msg)
    if not np.isclose(requested[-1], old[-1], rtol=0.0, atol=tolerance):
        msg = (
            f"Last rebin edge {requested[-1]} does not match the axis upper "
            f"edge {old[-1]}; a lossless rebin must cover the whole axis"
        )
        raise ValueError(msg)

    insertion = np.searchsorted(old, requested)
    # A requested edge may land on either side of its matching edge within
    # floating-point tolerance, so check the insertion point and the edge just
    # before it instead of trusting the insertion point alone.
    right_index = np.clip(insertion, 0, len(old) - 1)
    left_index = np.clip(insertion - 1, 0, len(old) - 1)
    right_distance = np.abs(old[right_index] - requested)
    left_distance = np.abs(old[left_index] - requested)
    matched_left = left_distance <= tolerance
    matched_right = right_distance <= tolerance
    if not np.all(matched_left | matched_right):
        first_cut = int(np.argmax(~(matched_left | matched_right)))
        cut_bin = int(np.clip(insertion[first_cut] - 1, 0, len(old) - 2))
        msg = (
            f"Requested edge {float(requested[first_cut])} cuts through "
            f"original bin {cut_bin} with edges "
            f"[{old[cut_bin]}, {old[cut_bin + 1]}]. "
            "That bin cannot be split from binned data alone; rebuild the "
            "histogram from the original data with the new binning"
        )
        raise ValueError(msg)
    indices = np.where(matched_left & ~matched_right, left_index, right_index)
    if np.any(np.diff(indices) == 0):
        msg = "Rebin edges must each fall on a distinct original edge"
        raise ValueError(msg)
    return np.asarray(indices, dtype=np.intp)


S = TypeVar("S", bound=bh.storage.Storage)

IntHists = TypeVar(
    "IntHists", bound="BaseHist[bh.storage.AtomicInt64] | BaseHist[bh.storage.Int64]"
)
FloatHists = TypeVar(
    "FloatHists", bound="BaseHist[bh.storage.Double] | BaseHist[bh.storage.Unlimited]"
)
ListHists = TypeVar("ListHists", bound="BaseHist[bh.storage.MultiCell]")
WeightHists = TypeVar("WeightHists", bound="BaseHist[bh.storage.Weight]")
MeanHists = TypeVar("MeanHists", bound="BaseHist[bh.storage.Mean]")
WeightedMeanHists = TypeVar(
    "WeightedMeanHists", bound="BaseHist[bh.storage.WeightedMean]"
)

bh_version = packaging.version.Version(importlib.metadata.version("boost-histogram"))
if typing.TYPE_CHECKING or bh_version >= packaging.version.Version("1.7.1"):
    _Histogram = bh.Histogram
else:

    class _Histogram:
        def __class_getitem__(cls, arg):
            return bh.Histogram


class BaseHist(_Histogram[S], Generic[S], metaclass=MetaConstructor, family=hist):
    __slots__ = ()

    @typing.overload
    def __init__(
        self,
        arg: dict[str, Any],
        /,
        *,
        data: np.typing.NDArray[Any] | None = ...,
        metadata: Any = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        arg: Self | bh.Histogram[S],
        /,
        *,
        data: np.typing.NDArray[Any] | None = ...,
        metadata: Any = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        *axes: AxisTypes | Storage | str,
        storage: S = ...,
        metadata: Any = ...,
        data: np.typing.NDArray[Any] | None = ...,
        label: str | None = ...,
        name: str | None = ...,
    ) -> None: ...

    def __init__(
        self,
        *in_args: Self | bh.Histogram[S] | dict[str, Any] | AxisTypes | Storage | str,
        storage: Storage | str | None = None,
        metadata: Any = NO_METADATA,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize BaseHist object. Axis params can contain the names.
        """
        # Making the histogram generic seems to confuse PyLint's slots check
        self._hist: Any = None  # pylint: disable=assigning-non-slot
        self.axes: NamedAxesTuple  # pylint: disable=assigning-non-slot

        args: tuple[AxisTypes, ...]

        if in_args and storage is None and isinstance(in_args[-1], (Storage, str)):
            storage = in_args[-1]
            args = in_args[:-1]  # type: ignore[assignment]
        else:
            args = in_args  # type: ignore[assignment]

        # Support raw Quick Construct being accidentally passed in
        args = tuple(ax for ax in process_mistaken_quick_construct(args))

        # Remove when boost-histogram 1.8.2+ is required; it does this itself
        if (
            len(args) == 1
            and not isinstance(args[0], bh.Histogram)
            and hasattr(args[0], "_to_uhi_")
        ):
            args = (args[0]._to_uhi_(),)  # type: ignore[union-attr]

        if isinstance(storage, str):
            storage_str = storage.title()
            if storage_str == "Atomicint64":
                storage_str = "AtomicInt64"
            elif storage_str == "Weightedmean":
                storage_str = "WeightedMean"
            storage = getattr(bh.storage, storage_str)()
        elif isinstance(storage, type):
            msg = f"Please use '{storage.__name__}()' instead of '{storage.__name__}'"  # type: ignore[unreachable]
            warnings.warn(msg, stacklevel=2)
            storage = storage()

        if metadata is NO_METADATA:
            super().__init__(*args, storage=storage)  # type: ignore[call-overload]
        else:
            super().__init__(*args, storage=storage, metadata=metadata)  # type: ignore[call-overload]

        disallowed_names = {"weight", "sample", "threads"}
        for ax in self.axes:
            if ax.name in disallowed_names:
                disallowed_warning = (
                    f"{ax.name} is a protected keyword and cannot be used as axis name"
                )
                warnings.warn(disallowed_warning, stacklevel=2)

        valid_names = [ax.name for ax in self.axes if ax.name]
        if len(valid_names) != len(set(valid_names)):
            msg_0 = f"{self.__class__.__name__} instance cannot contain axes with duplicated names"
            raise KeyError(msg_0)
        for i, ax in enumerate(self.axes):
            # label will return name if label is not set, so this is safe
            if not ax.label:
                ax.label = f"Axis {i}"

        if data is not None:
            self[...] = data

        # Keep name/label copied from an input histogram unless overridden
        if name is not None or "name" not in self.__dict__:
            self.name = name  # pylint: disable=assigning-non-slot
        if label is not None or "label" not in self.__dict__:
            self.label = label  # pylint: disable=assigning-non-slot

    def _generate_axes_(self) -> NamedAxesTuple:
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return NamedAxesTuple(self._axis(i) for i in range(self.ndim))

    def _repr_html_(self) -> str | None:
        if self.size == 0:
            return None

        if self.ndim == 1 and len(self.axes[0]) <= 1000:
            return str(
                html_hist(
                    self,
                    svg_hist_1d_c if self.axes[0].traits.circular else svg_hist_1d,
                )
            )

        if self.ndim == 2 and len(self.axes[0]) <= 200 and len(self.axes[1]) <= 200:
            return str(html_hist(self, svg_hist_2d))

        return None

    def __repr__(self) -> str:
        # Workaround for boost-histogram 1.7.0 and 1.7.1 having a 0
        old_repr = super().__repr__()
        if "MultiCell(0)" in old_repr:
            nelem = self._hist.nelem()
            return old_repr.replace("MultiCell(0)", f"MultiCell({nelem})")
        return old_repr

    def _name_to_index(self, name: str) -> int:
        """
            Transform axis name to axis index, given axis name, return axis \
            index.
        """
        for index, axis in enumerate(self.axes):
            if name == axis.name:
                return index

        msg = f"The axis name {name} could not be found"
        raise ValueError(msg)

    def _to_uhi_(self) -> dict[str, Any]:
        from .serialization import to_uhi

        return to_uhi(self)

    @classmethod
    def from_columns(
        cls,
        data: Mapping[str, ArrayLike],
        axes: Sequence[str | AxisProtocol],
        *,
        weight: str | None = None,
        storage: hist.storage.Storage | None = None,
    ) -> Self:
        if storage is None:
            storage = hist.storage.Double()
        axes_list: list[Any] = []
        for ax in axes:
            if isinstance(ax, str):
                assert ax in data, f"{ax} must be present in data={list(data)}"
                cats = set(data[ax])  # type: ignore[arg-type]
                if all(isinstance(a, str) for a in cats):
                    axes_list.append(hist.axis.StrCategory(sorted(cats), name=ax))  # type: ignore[arg-type]
                elif all(isinstance(a, (int, np.integer)) for a in cats):
                    axes_list.append(hist.axis.IntCategory(sorted(cats), name=ax))  # type: ignore[arg-type]
                else:
                    msg = f"{ax} must be all int or strings if axis not given"
                    raise TypeError(msg)
            elif not ax.name or ax.name not in data:
                msg = "All axes must have names present in the data"
                raise TypeError(msg)
            else:
                axes_list.append(ax)

        weight_arr = data[weight] if weight else None

        self = cls(*axes_list, storage=storage)
        data_list = {x.name: data[x.name] for x in axes_list}
        self.fill(**data_list, weight=weight_arr)  # type: ignore[arg-type]
        return self

    def project(self, *args: int | str, flow: bool = True) -> Self:
        """
        Projection of axis idx.
        """
        int_args = [self._name_to_index(a) if isinstance(a, str) else a for a in args]
        if flow:
            return super().project(*int_args)
        # flow=False requires boost-histogram 1.8+
        return super().project(*int_args, flow=False)

    @property
    def T(self) -> Self:
        # Skip subclass overrides, since NamedHist.project rejects indices
        return BaseHist.project(self, *reversed(range(self.ndim)))

    def fill(
        self,
        *args: ArrayLike,
        weight: ArrayLike | None = None,
        sample: ArrayLike | None = None,
        threads: int | None = None,
        **kwargs: ArrayLike,
    ) -> Self:
        """
        Insert data into the histogram using names and indices, return
        a Hist object.
        """

        if (
            issubclass(self.storage_type, (bh.storage.Mean, bh.storage.WeightedMean))
            and sample is None
        ):
            msg = "A Mean-based storage requires a sample= argument with the values to average"
            raise TypeError(msg)

        data_dict = {
            self._name_to_index(k) if isinstance(k, str) else k: v  # type: ignore[redundant-expr]
            for k, v in kwargs.items()
        }

        if set(data_dict) != set(range(len(args), self.ndim)):
            expected = set(range(len(args), self.ndim))
            provided = set(data_dict)

            missing = expected - provided
            missing_values = [self.axes[i].name for i in missing]

            msg_0 = (
                f"Missing values for single/multiple axis : {missing_values}."
                f"Expected axes : {[ax.name for ax in self.axes]}"
            )
            raise TypeError(msg_0)

        data = (data_dict[i] for i in range(len(args), self.ndim))

        return super().fill(*args, *data, weight=weight, sample=sample, threads=threads)

    def fill_flattened(
        self: Self,
        *args: Any,
        weight: Any | None = None,
        sample: Any | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ) -> Self:
        axis_names = {ax.name for ax in self.axes}

        non_user_kwargs = {}
        if weight is not None:
            non_user_kwargs["weight"] = weight
        if sample is not None:
            non_user_kwargs["sample"] = sample

        # Single arguments are either arrays for single-dimensional histograms, or
        # structures for multi-dimensional hists that must first be unpacked
        if len(args) == 1 and not kwargs:
            (arg,) = args
            destructured = interop.destructure(arg)
            # Try to unpack the array, if it's valid to do so, i.e. is the Awkward Array a record array?
            if destructured is None:
                # Can't unpack, fall back on broadcasting single array (to flatten and convert)
                broadcast = interop.broadcast_and_flatten(
                    (arg, *non_user_kwargs.values())
                )
                # Partition out non-user args
                user_args_broadcast = broadcast[:1]
                user_kwargs_broadcast = {}
                non_user_kwargs_broadcast = dict(
                    zip(non_user_kwargs.keys(), broadcast[1:], strict=True)
                )
            else:
                # Result must be broadcast, so unpack and rebuild
                broadcast = interop.broadcast_and_flatten(
                    (*destructured.values(), *non_user_kwargs.values())
                )
                # Partition into user and non-user args
                user_args_broadcast = ()
                user_kwargs_broadcast = {
                    k: v
                    for k, v in zip(
                        destructured, broadcast[: len(destructured)], strict=True
                    )
                    if k in axis_names
                }
                non_user_kwargs_broadcast = dict(
                    zip(non_user_kwargs, broadcast[len(destructured) :], strict=True)
                )
        # Multiple args: broadcast and flatten!
        else:
            inputs = (*args, *kwargs.values(), *non_user_kwargs.values())
            broadcast = interop.broadcast_and_flatten(inputs)
            user_args_broadcast = broadcast[: len(args)]
            user_kwargs_broadcast = dict(
                zip(kwargs, broadcast[len(args) : len(args) + len(kwargs)], strict=True)
            )
            non_user_kwargs_broadcast = dict(
                zip(non_user_kwargs, broadcast[len(args) + len(kwargs) :], strict=True)
            )
        return self.fill(
            *user_args_broadcast,
            **user_kwargs_broadcast,
            threads=threads,
            **non_user_kwargs_broadcast,
        )

    def sort(
        self,
        axis: int | str,
        key: (
            Callable[[int], SupportsLessThan] | Callable[[str], SupportsLessThan] | None
        ) = None,
        reverse: bool = False,
    ) -> Self:
        """
        Sort a categorical axis.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sorted_cats = sorted(self.axes[axis], key=key, reverse=reverse)
            # This can only return Self, not float, etc., so we ignore the extra types here
            return self[{axis: [bh.loc(x) for x in sorted_cats]}]  # type: ignore[dict-item, return-value]

    def _convert_index_wildcards(self, x: Any, ax_id: str | int | None = None) -> Any:
        """
        Convert wildcards to available indices before passing to bh.loc
        """

        if not any(
            isinstance(x, t) for t in [str, list]
        ):  # Process only lists and strings
            return x
        _x = x if isinstance(x, list) else [x]  # Convert to list if not already
        if not all(isinstance(n, str) for n in _x):
            return x
        if any(any(special in pattern for special in ["*", "?"]) for pattern in _x):
            available = [n for n in self.axes[ax_id] if isinstance(n, str)]
            # Exact names are kept as-is, so a missing one still raises a
            # KeyError in bh.loc instead of being silently dropped.
            all_matches = [
                [k for k in available if fnmatch.fnmatch(k, pattern)]
                if any(special in pattern for special in ["*", "?"])
                else [pattern]
                for pattern in _x
            ]
            matches = list(
                dict.fromkeys(list(itertools.chain.from_iterable(all_matches)))
            )
            if len(matches) == 0:
                msg = f"No matches found for {x}"
                raise ValueError(msg)
            return matches
        return x

    def _loc_shortcut(self, x: Any, ax_id: str | int | None = None) -> Any:
        """
        Convert some specific indices to location.
        """
        x = self._convert_index_wildcards(x, ax_id)
        if isinstance(x, list):
            return [self._loc_shortcut(each) for each in x]
        if isinstance(x, slice):
            return slice(
                self._loc_shortcut(x.start),
                self._loc_shortcut(x.stop),
                self._step_shortcut(x.step),
            )
        if isinstance(x, complex):
            if x.real % 1 != 0:
                msg = "The real part should be an integer"
                raise ValueError(msg)
            return bh.loc(x.imag, int(x.real))
        if isinstance(x, str):
            return bh.loc(x)
        return x

    @staticmethod
    def _step_shortcut(x: Any) -> Any:
        """
        Convert some specific indices to step.
        """

        if not isinstance(x, complex):
            return x

        if x.real != 0:
            msg = "The step should not have real part"
            raise ValueError(msg)
        if x.imag % 1 != 0:
            msg = "The imaginary part should be an integer"
            raise ValueError(msg)
        return bh.rebin(int(x.imag))

    def _index_transform(self, index: list[IndexingExpr] | IndexingExpr) -> Any:
        """
        Auxiliary function for __getitem__ and __setitem__.
        """

        if isinstance(index, dict):
            new_indices = {
                (
                    self._name_to_index(k) if isinstance(k, str) else k
                ): self._loc_shortcut(v, k)
                for k, v in index.items()
            }
            if len(new_indices) != len(index):
                msg = "Duplicate index keys, numbers and names cannot overlap"
                raise ValueError(msg)
            return new_indices

        if not isinstance(index, tuple):
            index = (index,)  # type: ignore[assignment]

        # An Ellipsis stands in for the missing axes, so entries after it
        # refer to axes counted from the end.
        offset = 0
        new_index = []
        for i, v in enumerate(index):  # type: ignore[arg-type]
            if v is Ellipsis:
                offset = self.ndim - len(index)  # type: ignore[arg-type]
            new_index.append(self._loc_shortcut(v, i + offset))
        return tuple(new_index)

    @typing.overload  # type: ignore[override]
    def __getitem__(self: FloatHists, index: IndexingExpr) -> FloatHists | float: ...

    @typing.overload
    def __getitem__(self: IntHists, index: IndexingExpr) -> IntHists | int: ...

    @typing.overload
    def __getitem__(
        self: ListHists, index: IndexingExpr
    ) -> ListHists | list[float]: ...

    @typing.overload
    def __getitem__(
        self: WeightHists, index: IndexingExpr
    ) -> WeightHists | bh.accumulators.WeightedSum: ...

    @typing.overload
    def __getitem__(
        self: MeanHists, index: IndexingExpr
    ) -> MeanHists | bh.accumulators.Mean: ...

    @typing.overload
    def __getitem__(
        self: WeightedMeanHists, index: IndexingExpr
    ) -> WeightedMeanHists | bh.accumulators.WeightedMean: ...

    @typing.overload
    def __getitem__(
        self, index: IndexingExpr
    ) -> Self | float | list[float] | int | bh.accumulators.Accumulator: ...

    def __getitem__(
        self, index: IndexingExpr
    ) -> Self | float | int | list[float] | bh.accumulators.Accumulator:
        """
        Get histogram item.
        """

        return super().__getitem__(self._index_transform(index))

    def __setitem__(  # type: ignore[override]
        self, index: IndexingExpr, value: ArrayLike | bh.accumulators.Accumulator
    ) -> None:
        """
        Set histogram item.
        """

        return super().__setitem__(self._index_transform(index), value)

    def profile(self, axis: int | str) -> Self:
        """
        Returns a profile (Mean/WeightedMean) histogram from a normal histogram
        with N-1 axes. The axis given is profiled over and removed from the
        final histogram.
        """

        if self.kind != bh.Kind.COUNT:
            msg = "Profile requires a COUNT histogram"
            raise TypeError(msg)

        axes = list(self.axes)
        iaxis = axis if isinstance(axis, int) else self._name_to_index(axis)
        axes.pop(iaxis)

        values = self.values()
        tmp_variances = self.variances()
        variances = tmp_variances if tmp_variances is not None else values
        centers = self.axes[iaxis].centers

        count = np.sum(values, axis=iaxis)

        num = np.tensordot(values, centers, ([iaxis], [0]))
        num_err = np.sqrt(np.tensordot(variances, centers**2, ([iaxis], [0])))

        # The denominator is the same sum of weights as ``count``.
        den_err = np.sqrt(np.sum(variances, axis=iaxis))

        with np.errstate(invalid="ignore"):
            new_values = num / count
            new_variances = (num_err / count) ** 2 - (den_err * num / count**2) ** 2

        retval = self.__class__(*axes, storage=hist.storage.Mean())
        retval[...] = np.stack([count, new_values, count * new_variances], axis=-1)
        return retval

    def density(self) -> np.typing.NDArray[Any]:
        """
        Density NumPy array.
        """
        total = np.sum(self.values()) * functools.reduce(operator.mul, self.axes.widths)
        dens: np.typing.NDArray[Any] = self.values() / np.where(total > 0, total, 1)
        return dens

    def show(self, **kwargs: Any) -> Any:
        """
        Pretty print histograms to the console.
        """

        return histoprint.print_hist(self, **kwargs)

    def plot(
        self, *args: Any, overlay: str | None = None, **kwargs: Any
    ) -> Hist1DArtists | Hist2DArtists:
        """
        Plot the histogram with mplhep, picking a 1D or 2D plot automatically.

        * 1D histograms, and 2D histograms with one continuous and one
          categorical axis (or with ``overlay`` given), are drawn with
          :meth:`plot1d`, which wraps ``mplhep.histplot``. For 2D histograms,
          one line is drawn per bin of the overlay axis.
        * Other 2D histograms are drawn with :meth:`plot2d`, which wraps
          ``mplhep.hist2dplot``.
        * Histograms with more than two dimensions must be projected first.

        Parameters
        ----------
        *args : Any
            Forwarded to :meth:`plot1d` or :meth:`plot2d`; both only take
            keyword arguments, so pass options by name.
        overlay : str, optional
            Name of the axis to overlay for a 2D histogram. Giving it
            forces a 1D overlay plot. If None, a 2D histogram with one
            continuous and one categorical axis overlays the categorical axis.
        **kwargs : Any
            Forwarded to :meth:`plot1d` (``ax``, ``legend``, then
            ``mplhep.histplot``) or :meth:`plot2d` (``ax``, then
            ``mplhep.hist2dplot``). Any remaining keywords go to matplotlib,
            and ``ls``/``*_ls`` are accepted as aliases for
            ``linestyle``/``*_linestyle``. Commonly used ``mplhep.histplot``
            options include:

            - ``histtype``: ``"step"`` (default), ``"fill"``, ``"errorbar"``,
              or another type supported by the installed mplhep.
            - ``yerr``: ``True`` for Poisson/sqrt(N) errors, or an array.
            - ``w2method``: ``"poisson"``, ``"sqrt"``, or a callable.
            - ``stack``: stack the overlaid histograms.
            - ``density``/``binwnorm``: normalize to unit area/to bin width.
            - ``flow``: how to show under/overflow bins, e.g. ``"hint"``,
              ``"show"``, ``"sum"`` or ``"none"``.
            - ``label``: legend label; for an overlay, a single string or one
              label per category (defaults to the category names).
            - ``sort``: order overlaid histograms by ``"label"`` or ``"yield"``.

            Commonly used ``mplhep.hist2dplot`` options include ``cbar``,
            ``cmin``/``cmax``, ``labels`` (``True`` prints bin values) and
            ``flow``. See the mplhep documentation for the full list.

        Returns
        -------
        Hist1DArtists or Hist2DArtists
            The matplotlib artists returned by mplhep (a list of
            ``Hist1DArtists`` for 1D plots).

        Raises
        ------
        NotImplementedError
            If the histogram has more than two dimensions.
        """
        ordered_traits = sum(t.ordered for t in self.axes.traits)
        discrete_traits = sum(t.discrete for t in self.axes.traits)
        _has_categorical = ordered_traits == discrete_traits == 1
        _project = _has_categorical or overlay is not None

        if self.ndim == 1 or (self.ndim == 2 and _project):
            return self.plot1d(*args, overlay=overlay, **kwargs)

        if self.ndim == 2:
            return self.plot2d(*args, **kwargs)

        msg = "Please project to 1D or 2D before calling plot"
        raise NotImplementedError(msg)

    def plot1d(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        overlay: str | int | None = None,
        legend: bool = True,
        **kwargs: Any,
    ) -> Hist1DArtists:
        """
        Plot1d method for BaseHist object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes or creates new ones.
        overlay : str or int, optional
            Name or index of the axis to overlay. If None, automatically selects
            the first discrete axis for multi-dimensional histograms.
        legend : bool, default True
            Whether to automatically add a legend when plotting stacked categories.
            The legend title is set from the axis label if available.
        **kwargs : Any
            Additional keyword arguments passed to ``mplhep.histplot`` (for
            example ``histtype``, ``yerr``, ``stack``, ``density``, ``flow``
            or ``label``; see :meth:`plot`), and from there to matplotlib.

        Returns
        -------
        Hist1DArtists
            The matplotlib artists created by the plot.
        """

        from hist import plot

        if self.ndim == 1:
            return plot.histplot(self, ax=ax, **_proc_kw_for_lw(kwargs))
        if overlay is None:
            (overlay,) = (i for i, ax in enumerate(self.axes) if ax.traits.discrete)
        assert overlay is not None
        cat_ax = self.axes[overlay]
        icats = np.arange(len(cat_ax))
        cats = cat_ax if cat_ax.traits.discrete else icats
        d1hists = [self[{overlay: cat}] for cat in icats]
        if "label" in kwargs:
            if not isinstance(kwargs["label"], str) and len(kwargs["label"]) == len(
                cats
            ):
                cats = kwargs["label"]
                kwargs.pop("label")
            elif isinstance(kwargs["label"], str):
                cats = [kwargs["label"]] * len(cats)
                kwargs.pop("label")
            else:
                msg = f"label ``{kwargs['label']}`` not understood for {len(cats)} categories"
                raise ValueError(msg)
        artists = plot.histplot(d1hists, ax=ax, label=cats, **_proc_kw_for_lw(kwargs))
        if legend:
            # Try to set legend title from axis label if available
            if ax is None:
                # Get axis from the first artist (mplhep returns Hist1DArtists tuple)
                # This will raise an error if artists is empty or doesn't have the expected structure,
                # which is intended behavior as specified in the requirements
                ax = artists[0].stairs.axes
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                title = getattr(cat_ax, "label", None)
                ax.legend(title=title or None)
        return artists

    def plot2d(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Hist2DArtists:
        """
        Plot2d method for BaseHist object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes or creates new ones.
        **kwargs : Any
            Additional keyword arguments passed to ``mplhep.hist2dplot``, such
            as ``cbar``, ``cmin``/``cmax``, ``labels`` or ``flow``, and from
            there to matplotlib's ``pcolormesh``.

        Returns
        -------
        Hist2DArtists
            The matplotlib artists created by the plot.
        """

        from hist import plot

        return plot.hist2dplot(self, ax=ax, **_proc_kw_for_lw(kwargs))

    def plot2d_full(
        self,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[Hist2DArtists, Hist1DArtists, Hist1DArtists]:
        """
        Plot2d_full method for BaseHist object.

        Pass a dict of axes to ``ax_dict``, otherwise, the current figure will be used.
        """
        # Type judgement

        from hist import plot

        return plot.plot2d_full(self, ax_dict=ax_dict, **kwargs)

    def plot_ratio(
        self,
        other: hist.BaseHist[Any]
        | Callable[[np.typing.NDArray[Any]], np.typing.NDArray[Any]]
        | str,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[MainAxisArtists, RatiolikeArtists]:
        """
        Plot a 1D histogram and its ratio to another histogram or a fitted model.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``

        Parameters
        ----------
        other : hist.BaseHist, collections.abc.Callable, or str
            A histogram to compare with, or a model to fit to this histogram.
            The strings ``"gauss"``, ``"gaus"``, and ``"normal"`` select a Gaussian
            model with initial parameter estimates derived from the histogram.
            Other strings are Python expressions in ``x``, such as
            ``"a * x + b"``, where ``a`` and ``b`` are fit parameters. NumPy is
            available as ``np``. Only use trusted expressions, since they are
            evaluated as Python code.
        ax_dict : dict[str, matplotlib.axes.Axes]
            Axes for the histogram and ratio subplot, keyed by ``"main_ax"`` and
            ``"ratio_ax"``, respectively. Both keys are required when passing a
            nonempty dictionary. If omitted or empty, both axes are created in
            the current figure.
        """

        from hist import plot

        return plot._plot_ratiolike(
            self, other, ax_dict=ax_dict, view="ratio", **kwargs
        )

    def plot_pull(
        self,
        func: Callable[[np.typing.NDArray[Any]], np.typing.NDArray[Any]] | str,
        *,
        ax_dict: dict[str, matplotlib.axes.Axes] | None = None,
        **kwargs: Any,
    ) -> tuple[FitResultArtists, RatiolikeArtists]:
        """
        Plot a 1D histogram with a fitted model and a pull subplot.

        Return a tuple of artists following a structure of
        ``(main_ax_artists, subplot_ax_artists)``

        Parameters
        ----------
        func : collections.abc.Callable or str
            A model to fit to this histogram. The strings ``"gauss"``, ``"gaus"``,
            and ``"normal"`` select a Gaussian model with initial parameter
            estimates derived from the histogram. Other strings are Python
            expressions in ``x``, such as ``"a * x + b"``, where ``a`` and ``b``
            are fit parameters. NumPy is available as ``np``. Only use trusted
            expressions, since they are evaluated as Python code.
        ax_dict : dict[str, matplotlib.axes.Axes]
            Axes for the histogram and pull subplot, keyed by ``"main_ax"`` and
            ``"pull_ax"``, respectively. Both keys are required when passing a
            nonempty dictionary. If omitted or empty, both axes are created in
            the current figure.
        """

        from hist import plot

        return plot._plot_ratiolike(self, func, ax_dict=ax_dict, view="pull", **kwargs)

    def plot_pie(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Any:
        from hist import plot

        return plot.plot_pie(self, ax=ax, **kwargs)

    def stack(self, axis: int | str) -> hist.stack.Stack[S]:
        """
        Returns a stack from a normal histogram axes.
        """
        if self.ndim < 2:
            msg = "Cannot stack with less than two axis"
            raise RuntimeError(msg)
        stack_histograms: Iterator[BaseHist[S]] = [  # type: ignore[assignment]
            self[{axis: i}] for i in range(len(self.axes[axis]))
        ]
        for name, h in zip(self.axes[axis], stack_histograms, strict=True):
            h.name = name

        return hist.stack.Stack(*stack_histograms)

    def expand_cats(self, *, name: Callable[..., str] | None = None) -> dict[str, Self]:
        """
        Expand all categorical axes into a dict of histograms.

        One histogram is produced for every combination of categories. The
        ``name`` callable receives the category values, one per categorical
        axis, and returns the key; the default joins them with ``_``.
        Keys must be unique across all combinations, including empty groups.
        A duplicate key raises ``ValueError``; pass ``name`` to choose unique keys.
        """
        cat_axes = [
            i
            for i, ax in enumerate(self.axes)
            if isinstance(ax, (bh.axis.IntCategory, bh.axis.StrCategory))
        ]
        if not cat_axes:
            msg = "Cannot expand a histogram without categorical axes"
            raise ValueError(msg)

        if name is None:

            def name(*cats: int | str) -> str:
                return "_".join(str(c) for c in cats)

        result: dict[str, Self] = {}
        for cats in itertools.product(*(self.axes[i] for i in cat_axes)):
            key = name(*cats)
            if key in result:
                msg = f"Duplicate histogram name {key!r} from categorical expansion; use name= to generate unique names"
                raise ValueError(msg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # NamedHist refuses integer keys
                index: dict[int | str, Any] = {
                    self.axes[i].name or i: bh.loc(c)
                    for i, c in zip(cat_axes, cats, strict=True)
                }
                h: Self = self[index]  # type: ignore[assignment]
            h.name = key
            result[key] = h
        return result

    @typing.overload
    def integrate(
        self,
        name: int | str,
        i_or_list: list[str | int],
        j: InnerIndexing | None = None,
    ) -> Self: ...

    @typing.overload
    def integrate(
        self: IntHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> IntHists | int: ...

    @typing.overload
    def integrate(
        self: FloatHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> FloatHists | float: ...

    @typing.overload
    def integrate(
        self: ListHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> ListHists | list[float]: ...

    @typing.overload
    def integrate(
        self: WeightHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> WeightHists | bh.accumulators.WeightedSum: ...

    @typing.overload
    def integrate(
        self: MeanHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> MeanHists | bh.accumulators.Mean: ...

    @typing.overload
    def integrate(
        self: WeightedMeanHists,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> WeightedMeanHists | bh.accumulators.WeightedMean: ...

    @typing.overload
    def integrate(
        self,
        name: int | str,
        i_or_list: InnerIndexing | None = None,
        j: InnerIndexing | None = None,
    ) -> Self | int | float | list[float] | bh.accumulators.Accumulator: ...

    def integrate(
        self,
        name: int | str,
        i_or_list: InnerIndexing | list[str | int] | None = None,
        j: InnerIndexing | None = None,
    ) -> Self | int | float | list[float] | bh.accumulators.Accumulator:
        if isinstance(i_or_list, list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                selection: Self = self[{name: i_or_list}]  # type: ignore[assignment, dict-item]
                return selection[{name: slice(0, len(selection.axes[name]), sum)}]

        return self[{name: slice(i_or_list, j, sum)}]

    def rebin_lossless(
        self, axis: int | str, edges: Sequence[float]
    ) -> Self:
        """
        Merge complete bins of one continuous axis onto coarser edges.

        The rebin is strictly lossless: every requested edge in ``edges`` must
        coincide with an existing edge of ``axis``, so each new bin is the sum
        of whole original bins. Counts and (where the storage carries them)
        variances or mean accumulators are added bin by bin. If a requested
        edge would split an original bin, no approximation is made — a
        ``ValueError`` names the bin that cannot be decomposed, and this
        histogram is left untouched so it can be reused with another binning
        or rebuilt from the original data.

        Underflow and overflow are not mixed into the first and last regular
        bins: the flow bins are copied unchanged into the result, and flow
        stays disabled on axes that had none.

        Parameters
        ----------
        axis : int or str
            Index or name of the axis to rebin. It must be a continuous,
            non-circular axis (``Regular``, ``Variable`` or ``Integer``);
            categorical and circular axes are rejected with ``TypeError``.
        edges : sequence of float
            Full edges of the requested coarse binning, from the axis lower
            edge to its upper edge, strictly increasing. Each edge must match
            an existing axis edge.

        Returns
        -------
        BaseHist
            A new histogram with a ``Variable`` axis carrying the requested
            edges; all other axes, the storage and the axis name/label are
            preserved. The original histogram is not modified.

        Raises
        ------
        TypeError
            If the axis is categorical or circular, or the storage is
            ``MultiCell``.
        ValueError
            If ``edges`` is not strictly increasing, does not span the whole
            axis, or contains an edge that cuts through an original bin.
        """
        iaxis = self._name_to_index(axis) if isinstance(axis, str) else axis
        old_axis = self.axes[iaxis]
        if isinstance(
            old_axis,
            (bh.axis.IntCategory, bh.axis.StrCategory, bh.axis.Boolean),
        ):
            msg = (
                f"Axis {iaxis!r} is categorical; rebin_lossless only supports "
                "continuous axes"
            )
            raise TypeError(msg)
        if old_axis.traits.circular:
            msg = (
                f"Axis {iaxis!r} is circular; rebin_lossless does not support "
                "circular axes"
            )
            raise TypeError(msg)
        if issubclass(self.storage_type, bh.storage.MultiCell):
            msg = "rebin_lossless does not support MultiCell storage"
            raise TypeError(msg)

        old_edges = np.asarray(old_axis.edges, dtype=float)
        groups = _match_rebin_edges(edges, old_edges)

        new_axis = hist.axis.Variable(
            old_edges[groups],
            name=old_axis.name,
            label=old_axis.label,
            underflow=old_axis.traits.underflow,
            overflow=old_axis.traits.overflow,
        )
        new_axes = [*self.axes]
        new_axes[iaxis] = new_axis
        result = self.__class__(*new_axes, storage=self.storage_type())

        source = self.view(flow=True)
        target = result.view(flow=True)
        has_underflow = bool(old_axis.traits.underflow)
        has_overflow = bool(old_axis.traits.overflow)
        offset = 1 if has_underflow else 0
        n_bins = len(old_axis)
        # First and last original-bin indices (flow-inclusive) of each group
        group_starts = (groups[:-1] + offset).tolist()
        group_stops = (groups[1:] + offset).tolist()
        out_regular = slice(offset, offset + len(group_starts))

        source_array = np.asarray(source)
        is_mean_storage = issubclass(
            self.storage_type, (bh.storage.Mean, bh.storage.WeightedMean)
        )

        if is_mean_storage:
            # Mean-family accumulators keep a running mean; adding their raw
            # value field would be wrong, so add the accumulators themselves.
            self._merge_mean_axis(
                source,
                target,
                iaxis,
                group_starts,
                group_stops,
                offset,
                has_underflow,
                has_overflow,
            )
        else:
            # Numeric storages. Weight storage carries two additive fields
            # (value, variance); the plain storages have a single array. A
            # dummy final start keeps reduceat from swallowing the overflow;
            # its result is discarded.
            reduce_indices = (
                [*group_starts, n_bins + offset]
                if has_overflow
                else group_starts
            )
            field_names = typing.cast(
                "tuple[str, ...] | None", source_array.dtype.names
            )
            indexer: list[Any] = [slice(None)] * source_array.ndim
            indexer[iaxis] = out_regular
            write_index = tuple(indexer)
            if field_names is None:
                merged = np.add.reduceat(
                    source_array, reduce_indices, axis=iaxis
                )
                target[write_index] = np.take(
                    merged, range(len(group_starts)), axis=iaxis
                )
            else:
                for field in field_names:
                    merged = np.add.reduceat(
                        source_array[field], reduce_indices, axis=iaxis
                    )
                    target[field][write_index] = np.take(
                        merged, range(len(group_starts)), axis=iaxis
                    )
            self._copy_flow_bins(
                source_array,
                target,
                iaxis,
                n_bins,
                offset,
                len(group_starts),
                has_underflow,
                has_overflow,
                field_names=field_names,
            )

        return result

    @staticmethod
    def _copy_flow_bins(
        source: np.typing.NDArray[Any],
        target: Any,
        iaxis: int,
        n_bins: int,
        offset: int,
        n_groups: int,
        has_underflow: bool,
        has_overflow: bool,
        field_names: tuple[str, ...] | None = None,
    ) -> None:
        # Flow bins are copied verbatim; they must never enter regular bins.
        def flow_index(position: int) -> tuple[Any, ...]:
            indexer: list[Any] = [slice(None)] * source.ndim
            indexer[iaxis] = position
            return tuple(indexer)

        if has_underflow:
            under = flow_index(0)
            if field_names is None:
                target[under] = source[under]
            else:
                for field in field_names:
                    target[field][under] = source[field][under]
        if has_overflow:
            src_over = flow_index(n_bins + offset)
            dst_over = flow_index(n_groups + offset)
            if field_names is None:
                target[dst_over] = source[src_over]
            else:
                for field in field_names:
                    target[field][dst_over] = source[field][src_over]

    @staticmethod
    def _merge_mean_axis(
        source: Any,
        target: Any,
        iaxis: int,
        group_starts: list[int],
        group_stops: list[int],
        offset: int,
        has_underflow: bool,
        has_overflow: bool,
    ) -> None:
        # Iterate over every line parallel to the rebinned axis and add the
        # accumulators of the original bins in each group.
        src_axis_size = source.shape[iaxis]
        dst_axis_size = target.shape[iaxis]
        line_shape = list(source.shape)
        del line_shape[iaxis]
        for line_index in np.ndindex(tuple(line_shape)):
            src_line_index: list[Any] = list(line_index)
            dst_line_index: list[Any] = list(line_index)
            src_line_index.insert(iaxis, slice(None))
            dst_line_index.insert(iaxis, slice(None))
            src_line = source[tuple(src_line_index)]
            dst_line = target[tuple(dst_line_index)]
            for out_pos, (start, stop) in enumerate(
                zip(group_starts, group_stops, strict=True)
            ):
                accumulator = src_line[start]
                for bin_index in range(start + 1, stop):
                    accumulator = accumulator + src_line[bin_index]
                dst_line[offset + out_pos] = accumulator
            if has_underflow:
                dst_line[0] = src_line[0]
            if has_overflow:
                dst_line[dst_axis_size - 1] = src_line[src_axis_size - 1]
