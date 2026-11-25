from __future__ import annotations

import boost_histogram as bh
import numpy as np
import pytest

import hist
from hist import Hist, NamedHist, axis, storage


def test_lossless_merge_variable_weighted() -> None:
    h = Hist(
        axis.Variable([0, 1, 2, 3, 4], name="x"), storage=storage.Weight()
    )
    h.fill([-1, 0.5, 1.5, 2.5, 3.5, 5], weight=[9, 1, 2, 3, 4, 25])

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert isinstance(coarse.axes[0], hist.axis.Variable)
    assert coarse.axes[0].edges.tolist() == [0, 2, 4]
    assert coarse.axes[0].name == "x"

    view = np.asarray(coarse.view(flow=True))
    # Whole fine bins are merged: counts add, squared weights (variance) add
    assert view["value"].tolist() == [9, 3, 7, 25]
    assert view["variance"].tolist() == [81, 5, 25, 625]
    # Flow is kept separate, never folded into the regular bins
    assert coarse.sum(flow=True) == h.sum(flow=True)
    assert coarse.sum() == h.sum()


def test_lossless_merge_regular_double() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"))
    h.fill([0.5, 1.5, 2.5, 3.5, 3.5])

    coarse = h.rebin_lossless(0, [0, 1, 3, 4])

    assert coarse.axes[0].edges.tolist() == [0, 1, 3, 4]
    assert np.asarray(coarse.view(flow=True)).tolist() == [0, 1, 2, 2, 0]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_merge_uneven_groups() -> None:
    h = Hist(axis.Variable([0, 1, 2, 4, 8], name="x"))
    h.fill([0.5, 1.5, 3.0, 5.0, 7.0])

    coarse = h.rebin_lossless("x", [0, 4, 8])

    assert coarse.axes[0].edges.tolist() == [0, 4, 8]
    assert np.asarray(coarse.view()).tolist() == [3, 2]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_merge_2d_touches_one_axis() -> None:
    h = Hist(
        axis.Regular(4, 0, 4, name="x"),
        axis.Regular(3, 0, 3, name="y"),
        storage=storage.Weight(),
    )
    h.fill(
        [0.5, 1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5, 0.5],
        weight=[1, 2, 3, 4],
    )

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert isinstance(coarse.axes[1], hist.axis.Regular)
    assert len(coarse.axes[1]) == 3
    assert coarse.axes[0].edges.tolist() == [0, 2, 4]
    assert coarse.sum(flow=True) == h.sum(flow=True)
    # Each merged cell is the sum of whole source cells
    src = np.asarray(h.view(flow=True))
    dst = np.asarray(coarse.view(flow=True))
    assert dst["value"][1].tolist() == (src["value"][1] + src["value"][2]).tolist()
    assert dst["value"][2].tolist() == (src["value"][3] + src["value"][4]).tolist()


def test_lossless_merge_mean_storage() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"), storage=storage.Mean())
    h.fill([0.5, 0.5, 1.5, 2.5, 3.5], sample=[10, 12, 20, 30, 40])

    coarse = h.rebin_lossless(0, [0, 2, 4])

    src_total = h.sum(flow=True)
    dst_total = coarse.sum(flow=True)
    # Counts and accumulator state survive; adding Mean accumulators pools
    # the means rather than summing them.
    assert dst_total.count == src_total.count
    assert dst_total.value == pytest.approx(
        (2 * 11 + 3 * 30) / 5
    )


def test_lossless_merge_weighted_mean_storage() -> None:
    h = Hist(
        axis.Regular(4, 0, 4, name="x"), storage=storage.WeightedMean()
    )
    h.fill([0.5, 1.5, 2.5, 3.5], sample=[10, 20, 30, 40], weight=[1, 1, 2, 1])

    coarse = h.rebin_lossless(0, [0, 2, 4])

    assert (
        coarse.sum(flow=True).sum_of_weights
        == h.sum(flow=True).sum_of_weights
    )
    assert (
        coarse.sum(flow=True).sum_of_weights_squared
        == h.sum(flow=True).sum_of_weights_squared
    )


def test_lossless_merge_integer_and_atomic() -> None:
    h = Hist(axis.Integer(0, 4, name="i"), storage=storage.AtomicInt64())
    h.fill([0, 1, 2, 3, 3])

    coarse = h.rebin_lossless("i", [0, 2, 4])

    # An Integer axis rebinning produces a Variable axis
    assert isinstance(coarse.axes[0], hist.axis.Variable)
    assert coarse.axes[0].edges.tolist() == [0, 2, 4]
    assert np.asarray(coarse.view()).tolist() == [2, 3]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_identity_rebin() -> None:
    h = Hist(axis.Variable([0, 1, 2, 3], name="x"), storage=storage.Weight())
    h.fill([0.5, 2.5], weight=[2, 3])

    same = h.rebin_lossless("x", [0, 1, 2, 3])

    assert same.axes[0].edges.tolist() == [0, 1, 2, 3]
    assert np.array_equal(
        np.asarray(same.view(flow=True)), np.asarray(h.view(flow=True))
    )


def test_lossless_rebin_does_not_mutate_source() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"))
    h.fill([0.5, 1.5, 2.5, 3.5])
    before = np.asarray(h.view(flow=True)).copy()

    _ = h.rebin_lossless(0, [0, 2, 4])

    assert np.array_equal(np.asarray(h.view(flow=True)), before)
    assert h.axes[0].edges.tolist() == [0, 1, 2, 3, 4]


def test_lossless_rebin_no_flow_axis() -> None:
    h = Hist(
        axis.Variable([0, 1, 2, 3, 4], name="x", flow=False),
        storage=storage.Weight(),
    )
    h.fill([0.5, 1.5, 2.5, 3.5], weight=[1, 2, 3, 4])

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert not coarse.axes[0].traits.underflow
    assert not coarse.axes[0].traits.overflow
    view = np.asarray(coarse.view(flow=True))
    assert view["value"].tolist() == [3, 7]
    assert view["variance"].tolist() == [5, 25]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_rebin_partial_flow() -> None:
    h = Hist(
        axis.Variable(
            [0, 1, 2, 3, 4], name="x", underflow=True, overflow=False
        ),
        storage=storage.Weight(),
    )
    h.fill([-1, 0.5, 1.5, 2.5, 3.5], weight=[9, 1, 2, 3, 4])

    coarse = h.rebin_lossless(0, [0, 2, 4])

    assert coarse.axes[0].traits.underflow
    assert not coarse.axes[0].traits.overflow
    view = np.asarray(coarse.view(flow=True))
    # Underflow survives untouched; no overflow slot exists
    assert view["value"].tolist() == [9, 3, 7]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_cut_edge_names_the_unsplittable_bin() -> None:
    h = Hist(axis.Variable([0, 1, 2, 3, 4], name="x"))

    with pytest.raises(ValueError, match="cuts through original bin 1"):
        h.rebin_lossless("x", [0, 1.5, 4])

    # Source histogram is intact and reusable
    assert h.axes[0].edges.tolist() == [0, 1, 2, 3, 4]
    assert h.sum(flow=True) == 0


def test_cut_edge_single_nonempty_bin_not_approximated() -> None:
    h = Hist(axis.Variable([0, 1, 2, 3], name="x"))
    h.fill([1.5])

    with pytest.raises(ValueError, match="cuts through original bin 1"):
        h.rebin_lossless("x", [0, 1.5, 3])

    # The one event is still entirely in the source flow/regular bins
    assert h.sum(flow=True) == 1


def test_rebin_edges_must_span_whole_axis() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"))

    with pytest.raises(ValueError, match="lower edge"):
        h.rebin_lossless(0, [1, 2, 4])
    with pytest.raises(ValueError, match="upper edge"):
        h.rebin_lossless(0, [0, 2, 5])


def test_rebin_edges_must_be_strictly_increasing() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"))

    with pytest.raises(ValueError, match="strictly increasing"):
        h.rebin_lossless(0, [0, 2, 2, 4])


def test_rebin_edges_need_at_least_two() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"))

    with pytest.raises(ValueError, match="At least two edges"):
        h.rebin_lossless(0, [0])  # type: ignore[arg-type]


def test_rebin_matches_edges_within_floating_tolerance() -> None:
    # np.linspace and the axis edges differ by a few ULPs at large bin counts
    h = Hist(axis.Regular(100_000, 0, 1, name="x"))
    edges = np.linspace(0, 1, 1001)

    coarse = h.rebin_lossless("x", edges)

    assert len(coarse.axes[0]) == 1000
    assert coarse.sum(flow=True) == h.sum(flow=True)

    # A perturbation of one ULP still maps onto the same original edge
    perturbed = edges.copy()
    perturbed[10] = np.nextafter(edges[10], np.inf)
    near = h.rebin_lossless("x", perturbed)
    assert near.axes[0].edges[10] == coarse.axes[0].edges[10]


def test_rebin_rejects_cut_at_large_scale() -> None:
    h = Hist(axis.Regular(1000, 0, 1e6, name="x"))
    edges = np.linspace(0, 1e6, 11)
    edges[3] += 1.0  # far above tolerance (~1e-6), far below the bin width

    with pytest.raises(ValueError, match="cuts through original bin 300"):
        h.rebin_lossless("x", edges)


def test_categorical_axis_rejected() -> None:
    h = Hist(axis.IntCategory([0, 1, 2], name="c"))

    with pytest.raises(TypeError, match="categorical"):
        h.rebin_lossless("c", [0, 1, 2])

    hs = Hist(axis.StrCategory(["a", "b"], name="s"))
    with pytest.raises(TypeError, match="categorical"):
        hs.rebin_lossless(0, [0, 1])


def test_circular_axis_rejected() -> None:
    h = Hist(
        axis.Regular(4, 0, 4, name="x", circular=True),
    )

    with pytest.raises(TypeError, match="circular"):
        h.rebin_lossless(0, [0, 2, 4])


def test_multicell_storage_rejected() -> None:
    # MultiCell needs an explicit capacity argument and holds per-event lists,
    # which cannot be merged like count/weight accumulators.
    multi = hist.new.Variable([0, 1, 2, 3]).MultiCell(4)
    assert issubclass(multi.storage_type, bh.storage.MultiCell)

    with pytest.raises(TypeError, match="MultiCell"):
        multi.rebin_lossless(0, [0, 2, 3])


def test_named_hist_lossless_rebin() -> None:
    h = NamedHist(axis.Regular(4, 0, 4, name="x"))
    h.fill(x=[0.5, 1.5, 2.5, 3.5])

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert isinstance(coarse, NamedHist)
    assert coarse.axes[0].edges.tolist() == [0, 2, 4]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_rebin_label_preserved() -> None:
    h = Hist(
        axis.Regular(4, 0, 4, name="x", label="measured [units]")
    )

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert coarse.axes[0].label == "measured [units]"
    assert coarse.axes[0].name == "x"


def test_lossless_rebin_unlimited_storage() -> None:
    h = Hist(axis.Regular(4, 0, 4, name="x"), storage=storage.Unlimited())
    h.fill([0.5, 1.5, 2.5, 3.5, 3.5])

    coarse = h.rebin_lossless("x", [0, 2, 4])

    assert np.asarray(coarse.view(flow=True)).tolist() == [0, 2, 3, 0]
    assert coarse.sum(flow=True) == h.sum(flow=True)


def test_lossless_rebin_accepts_array_and_tuple_edges() -> None:
    h = Hist(axis.Integer(0, 4, name="i"))
    h.fill([0, 1, 2, 3, 3])

    from_array = h.rebin_lossless("i", np.array([0, 2, 4]))
    from_tuple = h.rebin_lossless(0, (0, 2, 4))

    assert from_array.axes[0].edges.tolist() == [0.0, 2.0, 4.0]
    assert from_tuple.axes[0].edges.tolist() == [0.0, 2.0, 4.0]


def test_lossless_rebin_3d_weighted_middle_axis() -> None:
    rng = np.random.default_rng(42)
    h = Hist(
        axis.Regular(4, 0, 4, name="x"),
        axis.Variable([0, 1, 2, 4, 8], name="y"),
        axis.Regular(3, 0, 3, name="z"),
        storage=storage.Weight(),
    )
    h.fill(
        rng.uniform(0, 4, 50),
        rng.uniform(0, 8, 50),
        rng.uniform(0, 3, 50),
        weight=rng.uniform(0.5, 2, 50),
    )

    coarse = h.rebin_lossless("y", [0, 2, 8])

    assert coarse.axes[1].edges.tolist() == [0, 2, 8]
    # Untouched axes keep their edges exactly
    assert coarse.axes[0].edges.tolist() == h.axes[0].edges.tolist()
    assert coarse.axes[2].edges.tolist() == h.axes[2].edges.tolist()

    src = np.asarray(h.view(flow=True))
    dst = np.asarray(coarse.view(flow=True))
    # Squared weights (variance) accumulate for each fully merged pair
    np.testing.assert_allclose(
        dst["variance"][:, 1, :],
        src["variance"][:, 1, :] + src["variance"][:, 2, :],
    )
    np.testing.assert_allclose(
        dst["variance"][:, 2, :],
        src["variance"][:, 3, :] + src["variance"][:, 4, :],
    )
    # Flow bins along the rebinned axis are copied, not merged in
    np.testing.assert_array_equal(
        dst["value"][:, 0, :], src["value"][:, 0, :]
    )
    np.testing.assert_array_equal(
        dst["value"][:, -1, :], src["value"][:, -1, :]
    )


def test_lossless_rebin_3d_mean_middle_axis() -> None:
    rng = np.random.default_rng(7)
    h = Hist(
        axis.Regular(2, 0, 2, name="a"),
        axis.Regular(4, 0, 4, name="b"),
        axis.Regular(2, 0, 2, name="c"),
        storage=storage.WeightedMean(),
    )
    n = 40
    h.fill(
        rng.uniform(0, 2, n),
        rng.uniform(0, 4, n),
        rng.uniform(0, 2, n),
        sample=rng.normal(size=n),
        weight=rng.uniform(0.5, 1.5, n),
    )

    coarse = h.rebin_lossless("b", [0, 2, 4])

    src_total = h.sum(flow=True)
    dst_total = coarse.sum(flow=True)
    assert dst_total.sum_of_weights == src_total.sum_of_weights
    assert (
        dst_total.sum_of_weights_squared
        == src_total.sum_of_weights_squared
    )
    assert dst_total.value == pytest.approx(src_total.value)
