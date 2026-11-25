.. _usage-analyses:

Analyses examples
=================

Bool and category axes
----------------------

Taken together, the flexibility in axes and the tools to easily sum over
axes can be applied to transform the way you approach analysis with
histograms. For example, let’s say you are presented with the following
data in a 3xN table:

============== ========================
Data           Details
============== ========================
``value``
``is_valid``   True or False
``run_number`` A collection of integers
============== ========================

In a traditional analysis, you might bin over ``value`` where
``is_valid`` is True, and then make a collection of histograms, one for
each run number. With hist, you can make a single histogram,
and use an axis for each:

.. code-block:: python3

   value_ax = hist.axis.Regular(100, -5, 5)
   bool_ax = hist.axis.Integer(0, 2, underflow=False, overflow=False)
   run_number_ax = hist.axis.IntCategory([], growth=True)

Now, you can use these axes to create a single histogram that you can
fill. If you want to get a histogram of all run numbers and just the
True ``is_valid`` selection, you can use a ``sum``:

.. code-block:: python3

   h1 = hist[:, True, sum]

You can expand this example to any number of dimensions, boolean flags,
and categories.

Lossless rebinning onto common edges
-------------------------------------

Different instruments often store the same observable with fine or coarse
binning. To compare the histograms bin by bin they must first share a
common set of edges. ``rebin_lossless`` merges *whole* fine bins into
coarser bins, provided every requested edge coincides with an existing
edge:

.. code-block:: python3

   fine = hist.Hist(hist.axis.Variable([0, 1, 2, 3, 4], name="x"))
   fine.fill([0.5, 1.5, 2.5, 3.5, 3.5])

   coarse = fine.rebin_lossless("x", [0, 2, 4])
   coarse.axes[0].edges  # array([0., 2., 4.])

The merge adds complete bins only, so the total count and the variance
(for ``Weight`` storage) are preserved, and mean accumulators are pooled
correctly. Underflow and overflow bins are copied unchanged and are never
folded into the first or last regular bin.

If a requested edge falls inside an original bin, that bin cannot be split
from binned data alone, so no approximate allocation is made. A
``ValueError`` names the bin that cannot be decomposed and the original
histogram is left untouched, ready to be reused with another binning or
rebuilt from the original data:

.. code-block:: python3

   fine.rebin_lossless("x", [0, 1.5, 4])
   # ValueError: Requested edge 1.5 cuts through original bin 1 with edges
   # [1.0, 2.0]. That bin cannot be split from binned data alone; rebuild the
   # histogram from the original data with the new binning

Categorical and circular axes cannot be rebinned. The rebinned axis is
always returned as a ``Variable`` axis, keeping its name, label and flow
settings.
