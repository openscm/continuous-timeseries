# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Representations
#
# Here we explain our approach to representing our objects,
# particularly in IPython notebooks.
#
# As background, the way we approach this is based on three key sources:
#
# - the difference between `__repr__` and `__str__` in Python
#   (see e.g. https://realpython.com/python-repr-vs-str/)
# - the advice from the IPython docs about prettifying output
#   (https://ipython.readthedocs.io/en/8.26.0/config/integrating.html#rich-display)
# - the way that xarray handles formatting
#   (see https://github.com/pydata/xarray/blob/main/xarray/core/formatting.py)
# - the way that pint handles formatting
#   (see https://github.com/hgrecco/pint/blob/74b708661577623c0c288933d8ed6271f32a4b8b/pint/util.py#L1004)
#
# In short, we try and have as nice an experience for developers as possible.
#
# (As one other note/trick for representation of objects,
# you can control how numpy represents its objects using
# [numpy.set_printoptions](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html)).

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pint
from IPython.lib.pretty import pretty

import continuous_timeseries as ct

# %% [markdown]
# ## Set up pint

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %% [markdown]
# ## Set up some example objects

# %%
basic_ts = ct.TimeseriesDiscrete(
    name="basic",
    time_axis=ct.TimeAxis(Q([1750.0, 1850.0, 1950.0], "yr")),
    values_at_bounds=ct.ValuesAtBounds(Q([1.0, 2.0, 3.0], "m")),
)
basic_ts

# %%
long_ts = ct.TimeseriesDiscrete(
    name="basic",
    time_axis=ct.TimeAxis(Q(np.arange(1850.0, 2300.0, 1), "yr")),
    values_at_bounds=ct.ValuesAtBounds(Q(np.arange(450.0), "m")),
)
long_ts

# %%
really_long_ts = ct.TimeseriesDiscrete(
    name="basic",
    time_axis=ct.TimeAxis(Q(np.arange(1850.0, 2300.0, 1 / 12), "yr")),
    values_at_bounds=ct.ValuesAtBounds(Q(np.arange(450.0 * 12), "m")),
)
really_long_ts

# %% [markdown]
# ## HTML representation
#
# In a notebook environment, the default view is the HTML representation.
# If you're running this in a notebook, that is what you will have already seen above.

# %% [markdown]
# As a reminder, here is the default view.

# %%
basic_ts

# %% [markdown]
# Here is the HTML representation of the wrapped values.

# %%
basic_ts.values_at_bounds

# %%
basic_ts.values_at_bounds.values

# %% [markdown]
# Here is the raw HTML which is generated and sits underneath this view.

# %%
print(basic_ts._repr_html_())

# %%
long_ts._repr_html_()

# %%
really_long_ts._repr_html_()

# %% [markdown]
# ## Pretty representation
#
# There is also the pretty representation,
# which is used by the IPython `pretty` module
# (https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.lib.pretty.html#module-IPython.lib.pretty,
# not to be confused with the `pprint` module).

# %%
print(pretty(basic_ts))

# %%
print(pretty(basic_ts.values_at_bounds))

# %%
print(pretty(basic_ts.values_at_bounds.values))

# %%
print(pretty(long_ts))

# %%
print(pretty(really_long_ts))

# %% [markdown]
# ## String representation
#
# The string representations are intended for users.
# They are generally quite helpful.
# We let the underlying libraries handle most of the formatting decisions.

# %%
str(basic_ts)

# %% [markdown]
# The value displayed for the attributes of the object
# are simply the string representations of themselves.

# %%
str(basic_ts.values_at_bounds)

# %%
str(basic_ts.values_at_bounds.values)

# %% [markdown]
# With a large array,
# this leads to the slightly odd behaviour of showing all the values,
# as shown below.

# %%
str(long_ts)

# %% [markdown]
# For whatever reason, this is the behaviour of the underlying packages.

# %%
str(long_ts.values_at_bounds.values)

# %% [markdown]
# If we go to an even larger array, not all values are shown
# (which seems a more sensible choice to us).

# %%
str(really_long_ts)

# %%
str(really_long_ts.values_at_bounds.values)

# %% [markdown]
# ## `repr` representation
#
# The `repr` representation (which internally calls the `__repr__` method)
# is intended for developers,
# i.e. to allow cutting and pasting the output into Python directly
# (although neither numpy nor pint follows this exactly in all cases,
# and we don't try to change/fix this, their developers know better than us).
# As a result, it can be quite unwieldy.

# %%
repr(basic_ts)

# %%
repr(basic_ts.values_at_bounds)

# %%
# pint's output is not copy-pasteable because of the surrounding "<>"
# and lack of commas between numerical values.
repr(basic_ts.values_at_bounds.values)

# %%
# numpy does give copy-pasteable output for basic arrays
repr(basic_ts.values_at_bounds.values.m)

# %%
repr(long_ts)

# %%
repr(really_long_ts)

# %%
# numpy doesn't give copy-pasteable output for really large arrays
repr(really_long_ts.values_at_bounds.values.m)
