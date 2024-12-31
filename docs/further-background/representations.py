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
# Here we explain our approach to representing our objects.
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

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pint
from IPython.lib.pretty import pretty

from continuous_timeseries.values_at_bounds import ValuesAtBounds

# %% [markdown]
# ## Set up pint

# %%
UR = pint.get_application_registry()
Q = UR.Quantity

# %% [markdown]
# ## Set up some example objects

# %%
basic_array = ValuesAtBounds(Q([1, 2, 3], "m"))
basic_array

# %%
large_array = ValuesAtBounds(Q(np.arange(1850, 2300, 1), "m"))
large_array

# %%
really_large_array = ValuesAtBounds(Q(np.arange(1850, 2300, 1 / 12), "m"))
really_large_array

# %% [markdown]
# ## HTML representation
#
# In a notebook environment, the default view is the HTML representation.
# If you're running this in a notebook, that is what you will have already seen above.
# We show its raw value below.

# %% [markdown]
# To make the point clearer, here is the default view.

# %%
basic_array

# %% [markdown]
# Here is the HTML representation of the wrapped values.

# %%
basic_array.values

# %% [markdown]
# Here is the raw HTML which is generated.

# %%
basic_array._repr_html_()

# %%
large_array._repr_html_()

# %%
really_large_array._repr_html_()

# %% [markdown]
# ## Pretty representation
#
# There is also the pretty representation,
# which is used by the IPython `pretty` module
# (https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.lib.pretty.html#module-IPython.lib.pretty,
# not to be confused with the `pprint` module).

# %%
pretty(basic_array)

# %%
pretty(basic_array.values)

# %%
pretty(large_array)

# %%
pretty(really_large_array)

# %% [markdown]
# ## String representation
#
# The string representations are intended for users.
# They are generally quite helpful.
# We let the underlying libraries handle most of the formatting decisions.

# %%
str(basic_array)

# %% [markdown]
# The value displayed for the attributes of the object
# are simply the string representations of themselves.

# %%
str(basic_array.values)

# %% [markdown]
# With a large array,
# this leads to the slightly odd behaviour of showing all the values,
# as shown below.

# %%
str(large_array)

# %% [markdown]
# For whatever reason, this is the behaviour of the underlying packages.

# %%
str(large_array.values)

# %% [markdown]
# If we go to an even larger array, not all values are shown
# (which seems a more sensible choice to us).

# %%
str(really_large_array)

# %%
str(really_large_array.values)

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
repr(basic_array)

# %%
# pint's output is not copy-pasteable because of the surrounding "<>"
# and lack of commas between numerical values.
repr(basic_array.values)

# %%
# numpy does give copy-pasteable output for basic arrays
repr(basic_array.values.m)

# %%
repr(large_array)

# %%
repr(really_large_array)

# %%
# numpy doesn't give copy-pasteable output for really large arrays
repr(really_large_array.values.m)
