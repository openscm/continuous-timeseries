# Pandas accessors

Continuous timeseries also provides a [`pandas`][pandas] accessor.
For details of the implementation of this pattern, see
[pandas' docs](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors).

The accessors must be registered before they can be used
(we do this to avoid imports of any of our modules having side effects,
which is a pattern we have had bad experiences with in the past).
This is done with
[`register_pandas_accessor`][continuous_timeseries.pandas_accessors.register_pandas_accessor],

By default, the accessors are provided under the "ct" namespace
and this is how the accessors are documented below.
However, the namespace can be customised when using
[`register_pandas_accessor`][continuous_timeseries.pandas_accessors.register_pandas_accessor],
should you wish to use a different namespace for the accessor.

For the avoidance of doubt, in order to register/activate the accessors,
you will need to run something like:

```python
from continuous_timeseries.pandas_accessors import register_pandas_accessor

# The 'pd.Series.ct' namespace will not be available at this point.

# Register the accessors
register_pandas_accessor()

# The 'pd.Series.ct' namespace
# (or whatever other custom namespace you chose to register)
# will now be available.
```

The full accessor API is documented below.

::: continuous_timeseries.pandas_accessors.SeriesCTAccessor
    handler: python_accessors
    options:
        namespace: "pd.Series.ct"
        show_root_full_path: false
        show_root_heading: true
