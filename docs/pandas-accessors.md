# Pandas accessors

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

::: continuous_timeseries.pandas_accessors.DataFrameCTAccessor
    handler: python_accessors
    options:
        namespace: "pd.DataFrame.ct"
        show_root_full_path: false
        show_root_heading: true
