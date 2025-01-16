# Pandas accessors

By default, these are added under the accessor "ct".
This is how the accessors are shown below.
However, the namespace is modifiable with
[`register_pandas_accessor`][continuous_timeseries.pandas_accessors.register_pandas_accessor],
should you wish to use a different accessor.

::: continuous_timeseries.pandas_accessors.DataFrameCTAccessor
    handler: python_accessors
    options:
        namespace: "pd.DataFrame.ct"
        show_root_full_path: false
        show_root_heading: true
