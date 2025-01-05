# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## Continuous Timeseries v0.2.1 (2025-01-05)

### 🆕 Features

- Added [`budget_compatible_pathways`][continuous_timeseries.budget_compatible_pathways] to support the creation of pathways compatible with a given budget. ([#18](https://github.com/openscm/continuous-timeseries/pull/18))

### 📚 Improved Documentation

- Added a tutorial into our support for creating emissions pathways that are compatible with a given budget (see [Budget-compatible emissions pathways](../tutorials/budget_compatible_pathways)). ([#18](https://github.com/openscm/continuous-timeseries/pull/18))

### 🔧 Trivial/Internal Changes

- [#17](https://github.com/openscm/continuous-timeseries/pull/17), [#18](https://github.com/openscm/continuous-timeseries/pull/18)


## Continuous Timeseries v0.2.0 (2025-01-04)

### 🆕 Features

- Added [`ValuesAtBounds`][continuous_timeseries.values_at_bounds.ValuesAtBounds]. ([#12](https://github.com/openscm/continuous-timeseries/pull/12))
- Added [`TimeAxis`][continuous_timeseries.time_axis.TimeAxis]. ([#13](https://github.com/openscm/continuous-timeseries/pull/13))
- Added [`TimeseriesDiscrete`][continuous_timeseries.timeseries_discrete.TimeseriesDiscrete]. ([#14](https://github.com/openscm/continuous-timeseries/pull/14))
- Added [`TimeseriesContinuous`][continuous_timeseries.timeseries_continuous.TimeseriesContinuous]. ([#15](https://github.com/openscm/continuous-timeseries/pull/15))
- Added [`Timeseries`][continuous_timeseries.Timeseries]. ([#16](https://github.com/openscm/continuous-timeseries/pull/16))

### 📚 Improved Documentation

- Added background about how we handle representing our objects (see [Representations][representations]) and updated the default colour scheme. ([#12](https://github.com/openscm/continuous-timeseries/pull/12))
- Added a tutorial into our discrete timeseries handling (see [Discrete timeseries](../tutorials/discrete_timeseries_tutorial)). ([#14](https://github.com/openscm/continuous-timeseries/pull/14))
- Added a tutorial into our continuous timeseries handling (see [Continuous timeseries](../tutorials/continuous_timeseries_tutorial)). ([#15](https://github.com/openscm/continuous-timeseries/pull/15))
- - Added background into why we built this API (see [Why this API?](../further-background/why-this-api)).
  - Added a tutorial into our timeseries handling (see [Timeseries](../tutorials/timeseries_tutorial)).
  - Added a tutorial into higher-order interpolation (see [Higher-order interpolation](../tutorials/higher_order_interpolation)).
  - Added a how-to guide about how to make sharp, step forcings (see [How-to make a step forcing](../how-to-guides/how-to-make-a-step-forcing)).

  ([#16](https://github.com/openscm/continuous-timeseries/pull/16))

### 🔧 Trivial/Internal Changes

- [#11](https://github.com/openscm/continuous-timeseries/pull/11), [#12](https://github.com/openscm/continuous-timeseries/pull/12), [#15](https://github.com/openscm/continuous-timeseries/pull/15), [#16](https://github.com/openscm/continuous-timeseries/pull/16)


## Continuous Timeseries v0.1.7 (2024-12-27)

### 🔧 Trivial/Internal Changes

- [#9](https://github.com/openscm/continuous-timeseries/pull/9)


## Continuous Timeseries v0.1.6 (2024-12-27)

### 🔧 Trivial/Internal Changes

- [#8](https://github.com/openscm/continuous-timeseries/pull/8)


## Continuous Timeseries v0.1.5 (2024-12-26)

### 🔧 Trivial/Internal Changes

- [#7](https://github.com/openscm/continuous-timeseries/pull/7)


## Continuous Timeseries v0.1.4 (2024-12-26)

### 🔧 Trivial/Internal Changes

- [#4](https://github.com/openscm/continuous-timeseries/pull/4)


## Continuous Timeseries v0.1.3 (2024-12-26)

### 🔧 Trivial/Internal Changes

- [#6](https://github.com/openscm/continuous-timeseries/pull/6)


## Continuous Timeseries v0.1.2 (2024-12-26)

### 🔧 Trivial/Internal Changes

- [#3](https://github.com/openscm/continuous-timeseries/pull/3), [#5](https://github.com/openscm/continuous-timeseries/pull/5)


## Continuous Timeseries v0.1.1 (2024-12-21)

### 🔧 Trivial/Internal Changes

- [#2](https://github.com/openscm/continuous-timeseries/pull/2)


## Continuous Timeseries v0.1.0 (2024-12-21)

### 🔧 Trivial/Internal Changes

- [#1](https://github.com/openscm/continuous-timeseries/pull/1)
