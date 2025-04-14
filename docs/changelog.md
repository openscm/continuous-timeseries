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

## Continuous Timeseries v0.4.3 (2025-04-14)

### 🆕 Features

- Added anti-differentiation (indefinite integral), specifically [continuous_timeseries.timeseries.Timeseries.antidifferentiate][] and [continuous_timeseries.timeseries_continuous.TimeseriesContinuous.antidifferentiate][] ([#36](https://github.com/openscm/continuous-timeseries/pull/36))

### 🔧 Trivial/Internal Changes

- [#37](https://github.com/openscm/continuous-timeseries/pull/37)


## Continuous Timeseries v0.4.2 (2025-01-21)

### 🐛 Bug Fixes

- One more attempt to fix the locked installation on windows with python 3.13. ([#35](https://github.com/openscm/continuous-timeseries/pull/35))

### 🔧 Trivial/Internal Changes

- [#34](https://github.com/openscm/continuous-timeseries/pull/34)


## Continuous Timeseries v0.4.1 (2025-01-21)

### 🐛 Bug Fixes

- Fixed up dependency pinning to allow the locked version to install on windows with Python 3.13 (failing test run: https://github.com/openscm/continuous-timeseries/actions/runs/12875459495/job/35897080123). ([#33](https://github.com/openscm/continuous-timeseries/pull/33))

### 🔧 Trivial/Internal Changes

- [#31](https://github.com/openscm/continuous-timeseries/pull/31)


## Continuous Timeseries v0.4.0 (2025-01-20)

### 🆕 Features

- Added the skeleton for a pandas accessor ([#26](https://github.com/openscm/continuous-timeseries/pull/26))

### 🐛 Bug Fixes

- Fixed the minimum versions of our requirements (also tested that installation with minimum versions works). ([#29](https://github.com/openscm/continuous-timeseries/pull/29))

### 📚 Improved Documentation

- Added documentation on our dependency pinning and testing strategy. ([#29](https://github.com/openscm/continuous-timeseries/pull/29))

### 🔧 Trivial/Internal Changes

- [#27](https://github.com/openscm/continuous-timeseries/pull/27), [#29](https://github.com/openscm/continuous-timeseries/pull/29), [#30](https://github.com/openscm/continuous-timeseries/pull/30)


## Continuous Timeseries v0.3.3 (2025-01-08)

### 🔧 Trivial/Internal Changes

- [#24](https://github.com/openscm/continuous-timeseries/pull/24)


## Continuous Timeseries v0.3.2 (2025-01-08)

### 🔧 Trivial/Internal Changes

- [#23](https://github.com/openscm/continuous-timeseries/pull/23)


## Continuous Timeseries v0.3.1 (2025-01-08)

### 📚 Improved Documentation

- Updated docs demonstrating how to use a cubic fit to find a budget-compatible pathway. ([#22](https://github.com/openscm/continuous-timeseries/pull/22))

### 🔧 Trivial/Internal Changes

- [#20](https://github.com/openscm/continuous-timeseries/pull/20), [#21](https://github.com/openscm/continuous-timeseries/pull/21)


## Continuous Timeseries v0.3.0 (2025-01-06)

### ⚠️ Breaking Changes

- - Changed the input arguments of [`discrete_to_continuous`][continuous_timeseries.discrete_to_continuous.discrete_to_continuous].
    We have updated so that, rather than taking in a `discrete` argument,
    we take in an `x` and a `y` array and a `name`.
    This API better represents the separation between
    discrete representations, continuous representations
    and the conversion in between them (which is a different thing again).
    All other discrete to continuous conversion functions were updated to match this change in API.
  - Changed the input arguments of [`Timeseries.from_arrays`][continuous_timeseries.Timeseries.from_arrays].
    We have updated so that `time_axis_bounds` is now `x` and `values_at_bounds` is now `y`.
    This update reflects the fact that, depending on the interpolation choice,
    the passed in values will not always end up being the values at the bounds.

  ([#19](https://github.com/openscm/continuous-timeseries/pull/19))

### 🎉 Improvements

- Added a check to [`TimeseriesDiscrete.to_continuous_timeseries`][continuous_timeseries.TimeseriesDiscrete.to_continuous_timeseries]
  so that the user is aware if the chosen interpolation choice means that the instance's
  values at bounds are not actually respected.
  The warning can be controlled with the new `warn_if_output_values_at_bounds_could_confuse` and `check_change_func` arguments
  to [`TimeseriesDiscrete.to_continuous_timeseries`][continuous_timeseries.TimeseriesDiscrete.to_continuous_timeseries]. ([#19](https://github.com/openscm/continuous-timeseries/pull/19))

### 📚 Improved Documentation

- Added further background into our discrete to continuous conversion (see [Discrete to continuous conversions](../further-background/discrete_to_continuous_conversions)). ([#19](https://github.com/openscm/continuous-timeseries/pull/19))


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
