"""
Unit tests of `continuous_timeseries.domain_helpers`
"""

from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import pint
import pint.testing
import pytest

from continuous_timeseries.domain_helpers import check_no_times_outside_domain

UR = pint.get_application_registry()
Q = UR.Quantity


@pytest.mark.parametrize(
    "times, domain, expectation",
    (
        pytest.param(
            Q(10, "yr"),
            (Q(1, "yr"), Q(100, "yr")),
            does_not_raise(),
            id="scalar_in_domain",
        ),
        pytest.param(
            Q(0, "yr"),
            (Q(1, "yr"), Q(100, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(1, 'year')>, <Quantity(100, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([0], 'year')>."
                ),
            ),
            id="scalar_pre_domain",
        ),
        pytest.param(
            Q(110, "yr"),
            (Q(1, "yr"), Q(100, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(1, 'year')>, <Quantity(100, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([110], 'year')>."
                ),
            ),
            id="scalar_post_domain",
        ),
        pytest.param(
            Q([10, 20], "yr"),
            (Q(1, "yr"), Q(100, "yr")),
            does_not_raise(),
            id="all_in_domain",
        ),
        pytest.param(
            Q([10, 20], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            does_not_raise(),
            id="all_in_domain_edge",
        ),
        pytest.param(
            Q([10 - 1e-5, 20], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(10, 'year')>, <Quantity(20, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([9.99999], 'year')>."
                ),
            ),
            id="one_before_domain_edge",
        ),
        pytest.param(
            Q([10, 10 - 1e-5, 20], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(10, 'year')>, <Quantity(20, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([9.99999], 'year')>."
                ),
            ),
            id="one_before_domain_edge_out_of_order",
        ),
        pytest.param(
            Q([10, 20.001], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(10, 'year')>, <Quantity(20, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([20.001], 'year')>."
                ),
            ),
            id="one_after_domain_edge",
        ),
        pytest.param(
            Q([10, 20 + 1e-5, 20], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(10, 'year')>, <Quantity(20, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([20.00001], 'year')>."
                ),
            ),
            id="one_before_domain_edge_out_of_order",
        ),
        pytest.param(
            Q([5.0, 10.0, 15.0, 25.0, 20.0, 30.0], "yr"),
            (Q(10, "yr"), Q(20, "yr")),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "The domain=(<Quantity(10, 'year')>, <Quantity(20, 'year')>). "
                    "There are time values that are outside this domain: "
                    "outside_domain=<Quantity([ 5. 25. 30.], 'year')>."
                ),
            ),
            id="multiple_outside_domain_out_of_order",
        ),
    ),
)
def test_validation(times, domain, expectation):
    with expectation:
        check_no_times_outside_domain(times, domain)


def test_domain_wrong_length():
    with pytest.raises(AssertionError):
        check_no_times_outside_domain(
            ["not", "used"], (Q(1, "yr"), Q(2, "yr"), Q(3.0, "yr"))
        )


def test_domain_out_of_order():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "domain[1] must be greater than domain[0]. "
            "Received domain=(<Quantity(10.0, 'year')>, <Quantity(9.0, 'year')>)."
        ),
    ):
        check_no_times_outside_domain(["not", "used"], (Q(10.0, "yr"), Q(9.0, "yr")))
