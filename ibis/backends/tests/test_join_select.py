"""Window functions and reductions in projections of joins.

`Join.select` follows the same rules as `Table.select`:

- analytic functions and reductions over the joined tables become window
  functions over the join result, so the output has one row per joined row
- reductions over tables outside the join become scalar subqueries, which
  are correlated when they reference a joined table

A window function is evaluated over the join it was selected from, even if
more tables are joined afterwards.
"""

from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis.backends.tests.errors import (
    ExaQueryError,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    PsycoPg2InternalError,
    Py4JJavaError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    SingleStoreDBOperationalError,
)

pytest.importorskip("pandas")

pytestmark = pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="window functions and scalar subqueries aren't implemented",
)

correlated_subquery_marks = [
    pytest.mark.notyet(
        ["mysql"],
        raises=MySQLOperationalError,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["exasol"],
        raises=ExaQueryError,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["impala"],
        raises=ImpalaHiveServer2Error,
        reason="the subquery can't reference the outer query from a derived table",
    ),
    pytest.mark.notyet(
        ["druid"],
        raises=PyDruidProgrammingError,
        reason="correlated subqueries aren't supported",
    ),
    pytest.mark.notyet(
        ["flink"], reason="unexpected correlate variable in the query plan"
    ),
]

row_number_frame_marks = [
    pytest.mark.notyet(
        ["risingwave"],
        raises=PsycoPg2InternalError,
        reason="ROW_NUMBER doesn't allow a frame clause",
    ),
    pytest.mark.notyet(
        ["druid"],
        raises=PyDruidProgrammingError,
        reason="ROW_NUMBER doesn't allow a frame clause",
    ),
]

# customers 2 and 3 have orders, so joining customers and orders gives two rows:
# (customer_id=2, balance=20, quantity=1) and (customer_id=3, balance=30, quantity=2)
customers = ibis.memtable({"customer_id": [1, 2, 3], "balance": [10, 20, 30]})
orders = ibis.memtable({"customer_id": [2, 3, 4], "quantity": [1, 2, 3]})
refunds = ibis.memtable({"customer_id": [2, 3], "amount": [100, 200]})
vips = ibis.memtable({"customer_id": [3]})


def rows(con, expr):
    """Execute `expr` and return its rows sorted, as tuples."""
    df = con.execute(expr)
    df = df.astype(object).where(df.notna(), None)
    return sorted(df.itertuples(index=False, name=None))


@pytest.mark.notyet(
    ["mssql"],
    raises=PyODBCProgrammingError,
    reason="ROW_NUMBER requires an ORDER BY",
)
@pytest.mark.notyet(
    ["oracle"],
    raises=com.UnsupportedOperationError,
    reason="ROW_NUMBER requires an ORDER BY",
)
@pytest.mark.notyet(["flink"], raises=Py4JJavaError)
@pytest.mark.notyet(
    ["impala"],
    raises=AssertionError,
    reason="the rows are numbered before the join is applied",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="ROW_NUMBER doesn't allow a frame clause",
)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="ROW_NUMBER doesn't allow a frame clause",
)
def test_join_select_row_number(con):
    # row_number() numbers the rows of the join, not the rows of `customers`,
    # so the two joined rows get 0 and 1
    expr = customers.join(orders, "customer_id").select(
        "customer_id", rn=ibis.row_number()
    )
    # the order is unspecified, so only the numbers are checked
    assert sorted(rn for _, rn in rows(con, expr)) == [0, 1]


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            # with an explicit window, the joined rows are numbered in
            # customer_id order
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", rn=ibis.row_number().over(order_by=customers.customer_id)
            ),
            [(2, 0), (3, 1)],
            id="row_number_explicit_over",
            marks=row_number_frame_marks,
        ),
        param(
            # the sum is a window over the join, so customer 1, who has no orders,
            # is left out of the total (20 + 30), and every joined row keeps its
            # own customer_id
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", total_balance=customers.balance.sum()
            ),
            [(2, 50), (3, 50)],
            id="reduction_with_column",
        ),
        param(
            # a reduction alone is still a window over the join, so there is one
            # row per joined row; use `.aggregate()` to get a single row
            #
            # This differs from SQL, where `SELECT SUM(c.balance) FROM c JOIN o`
            # returns a single row. Someone coming from SQL will expect one row
            # of 50 and get one row of 50 per joined row instead.
            #
            # Options:
            # - window over the join (what this test expects): the same as
            #   `Table.select`, where `t.select(s=t.x.sum())` also returns one
            #   row per row of `t`, so there is one rule for every projection.
            #   Differs from SQL.
            # - a single aggregated row: matches SQL, but makes `Join.select`
            #   behave differently from `Table.select`, and the result changes
            #   from one row to many as soon as a non-reduction column is added
            #   to the selection.
            #
            # Recommendation: keep the window, so `Join.select` and
            # `Table.select` follow the same rule, call it out as a breaking
            # change in the release notes, and point to `.aggregate()` in the
            # `select` docstring for a single row.
            lambda: customers.join(orders, "customer_id").select(
                total_balance=customers.balance.sum()
            ),
            [(50,), (50,)],
            id="reduction_only",
        ),
        param(
            # a reduction can combine columns of both joined tables:
            # 20 * 1 + 30 * 2
            lambda: customers.join(orders, "customer_id").select(
                "customer_id",
                weighted_balance=(customers.balance * orders.quantity).sum(),
            ),
            [(2, 80), (3, 80)],
            id="reduction_over_both_sides",
        ),
        param(
            # an explicit `.over()` gives the same result as the implicit window
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", total_balance=customers.balance.sum().over()
            ),
            [(2, 50), (3, 50)],
            id="reduction_explicit_over",
        ),
        param(
            # `refunds` isn't part of the join, so its sum is a scalar subquery
            # over all of `refunds` (100 + 200), repeated on every joined row
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", total_refunds=refunds.amount.sum()
            ),
            [(2, 300), (3, 300)],
            id="uncorrelated_subquery",
        ),
        param(
            # the subquery references the joined `customers`, so it sums each
            # joined customer's own refunds
            lambda: customers.join(orders, "customer_id").select(
                "customer_id",
                refunded=refunds.filter(
                    refunds.customer_id == customers.customer_id
                ).amount.sum(),
            ),
            [(2, 100), (3, 200)],
            id="correlated_subquery",
            marks=correlated_subquery_marks,
        ),
        param(
            # `orders` only filters the rows and isn't part of the output, so its
            # sum is a scalar subquery over all of `orders` (1 + 2 + 3)
            #
            # In SQL the right side of a semi join (`WHERE EXISTS (...)`) isn't
            # in scope in the select list, so this has no SQL equivalent. In
            # ibis, selecting `orders.quantity` without the sum raises
            # `IntegrityError`, but `orders.quantity.sum()` is accepted.
            #
            # Options:
            # - scalar subquery over all of `orders`, giving 6 (what this test
            #   expects): the same treatment as any table outside the join,
            #   like `refunds` above. But `orders` is part of the query, so a
            #   reader may expect the sum over the matched orders (1 + 2 = 3),
            #   and the reduction is accepted while the plain column is
            #   rejected.
            # - sum over the matched orders, giving 3: matches what a reader of
            #   the expression likely means, but a semi join doesn't produce
            #   the matched right rows, so this needs rewriting into an inner
            #   join and aggregation, and a customer with several matching
            #   orders makes it unclear which rows are summed.
            # - raise `IntegrityError`, the same as for `orders.quantity`:
            #   matches SQL scoping and the non-reduction case, and both
            #   intents remain expressible explicitly:
            #   `orders.quantity.sum().as_scalar()` for all orders, or an inner
            #   join followed by `.aggregate()` for the matched orders.
            #
            # Recommendation: raise, and expect the error here instead of 6.
            lambda: customers.semi_join(orders, "customer_id").select(
                "customer_id", total_quantity=orders.quantity.sum()
            ),
            [(2, 6), (3, 6)],
            id="semi_join_reduction_over_right",
            marks=[
                pytest.mark.notyet(
                    ["druid"],
                    raises=PyDruidProgrammingError,
                    reason="EXISTS isn't supported",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="semi joins aren't supported",
                ),
            ],
        ),
        param(
            # a correlated subquery containing an uncorrelated subquery over
            # `customers`: the inner `customers.filter(...)` is the whole
            # `customers` table (3 rows), so the threshold is 3 * 40 = 120,
            # and only customer 2's refund (100) is below it
            lambda: customers.join(orders, "customer_id").select(
                "customer_id",
                refunded=refunds.filter(
                    refunds.customer_id == customers.customer_id,
                    refunds.amount
                    < customers.filter(customers.balance > 0).balance.count() * 40,
                ).amount.sum(),
            ),
            [(2, 100), (3, None)],
            id="correlated_subquery_containing_subquery",
            marks=pytest.mark.xfail(
                reason=(
                    "the inner subquery references an alias missing from the "
                    "compiled query"
                ),
            ),
        ),
    ],
)
def test_join_select(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            # mutate keeps every joined column and adds the sum over the join
            lambda: customers.join(orders, "customer_id").mutate(
                total_balance=customers.balance.sum()
            ),
            [(2, 20, 1, 50), (3, 30, 2, 50)],
            id="reduction",
        ),
        param(
            # the subquery references the joined `customers`, so it sums each
            # joined customer's own refunds
            lambda: customers.join(orders, "customer_id").mutate(
                refunded=refunds.filter(
                    refunds.customer_id == customers.customer_id
                ).amount.sum()
            ),
            [(2, 20, 1, 100), (3, 30, 2, 200)],
            id="correlated_subquery",
            marks=correlated_subquery_marks,
        ),
    ],
)
def test_join_mutate(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected


impala_window_then_join = pytest.mark.notyet(
    ["impala"],
    raises=AssertionError,
    reason="the window function is evaluated after the join",
)


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            # the sum is computed over customers joined with orders (20 + 30),
            # then the result is joined with vips; joining vips first would
            # give 30
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", total_balance=customers.balance.sum()
            ),
            [(3, 50)],
            id="select_reduction",
            marks=impala_window_then_join,
        ),
        param(
            # the same as above, with an explicit `.over()`
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", total_balance=customers.balance.sum().over()
            ),
            [(3, 50)],
            id="select_reduction_explicit_over",
            marks=impala_window_then_join,
        ),
        param(
            # the rows are numbered before joining vips, so customer 3 keeps
            # number 1 instead of being renumbered to 0
            lambda: customers.join(orders, "customer_id").select(
                "customer_id", rn=ibis.row_number().over(order_by=customers.customer_id)
            ),
            [(3, 1)],
            id="select_row_number_explicit_over",
            marks=row_number_frame_marks,
        ),
        param(
            # a window function added by mutate is also computed before joining
            # vips
            lambda: (
                customers.join(orders, "customer_id")
                .mutate(total_balance=customers.balance.sum())
                .select("customer_id", "total_balance")
            ),
            [(3, 50)],
            id="mutate_reduction",
            marks=impala_window_then_join,
        ),
    ],
)
def test_window_then_join(con, make_expr, expected):
    # window functions are evaluated over the join they were selected from,
    # before the later join with vips
    expr = make_expr().join(vips, "customer_id")
    assert rows(con, expr) == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            # the subquery sums each joined customer's refunds, and only customers
            # whose balance is below their refunds are kept (20 < 100, 30 < 200)
            lambda: customers.join(orders, "customer_id").filter(
                customers.balance
                < refunds.filter(
                    refunds.customer_id == customers.customer_id
                ).amount.sum()
            ),
            [(2, 20, 1), (3, 30, 2)],
            id="join_filter",
            marks=[
                *correlated_subquery_marks,
                pytest.mark.notyet(
                    ["singlestoredb"],
                    raises=SingleStoreDBOperationalError,
                    reason=(
                        "the subquery can't reference the outer query "
                        "from a derived table"
                    ),
                ),
                pytest.mark.notyet(
                    ["datafusion"],
                    raises=Exception,
                    reason="the physical optimizer fails with a schema mismatch",
                ),
            ],
        ),
        param(
            # the subquery references `customers`, the table under the filter,
            # and is correlated with each filtered row; customer 1 has no
            # refunds, so the sum is NULL
            lambda: customers.filter(customers.balance > 0).select(
                "customer_id",
                refunded=refunds.filter(
                    refunds.customer_id == customers.customer_id
                ).amount.sum(),
            ),
            [(1, None), (2, 100), (3, 200)],
            id="filter_select",
            marks=correlated_subquery_marks,
        ),
    ],
)
def test_correlated_subquery(con, make_expr, expected):
    expr = make_expr()
    assert rows(con, expr) == expected
