# Code Review: feat(mysql): switch to adbc (#11958)

## Overall Assessment

This is a well-motivated PR that aligns the MySQL backend with the project's broader ADBC direction.
The approach is sound and the performance/licensing/installation wins are real. However, there are
several correctness and robustness issues that should be addressed before merging.

---

## Issues

### 1. Fragile error detection in `get_schema()` (correctness risk)

```python
# new code
except Exception as e:
    if "doesn't exist" in str(e):
        raise com.TableNotFound(name) from e
    raise
```

The old code checked `e.args[0] == ER.NO_SUCH_TABLE` — a precise MySQL error code. The new code
matches on the string `"doesn't exist"` in the exception message. This is brittle: any exception
whose message happens to contain that substring (a network error, a driver message, etc.) will be
misclassified as `TableNotFound`. I'd strongly prefer a check against a specific ADBC error code
or exception type if one is available.

### 2. `from_connection()` loses `@util.experimental` and its type annotation

```python
# old
@util.experimental
@classmethod
def from_connection(cls, con: MySQLdb.Connection, /) -> Backend:

# new
@classmethod
def from_connection(cls, con, /, **kwargs):
```

`@util.experimental` was intentional — it told users this API could change. Dropping it promotes
`from_connection` to stable public API. The `**kwargs` is accepted but silently ignored (nothing
is done with it inside the method). Either document what kwargs are valid, use them, or remove
the parameter.

### 3. `autocommit` parameter silently dropped (breaking change)

The old `do_connect` accepted `autocommit: bool = True`. The new version removes it. Any user
who called `ibis.mysql.connect(autocommit=False)` will now have that keyword argument silently
ignored (it flows into `**kwargs`, which is only used to extract `database`/`db`, so `autocommit`
is just lost). Since ADBC always uses autocommit, consider at least issuing a deprecation warning
if `autocommit` is found in kwargs, or documenting the behavior change explicitly.

### 4. Password embedded in a URI string

```python
uri = f"{user}:{password}@tcp({host}:{port})/{database or ''}"
```

URI-style credential embedding means the password can show up in tracebacks, log messages, and
`repr()` output of connection objects. The old driver accepted `password` as a separate keyword
argument, keeping it out of string-formatted values. Verify that the ADBC driver manager doesn't
expose this URI in any error messages or debug output.

### 5. `_decode_opaque_storage()` iterates element-by-element in Python (performance)

```python
for val in storage:
    raw = val.as_py()
    ...
decoded.append(...)
return pa.array(decoded, type=pa.string())
```

This converts an Arrow array to Python objects one element at a time. For large result sets this
will be slow. Consider using `storage.to_pylist()` to do the Python conversion in one call, then
process the list:

```python
raws = storage.to_pylist()
decoded = [
    bytes(int(x) for x in raw[1:-1].split()).decode("ascii") if raw and raw.startswith("[") else raw
    for raw in raws
]
return pa.array(decoded, type=pa.string())
```

### 6. `_cast_adbc_column()` silently swallows conversion failures

```python
except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
    try:
        return col.cast(pa.float64()).cast(target_type)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        # If that also fails ... leave as-is
        return col
```

Silently returning the original column when casting fails means callers receive data with the
wrong type, which can cause downstream errors far from the source. At minimum this should emit a
`warnings.warn(...)` so users know type coercion failed.

### 7. `begin()` is misleadingly named after the change

```python
@contextlib.contextmanager
def begin(self):
    cur = self.con.cursor()
    try:
        yield cur
    finally:
        cur.close()
```

`begin()` used to start a transaction. Now it just opens/closes a cursor. It's still called from
`create_table()`. The misleading name will confuse future contributors. Rename to
`_cursor()` or similar, or add a comment explaining why it's kept but no longer manages
transactions.

### 8. Inline imports in `execute()`

```python
def execute(self, ...):
    ...
    import pandas as pd
    from ibis.formats.pandas import PandasData
```

These imports appear mid-method. Other methods in this file that need pyarrow import at the top
of their function body consistently. Either move these to the top of the file (preferred) or at
least to the top of the method for readability.

### 9. `_cast_adbc_table()` has no column count validation

```python
columns = [
    cls._cast_adbc_column(table.column(i), field.type)
    for i, field in enumerate(target_schema)
]
```

If the ADBC driver returns a table with a different number of columns than `target_schema`,
`table.column(i)` will raise an `IndexError` with a confusing message, or silently skip columns.
An explicit check (`assert table.num_columns == len(target_schema)`) would produce a better error.

### 10. CI: `dbc` installed without version pinning

```yaml
pipx install dbc
dbc install mysql
```

Unpinned installs in CI mean the driver version can change between runs, causing flaky failures
or silent behaviour changes. Pin `dbc` to a minimum version (`pipx install 'dbc>=X.Y'`) and
consider also pinning the MySQL driver version.

---

## Minor Nits

### `test_get_schema_from_query_special_cases` should be renamed
(NickCrews already flagged this in the inline comments) — now that both old expected-type
arguments collapse to one, the "special cases" framing no longer applies. Rename to
`test_get_schema_from_query` or similar.

### `_get_schema_using_query()` — noqa suppresses the SQL injection warning
```python
create_sql = f"CREATE TEMPORARY TABLE {quoted_tmp} AS SELECT * FROM ({query}) AS _t LIMIT 0"  # noqa: S608
```
The `query` argument is documented as internal/generated, so the risk is low. A brief comment
explaining *why* it's safe to suppress here would help future readers.

### `batch_size` workaround should reference the fix version
```python
# this is already fixed in mysql driver. we can drop batch_size calculation here
# once a new version of the driver is out.
```
Add the specific version number once the upstream release ships, so future maintainers know
when this can be removed.

---

## Positive Changes Worth Calling Out

- Removing `datatypes.py` (150 lines of custom type-code wrangling) is a real simplification.
- Null-typed columns are now coerced to `dt.string` rather than raising an error — a UX improvement.
- `disconnect()` method is a good addition.
- The `_decode_opaque_storage()` function is well-documented given its hack-y nature.
- `_get_schema_using_query()` is cleaner using `DESCRIBE` rather than parsing cursor metadata flags.
