import ibis


alltypes = ibis.table(
    name="alltypes",
    schema={
        "a": "int8",
        "b": "int16",
        "c": "int32",
        "d": "int64",
        "e": "float32",
        "f": "float64",
        "g": "string",
        "h": "boolean",
        "i": "timestamp",
        "j": "date",
        "k": "time",
    },
)
lit = ibis.literal("bar")

result = alltypes.select(
    alltypes.g.cases(
        (ibis.literal("foo"), lit),
        (ibis.literal("baz"), ibis.literal("qux")),
        else_=ibis.literal("default"),
    ).name("col1"),
    ibis.cases(
        ((alltypes.g == "foo"), lit),
        ((alltypes.g == "baz"), alltypes.g),
        else_=ibis.literal(None),
    ).name("col2"),
    alltypes.a,
    alltypes.b,
    alltypes.c,
    alltypes.d,
    alltypes.e,
    alltypes.f,
    alltypes.g,
    alltypes.h,
    alltypes.i,
    alltypes.j,
    alltypes.k,
)
