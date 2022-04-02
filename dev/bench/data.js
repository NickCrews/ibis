window.BENCHMARK_DATA = {
  "lastUpdate": 1648919170166,
  "repoUrl": "https://github.com/ibis-project/ibis",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "2202e665eedb4809439264560a3852ea81820c25",
          "message": "chore: remove unused patch",
          "timestamp": "2022-04-01T12:39:31-04:00",
          "tree_id": "0bb2a82050dad4583310981b5e7dce21a28cf5a4",
          "url": "https://github.com/ibis-project/ibis/commit/2202e665eedb4809439264560a3852ea81820c25"
        },
        "date": 1648831758770,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1537.5595505900733,
            "unit": "iter/sec",
            "range": "stddev: 0.00003172364789063014",
            "extra": "mean: 650.3813134367558 usec\nrounds: 67"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 182.87939573196772,
            "unit": "iter/sec",
            "range": "stddev: 0.00017311134782081289",
            "extra": "mean: 5.468084559212035 msec\nrounds: 152"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 73.30229193932696,
            "unit": "iter/sec",
            "range": "stddev: 0.0002657822475548168",
            "extra": "mean: 13.642138240748459 msec\nrounds: 54"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 368761.03130134864,
            "unit": "iter/sec",
            "range": "stddev: 4.869226526134625e-7",
            "extra": "mean: 2.7117832827157042 usec\nrounds: 646"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 122244.86389085805,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011519204591836826",
            "extra": "mean: 8.180302780596282 usec\nrounds: 36898"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 3338372.3548080493,
            "unit": "iter/sec",
            "range": "stddev: 6.421825020332677e-8",
            "extra": "mean: 299.54717260935934 nsec\nrounds: 212"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 89.51091326496996,
            "unit": "iter/sec",
            "range": "stddev: 0.0006729990936911464",
            "extra": "mean: 11.171822111118482 msec\nrounds: 36"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 19693.16809822773,
            "unit": "iter/sec",
            "range": "stddev: 0.000015336970195148673",
            "extra": "mean: 50.77903133777618 usec\nrounds: 6382"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 484.95861916269894,
            "unit": "iter/sec",
            "range": "stddev: 0.00007408520823481697",
            "extra": "mean: 2.062031605349218 msec\nrounds: 299"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 62.3677102905233,
            "unit": "iter/sec",
            "range": "stddev: 0.0009570676746097203",
            "extra": "mean: 16.033937999996592 msec\nrounds: 49"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 813016.6838730662,
            "unit": "iter/sec",
            "range": "stddev: 4.005996582541448e-7",
            "extra": "mean: 1.2299870591045914 usec\nrounds: 53474"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 39.65085743971033,
            "unit": "iter/sec",
            "range": "stddev: 0.0005709526835410805",
            "extra": "mean: 25.220135567572875 msec\nrounds: 37"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 10712.681412904438,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051565459893287135",
            "extra": "mean: 93.34731067381557 usec\nrounds: 3560"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 174.05073108019482,
            "unit": "iter/sec",
            "range": "stddev: 0.0001554759280239515",
            "extra": "mean: 5.745451304879865 msec\nrounds: 164"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 74.95211901643447,
            "unit": "iter/sec",
            "range": "stddev: 0.006590815139522477",
            "extra": "mean: 13.341850945945023 msec\nrounds: 74"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 389142.2535765277,
            "unit": "iter/sec",
            "range": "stddev: 6.144910877131246e-7",
            "extra": "mean: 2.56975435283422 usec\nrounds: 14244"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 26043.644040589486,
            "unit": "iter/sec",
            "range": "stddev: 0.000003109840024990715",
            "extra": "mean: 38.39708446488833 usec\nrounds: 2948"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.1340371709821613,
            "unit": "iter/sec",
            "range": "stddev: 0.03723425087003634",
            "extra": "mean: 7.460617026399996 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 93.18342553008353,
            "unit": "iter/sec",
            "range": "stddev: 0.00026194887151752825",
            "extra": "mean: 10.731522202702863 msec\nrounds: 74"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 751.6397050478358,
            "unit": "iter/sec",
            "range": "stddev: 0.00004786892525152679",
            "extra": "mean: 1.3304246612894912 msec\nrounds: 62"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1170304.5899643057,
            "unit": "iter/sec",
            "range": "stddev: 3.136447295816407e-7",
            "extra": "mean: 854.478405515354 nsec\nrounds: 185186"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 307.49283276292294,
            "unit": "iter/sec",
            "range": "stddev: 0.0001847135095221366",
            "extra": "mean: 3.252108320752309 msec\nrounds: 265"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 6194.380274572146,
            "unit": "iter/sec",
            "range": "stddev: 0.00005075623379053042",
            "extra": "mean: 161.43664994301164 usec\nrounds: 2631"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 40.71402580127961,
            "unit": "iter/sec",
            "range": "stddev: 0.0012664561318591599",
            "extra": "mean: 24.561560305553737 msec\nrounds: 36"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 164.4204639023999,
            "unit": "iter/sec",
            "range": "stddev: 0.00025836780117650737",
            "extra": "mean: 6.08196800000273 msec\nrounds: 159"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 7233676.102855893,
            "unit": "iter/sec",
            "range": "stddev: 1.4649568510599878e-8",
            "extra": "mean: 138.24229697059226 nsec\nrounds: 58477"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 62.051269278588045,
            "unit": "iter/sec",
            "range": "stddev: 0.000772929902042319",
            "extra": "mean: 16.115705796610815 msec\nrounds: 59"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 141.4437935748706,
            "unit": "iter/sec",
            "range": "stddev: 0.00018993875449271183",
            "extra": "mean: 7.069946122950025 msec\nrounds: 122"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 5702332.790703137,
            "unit": "iter/sec",
            "range": "stddev: 1.730965976586976e-8",
            "extra": "mean: 175.36682559647642 nsec\nrounds: 61725"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 11117.070850953181,
            "unit": "iter/sec",
            "range": "stddev: 0.000005793625127199813",
            "extra": "mean: 89.95175198638404 usec\nrounds: 6169"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 966534.9773401035,
            "unit": "iter/sec",
            "range": "stddev: 1.829586020647977e-7",
            "extra": "mean: 1.034623705758048 usec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 4245404.796450618,
            "unit": "iter/sec",
            "range": "stddev: 4.2466522837703864e-8",
            "extra": "mean: 235.5487987474899 nsec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 726.7955843202369,
            "unit": "iter/sec",
            "range": "stddev: 0.00003138922622149807",
            "extra": "mean: 1.3759026906241978 msec\nrounds: 640"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 35.42868087282817,
            "unit": "iter/sec",
            "range": "stddev: 0.0012402206926885387",
            "extra": "mean: 28.22571925806429 msec\nrounds: 31"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1339.7393190514833,
            "unit": "iter/sec",
            "range": "stddev: 0.00003709011533919282",
            "extra": "mean: 746.4138625923034 usec\nrounds: 131"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "a0f3bd96b2112454ba5431df4f5073de4a7954b0",
          "message": "fix(dask): workaround dask bug",
          "timestamp": "2022-04-01T17:56:29-04:00",
          "tree_id": "18eed2682f679e3884af7dd1a28a7885376388ea",
          "url": "https://github.com/ibis-project/ibis/commit/a0f3bd96b2112454ba5431df4f5073de4a7954b0"
        },
        "date": 1648850400728,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 89.46379364063576,
            "unit": "iter/sec",
            "range": "stddev: 0.005407691416905492",
            "extra": "mean: 11.177706190471511 msec\nrounds: 42"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 119269.36605768226,
            "unit": "iter/sec",
            "range": "stddev: 4.115662061712348e-7",
            "extra": "mean: 8.384382621069436 usec\nrounds: 33558"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 190.0430671108563,
            "unit": "iter/sec",
            "range": "stddev: 0.00010585096662361745",
            "extra": "mean: 5.261965170329933 msec\nrounds: 182"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 154.91791200228502,
            "unit": "iter/sec",
            "range": "stddev: 0.00003579645277598301",
            "extra": "mean: 6.455031487806589 msec\nrounds: 123"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 30323.50979365074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013317852317247953",
            "extra": "mean: 32.9777128968555 usec\nrounds: 1954"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 404527.42375284183,
            "unit": "iter/sec",
            "range": "stddev: 2.2306192050253585e-7",
            "extra": "mean: 2.472020291536477 usec\nrounds: 15129"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 789.2745108743194,
            "unit": "iter/sec",
            "range": "stddev: 0.000056313329744898054",
            "extra": "mean: 1.2669863098610004 msec\nrounds: 71"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 3973063.9656055057,
            "unit": "iter/sec",
            "range": "stddev: 6.55361526153057e-8",
            "extra": "mean: 251.69491572673365 nsec\nrounds: 236"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 11705.8872577245,
            "unit": "iter/sec",
            "range": "stddev: 0.000011169646552205159",
            "extra": "mean: 85.42709988430126 usec\nrounds: 7759"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1647.8270787654901,
            "unit": "iter/sec",
            "range": "stddev: 0.000005801203023332109",
            "extra": "mean: 606.8597930489007 usec\nrounds: 604"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 8771660.778305065,
            "unit": "iter/sec",
            "range": "stddev: 4.239459190633026e-9",
            "extra": "mean: 114.00349663238569 nsec\nrounds: 72464"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 1029880.1073803278,
            "unit": "iter/sec",
            "range": "stddev: 7.442862380816567e-8",
            "extra": "mean: 970.9868098561462 nsec\nrounds: 172414"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 208.50996761539943,
            "unit": "iter/sec",
            "range": "stddev: 0.00010009444500928006",
            "extra": "mean: 4.795933793652104 msec\nrounds: 189"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 87.36191454446887,
            "unit": "iter/sec",
            "range": "stddev: 0.00013897649762448762",
            "extra": "mean: 11.44663558730711 msec\nrounds: 63"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 70.47476296622297,
            "unit": "iter/sec",
            "range": "stddev: 0.0005759518568592025",
            "extra": "mean: 14.1894766000033 msec\nrounds: 55"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 11101.66077261526,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016532674226212192",
            "extra": "mean: 90.07661290342473 usec\nrounds: 806"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 21177.962258075408,
            "unit": "iter/sec",
            "range": "stddev: 0.000011824380734945125",
            "extra": "mean: 47.21889612484733 usec\nrounds: 6864"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 107.4064149494674,
            "unit": "iter/sec",
            "range": "stddev: 0.00034726787939390975",
            "extra": "mean: 9.310430857137167 msec\nrounds: 7"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 91.53517920691023,
            "unit": "iter/sec",
            "range": "stddev: 0.00005978664201086971",
            "extra": "mean: 10.924761481479763 msec\nrounds: 81"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 181.45373214709312,
            "unit": "iter/sec",
            "range": "stddev: 0.00018235356681256334",
            "extra": "mean: 5.511046745455546 msec\nrounds: 165"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 48.93043489720709,
            "unit": "iter/sec",
            "range": "stddev: 0.0009240286644717398",
            "extra": "mean: 20.437177844439702 msec\nrounds: 45"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1676727.2465607428,
            "unit": "iter/sec",
            "range": "stddev: 2.181890994198837e-8",
            "extra": "mean: 596.3999225582446 nsec\nrounds: 82645"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 5217018.986476977,
            "unit": "iter/sec",
            "range": "stddev: 5.822152802386277e-9",
            "extra": "mean: 191.68034515341816 nsec\nrounds: 51021"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 876196.7931409802,
            "unit": "iter/sec",
            "range": "stddev: 1.4433069452296108e-7",
            "extra": "mean: 1.1412961195797253 usec\nrounds: 66230"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.16382936362958966,
            "unit": "iter/sec",
            "range": "stddev: 0.025077957238500793",
            "extra": "mean: 6.103911886399999 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 6746604.845021276,
            "unit": "iter/sec",
            "range": "stddev: 5.08602534567748e-9",
            "extra": "mean: 148.22270208069608 nsec\nrounds: 69935"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 365972.9988986437,
            "unit": "iter/sec",
            "range": "stddev: 4.0516769191964713e-7",
            "extra": "mean: 2.732442018972417 usec\nrounds: 733"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 7011.147480184924,
            "unit": "iter/sec",
            "range": "stddev: 0.00004184918602784953",
            "extra": "mean: 142.6300049779618 usec\nrounds: 3013"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 48.04364616829693,
            "unit": "iter/sec",
            "range": "stddev: 0.00029914679148612574",
            "extra": "mean: 20.81440689361917 msec\nrounds: 47"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1602.6117950730911,
            "unit": "iter/sec",
            "range": "stddev: 0.000011321196099647877",
            "extra": "mean: 623.9814302342586 usec\nrounds: 172"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 870.1231871349802,
            "unit": "iter/sec",
            "range": "stddev: 0.000017856592484839232",
            "extra": "mean: 1.1492625582047296 msec\nrounds: 713"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 330.6968010142845,
            "unit": "iter/sec",
            "range": "stddev: 0.003661610492906536",
            "extra": "mean: 3.023917972393101 msec\nrounds: 326"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 75.78313493788731,
            "unit": "iter/sec",
            "range": "stddev: 0.0004523370234418533",
            "extra": "mean: 13.19554807041977 msec\nrounds: 71"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 41.480168960503065,
            "unit": "iter/sec",
            "range": "stddev: 0.0009775766931203328",
            "extra": "mean: 24.10790565853742 msec\nrounds: 41"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 520.4001546761097,
            "unit": "iter/sec",
            "range": "stddev: 0.000013825840345885233",
            "extra": "mean: 1.9215981990289515 msec\nrounds: 412"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ibis-squawk-bot[bot]@users.noreply.github.com",
            "name": "ibis-squawk-bot[bot]",
            "username": "ibis-squawk-bot[bot]"
          },
          "committer": {
            "email": "96536427+ibis-squawk-bot[bot]@users.noreply.github.com",
            "name": "ibis-squawk-bot[bot]",
            "username": "ibis-squawk-bot[bot]"
          },
          "distinct": true,
          "id": "e76fbde8b8fbcdcc2d8f94010a84b476b7c74807",
          "message": "chore(deps/poetry2nix): update",
          "timestamp": "2022-04-02T01:15:54Z",
          "tree_id": "fd8b8fe248e780b8b7de01243ed8d8168eaa8cff",
          "url": "https://github.com/ibis-project/ibis/commit/e76fbde8b8fbcdcc2d8f94010a84b476b7c74807"
        },
        "date": 1648862274321,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 153.03373307271576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000816584974218829",
            "extra": "mean: 6.534507000001355 msec\nrounds: 8"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 355.7729760574634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000704265672738496",
            "extra": "mean: 2.8107812208830696 msec\nrounds: 249"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 883.4845873729904,
            "unit": "iter/sec",
            "range": "stddev: 0.00002122556294960238",
            "extra": "mean: 1.1318816584831026 msec\nrounds: 448"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 21509.94508952503,
            "unit": "iter/sec",
            "range": "stddev: 0.000011618074677312019",
            "extra": "mean: 46.49012332844041 usec\nrounds: 6730"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 87.57140966393631,
            "unit": "iter/sec",
            "range": "stddev: 0.00010171044354231126",
            "extra": "mean: 11.419252057693212 msec\nrounds: 52"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1632.6179428999617,
            "unit": "iter/sec",
            "range": "stddev: 0.00001005366138348897",
            "extra": "mean: 612.5131751423332 usec\nrounds: 177"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 820.7188285322496,
            "unit": "iter/sec",
            "range": "stddev: 0.000017270410366632936",
            "extra": "mean: 1.2184440824738623 msec\nrounds: 97"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 184.0019498832579,
            "unit": "iter/sec",
            "range": "stddev: 0.00021603637941333256",
            "extra": "mean: 5.434725015873262 msec\nrounds: 63"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 8663415.037578546,
            "unit": "iter/sec",
            "range": "stddev: 4.445194732100368e-9",
            "extra": "mean: 115.42792255270628 nsec\nrounds: 71429"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 188.75855291834796,
            "unit": "iter/sec",
            "range": "stddev: 0.00006450038770943687",
            "extra": "mean: 5.297773184521996 msec\nrounds: 168"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 518.0481634036436,
            "unit": "iter/sec",
            "range": "stddev: 0.00001628915796218689",
            "extra": "mean: 1.9303224500013094 msec\nrounds: 320"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 7054.366245622548,
            "unit": "iter/sec",
            "range": "stddev: 0.00004118590921596773",
            "extra": "mean: 141.7561783980993 usec\nrounds: 3296"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 6654700.016857315,
            "unit": "iter/sec",
            "range": "stddev: 4.424765438841871e-9",
            "extra": "mean: 150.26973379216417 nsec\nrounds: 70922"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 394119.12827737513,
            "unit": "iter/sec",
            "range": "stddev: 0.000006645977519879634",
            "extra": "mean: 2.537303896846679 usec\nrounds: 15038"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 49.77955821675298,
            "unit": "iter/sec",
            "range": "stddev: 0.00019007188488642305",
            "extra": "mean: 20.088567191491396 msec\nrounds: 47"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 844435.7513847126,
            "unit": "iter/sec",
            "range": "stddev: 2.105941331371106e-7",
            "extra": "mean: 1.1842227171933353 usec\nrounds: 68028"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 11754.378546923388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016397879049326963",
            "extra": "mean: 85.0746805548254 usec\nrounds: 1512"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 209.26572765384736,
            "unit": "iter/sec",
            "range": "stddev: 0.00008266519455097708",
            "extra": "mean: 4.778613350649226 msec\nrounds: 154"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 52.43145058408021,
            "unit": "iter/sec",
            "range": "stddev: 0.0003482790338915461",
            "extra": "mean: 19.07252210000138 msec\nrounds: 50"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 30788.59003790497,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012288425030922617",
            "extra": "mean: 32.479564629912026 usec\nrounds: 3404"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 78.29828035264765,
            "unit": "iter/sec",
            "range": "stddev: 0.00030762707404448467",
            "extra": "mean: 12.771672576921226 msec\nrounds: 78"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 11738841.425058791,
            "unit": "iter/sec",
            "range": "stddev: 3.735319588283007e-9",
            "extra": "mean: 85.18728244040686 nsec\nrounds: 119048"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 111.87954779376079,
            "unit": "iter/sec",
            "range": "stddev: 0.0003206299825359803",
            "extra": "mean: 8.938184142855173 msec\nrounds: 7"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1640798.5660360868,
            "unit": "iter/sec",
            "range": "stddev: 2.137016115780919e-8",
            "extra": "mean: 609.4593332171165 nsec\nrounds: 81301"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 44.08806800494741,
            "unit": "iter/sec",
            "range": "stddev: 0.0004470155162752231",
            "extra": "mean: 22.68187392307106 msec\nrounds: 13"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.16602173767815803,
            "unit": "iter/sec",
            "range": "stddev: 0.04143164797593405",
            "extra": "mean: 6.023307634200006 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 97.68769020050053,
            "unit": "iter/sec",
            "range": "stddev: 0.0006309512982366289",
            "extra": "mean: 10.23670431706938 msec\nrounds: 82"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 92.16608384058648,
            "unit": "iter/sec",
            "range": "stddev: 0.00025078122385440506",
            "extra": "mean: 10.84997819512038 msec\nrounds: 82"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 69.7329442185076,
            "unit": "iter/sec",
            "range": "stddev: 0.0003234905504022152",
            "extra": "mean: 14.340424188408113 msec\nrounds: 69"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 11505.843843346263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011927806324970037",
            "extra": "mean: 86.91235633084766 usec\nrounds: 3783"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 1048834.0862048664,
            "unit": "iter/sec",
            "range": "stddev: 4.882269764585955e-8",
            "extra": "mean: 953.4396461301484 nsec\nrounds: 166667"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 352497.01598190004,
            "unit": "iter/sec",
            "range": "stddev: 1.8671501211947607e-7",
            "extra": "mean: 2.8369034478616633 usec\nrounds: 290"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 5106099.023836179,
            "unit": "iter/sec",
            "range": "stddev: 1.3367636231908587e-8",
            "extra": "mean: 195.84422380655244 nsec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 120942.44854576288,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011551666223045264",
            "extra": "mean: 8.268395522202566 usec\nrounds: 39841"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1616.892193448169,
            "unit": "iter/sec",
            "range": "stddev: 0.000012505462385702324",
            "extra": "mean: 618.4704237252884 usec\nrounds: 118"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "16d7cd4acea25ad32bfee745e0cd1725c8325996",
          "message": "docs: set up versioned documentation",
          "timestamp": "2022-04-02T11:57:18-04:00",
          "tree_id": "1624e766967bbb414fee7a1c67e87754f3cc5da4",
          "url": "https://github.com/ibis-project/ibis/commit/16d7cd4acea25ad32bfee745e0cd1725c8325996"
        },
        "date": 1648915251964,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 381442.0663205274,
            "unit": "iter/sec",
            "range": "stddev: 1.9945571375439968e-7",
            "extra": "mean: 2.621630093519092 usec\nrounds: 638"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1582.1097214902934,
            "unit": "iter/sec",
            "range": "stddev: 0.00002394332995853029",
            "extra": "mean: 632.0674137935479 usec\nrounds: 58"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 194.65297323769852,
            "unit": "iter/sec",
            "range": "stddev: 0.00007412073158743832",
            "extra": "mean: 5.137347677596788 msec\nrounds: 183"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 3895446.885623985,
            "unit": "iter/sec",
            "range": "stddev: 6.551135276605182e-8",
            "extra": "mean: 256.7099563571168 nsec\nrounds: 231"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 125448.92645754195,
            "unit": "iter/sec",
            "range": "stddev: 4.02834419819318e-7",
            "extra": "mean: 7.971371523362129 usec\nrounds: 32574"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 8339588.0046201935,
            "unit": "iter/sec",
            "range": "stddev: 4.755251724516587e-9",
            "extra": "mean: 119.91000028367515 nsec\nrounds: 70423"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 78.50415734963035,
            "unit": "iter/sec",
            "range": "stddev: 0.00012909149160590698",
            "extra": "mean: 12.738178890913332 msec\nrounds: 55"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 156.36483740816445,
            "unit": "iter/sec",
            "range": "stddev: 0.00012613646948895935",
            "extra": "mean: 6.395299714280815 msec\nrounds: 7"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 12372.347303800761,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020704261136551883",
            "extra": "mean: 80.82540648473406 usec\nrounds: 6785"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 39.963242646735466,
            "unit": "iter/sec",
            "range": "stddev: 0.009373139695729211",
            "extra": "mean: 25.022994476192448 msec\nrounds: 42"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 839.3559394622436,
            "unit": "iter/sec",
            "range": "stddev: 0.000014487847320311502",
            "extra": "mean: 1.1913896750890658 msec\nrounds: 554"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1716.4574512594793,
            "unit": "iter/sec",
            "range": "stddev: 0.00000837495913999806",
            "extra": "mean: 582.595274509271 usec\nrounds: 255"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 5722.3466819335745,
            "unit": "iter/sec",
            "range": "stddev: 0.00014180386290750372",
            "extra": "mean: 174.7534806231106 usec\nrounds: 129"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 68.41324127971559,
            "unit": "iter/sec",
            "range": "stddev: 0.0005030802496341452",
            "extra": "mean: 14.617053384612815 msec\nrounds: 52"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 63.650029571784735,
            "unit": "iter/sec",
            "range": "stddev: 0.0007049368169272997",
            "extra": "mean: 15.710911789478375 msec\nrounds: 57"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 96.43827087739893,
            "unit": "iter/sec",
            "range": "stddev: 0.0006885541599905694",
            "extra": "mean: 10.369327352118233 msec\nrounds: 71"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 346.1539042818942,
            "unit": "iter/sec",
            "range": "stddev: 0.00008274394195122732",
            "extra": "mean: 2.8888884037709395 msec\nrounds: 265"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 5421834.750278773,
            "unit": "iter/sec",
            "range": "stddev: 5.40068808855458e-9",
            "extra": "mean: 184.43940954643782 nsec\nrounds: 54348"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 90.73217142321487,
            "unit": "iter/sec",
            "range": "stddev: 0.00032120600711778665",
            "extra": "mean: 11.021448999997574 msec\nrounds: 6"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 942420.098993877,
            "unit": "iter/sec",
            "range": "stddev: 1.4575372464313039e-7",
            "extra": "mean: 1.061097912775412 usec\nrounds: 55866"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 93.02865415967591,
            "unit": "iter/sec",
            "range": "stddev: 0.0002589677583932683",
            "extra": "mean: 10.749376189873539 msec\nrounds: 79"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 29694.683575809915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011006418794197204",
            "extra": "mean: 33.676061825916435 usec\nrounds: 2135"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 831.7423000124936,
            "unit": "iter/sec",
            "range": "stddev: 0.00001739918998638495",
            "extra": "mean: 1.202295470586237 msec\nrounds: 68"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 536.5356272306134,
            "unit": "iter/sec",
            "range": "stddev: 0.000011891386428931204",
            "extra": "mean: 1.8638091288766938 msec\nrounds: 419"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 1178561.105870814,
            "unit": "iter/sec",
            "range": "stddev: 2.88706907964753e-8",
            "extra": "mean: 848.4922801359623 nsec\nrounds: 59172"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.14870695811550752,
            "unit": "iter/sec",
            "range": "stddev: 0.04867715364514109",
            "extra": "mean: 6.724634897199996 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 6481789.066624568,
            "unit": "iter/sec",
            "range": "stddev: 2.323470143421423e-8",
            "extra": "mean: 154.2783928513403 nsec\nrounds: 68494"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 205.10092369551222,
            "unit": "iter/sec",
            "range": "stddev: 0.00017309895910567295",
            "extra": "mean: 4.875648446540277 msec\nrounds: 159"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 184.86136511618085,
            "unit": "iter/sec",
            "range": "stddev: 0.0002799297853176244",
            "extra": "mean: 5.409459133721772 msec\nrounds: 172"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 425133.3911432047,
            "unit": "iter/sec",
            "range": "stddev: 1.7225658243722641e-7",
            "extra": "mean: 2.3522029105052193 usec\nrounds: 15874"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 39.49595573105995,
            "unit": "iter/sec",
            "range": "stddev: 0.0010640736064094123",
            "extra": "mean: 25.319048026317073 msec\nrounds: 38"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1333729.883186049,
            "unit": "iter/sec",
            "range": "stddev: 2.021509725835929e-7",
            "extra": "mean: 749.7770070287198 nsec\nrounds: 151516"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 21980.524211796168,
            "unit": "iter/sec",
            "range": "stddev: 0.000033845399543433784",
            "extra": "mean: 45.49482034024172 usec\nrounds: 5644"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 37.584929074101396,
            "unit": "iter/sec",
            "range": "stddev: 0.001112603783639274",
            "extra": "mean: 26.606409128202102 msec\nrounds: 39"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 11443.334299823935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012193852717661076",
            "extra": "mean: 87.38711758297457 usec\nrounds: 1820"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "09a9aedf9d59b772f644a3cade3f0d7a98bf9aa2",
          "message": "ci: configure git info before pushing docs",
          "timestamp": "2022-04-02T12:37:30-04:00",
          "tree_id": "87be424b51bdc8c9f18680c4c9582df41ece6253",
          "url": "https://github.com/ibis-project/ibis/commit/09a9aedf9d59b772f644a3cade3f0d7a98bf9aa2"
        },
        "date": 1648918010139,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 38.97368778464055,
            "unit": "iter/sec",
            "range": "stddev: 0.001180441304457418",
            "extra": "mean: 25.65833660714288 msec\nrounds: 28"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 61.32691619657315,
            "unit": "iter/sec",
            "range": "stddev: 0.0006981644659948446",
            "extra": "mean: 16.306053883333504 msec\nrounds: 60"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 478.51096037776017,
            "unit": "iter/sec",
            "range": "stddev: 0.00006279728800046",
            "extra": "mean: 2.089816290123325 msec\nrounds: 324"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 312.66254924247437,
            "unit": "iter/sec",
            "range": "stddev: 0.00017786708360402627",
            "extra": "mean: 3.198336361111434 msec\nrounds: 216"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1357.212824202381,
            "unit": "iter/sec",
            "range": "stddev: 0.000018974469675871747",
            "extra": "mean: 736.8041195659118 usec\nrounds: 92"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 379877.9707115818,
            "unit": "iter/sec",
            "range": "stddev: 7.873744331626035e-7",
            "extra": "mean: 2.6324242970099445 usec\nrounds: 13870"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 90.28888736802445,
            "unit": "iter/sec",
            "range": "stddev: 0.0006419098463203133",
            "extra": "mean: 11.07556011764685 msec\nrounds: 34"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 7298609.181247818,
            "unit": "iter/sec",
            "range": "stddev: 2.174519353780026e-8",
            "extra": "mean: 137.01240540044054 nsec\nrounds: 58140"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 41.470566368636085,
            "unit": "iter/sec",
            "range": "stddev: 0.0010598229105454634",
            "extra": "mean: 24.11348789189176 msec\nrounds: 37"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1157059.5027920117,
            "unit": "iter/sec",
            "range": "stddev: 4.6505263713823793e-7",
            "extra": "mean: 864.2597874931898 nsec\nrounds: 172414"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 136.76604246869422,
            "unit": "iter/sec",
            "range": "stddev: 0.0001340495658454939",
            "extra": "mean: 7.31175650000182 msec\nrounds: 6"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 844768.0588933228,
            "unit": "iter/sec",
            "range": "stddev: 5.607703029938932e-7",
            "extra": "mean: 1.1837568779648662 usec\nrounds: 65790"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 91.64866217788125,
            "unit": "iter/sec",
            "range": "stddev: 0.0005344179445291274",
            "extra": "mean: 10.91123401298642 msec\nrounds: 77"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 33.553901710198055,
            "unit": "iter/sec",
            "range": "stddev: 0.0017804874060871072",
            "extra": "mean: 29.802793387097203 msec\nrounds: 31"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.13012444418480304,
            "unit": "iter/sec",
            "range": "stddev: 0.0703629776190998",
            "extra": "mean: 7.6849511732 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 11124.803473760594,
            "unit": "iter/sec",
            "range": "stddev: 0.00000961976220967235",
            "extra": "mean: 89.8892283678215 usec\nrounds: 3363"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 9783563.030545715,
            "unit": "iter/sec",
            "range": "stddev: 1.748819967141683e-8",
            "extra": "mean: 102.21225098438394 nsec\nrounds: 99010"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 25694.899099947852,
            "unit": "iter/sec",
            "range": "stddev: 0.000002237546095123335",
            "extra": "mean: 38.91823027248352 usec\nrounds: 2015"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 182.4353178512478,
            "unit": "iter/sec",
            "range": "stddev: 0.000183252870605631",
            "extra": "mean: 5.481394785714516 msec\nrounds: 168"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 57.24394118493256,
            "unit": "iter/sec",
            "range": "stddev: 0.00953370490331581",
            "extra": "mean: 17.469097677418734 msec\nrounds: 62"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 400183.58827920083,
            "unit": "iter/sec",
            "range": "stddev: 0.00000810046939569847",
            "extra": "mean: 2.4988530996486493 usec\nrounds: 742"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 83.68893855317499,
            "unit": "iter/sec",
            "range": "stddev: 0.00047794824017004294",
            "extra": "mean: 11.94901043421182 msec\nrounds: 76"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 10718.763357263513,
            "unit": "iter/sec",
            "range": "stddev: 0.000006675998967987634",
            "extra": "mean: 93.29434438184096 usec\nrounds: 3560"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 819286.3192223519,
            "unit": "iter/sec",
            "range": "stddev: 4.6207370065864767e-7",
            "extra": "mean: 1.2205745128872374 usec\nrounds: 158731"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 124834.36433550323,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015409902288559986",
            "extra": "mean: 8.010614747974468 usec\nrounds: 31055"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 173.6696831225727,
            "unit": "iter/sec",
            "range": "stddev: 0.00044297133133353316",
            "extra": "mean: 5.758057376624677 msec\nrounds: 154"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 73.36082184878657,
            "unit": "iter/sec",
            "range": "stddev: 0.000616875618700419",
            "extra": "mean: 13.631254050850584 msec\nrounds: 59"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 4570408.111284482,
            "unit": "iter/sec",
            "range": "stddev: 5.205085543511313e-8",
            "extra": "mean: 218.79884151495696 nsec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 4958768.504736293,
            "unit": "iter/sec",
            "range": "stddev: 2.986947122569244e-8",
            "extra": "mean: 201.66297318475287 nsec\nrounds: 55866"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 6324.453794134421,
            "unit": "iter/sec",
            "range": "stddev: 0.000053070283521362466",
            "extra": "mean: 158.1164212042223 usec\nrounds: 3122"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 766.2131498195042,
            "unit": "iter/sec",
            "range": "stddev: 0.00013510569493308731",
            "extra": "mean: 1.3051198615366608 msec\nrounds: 65"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 737.9709196902618,
            "unit": "iter/sec",
            "range": "stddev: 0.000043127465042669835",
            "extra": "mean: 1.3550669454830497 msec\nrounds: 642"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1535.956579575339,
            "unit": "iter/sec",
            "range": "stddev: 0.000052065208144361105",
            "extra": "mean: 651.0600711619594 usec\nrounds: 534"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 174.8547885154627,
            "unit": "iter/sec",
            "range": "stddev: 0.0002059957460714592",
            "extra": "mean: 5.719031251532288 msec\nrounds: 163"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 15424.837104356136,
            "unit": "iter/sec",
            "range": "stddev: 0.0011356862119656023",
            "extra": "mean: 64.83050636026421 usec\nrounds: 5660"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "bb9c4460e2149b109b3b6f7d32b7f04f3996c377",
          "message": "ci: increase fetch depth",
          "timestamp": "2022-04-02T12:49:26-04:00",
          "tree_id": "ceea737c371440b9d5b114b6dfa470690ef25605",
          "url": "https://github.com/ibis-project/ibis/commit/bb9c4460e2149b109b3b6f7d32b7f04f3996c377"
        },
        "date": 1648918746497,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 2540721.8907665666,
            "unit": "iter/sec",
            "range": "stddev: 8.437795013610896e-7",
            "extra": "mean: 393.58892590101146 nsec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 344164.13367856224,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017343364459478818",
            "extra": "mean: 2.9055903917459522 usec\nrounds: 12739"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 67.65716825427069,
            "unit": "iter/sec",
            "range": "stddev: 0.0012607104060432042",
            "extra": "mean: 14.780399857141191 msec\nrounds: 28"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 929920.1544059837,
            "unit": "iter/sec",
            "range": "stddev: 0.000001382892497389184",
            "extra": "mean: 1.0753611428481618 usec\nrounds: 185186"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 425.4519588922075,
            "unit": "iter/sec",
            "range": "stddev: 0.00026281334311415144",
            "extra": "mean: 2.35044164000044 msec\nrounds: 75"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 32.9260319363204,
            "unit": "iter/sec",
            "range": "stddev: 0.0012260987257910238",
            "extra": "mean: 30.371105814816072 msec\nrounds: 27"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1251.7854555955973,
            "unit": "iter/sec",
            "range": "stddev: 0.00013170009676203816",
            "extra": "mean: 798.8589382708572 usec\nrounds: 81"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 105819.07430360078,
            "unit": "iter/sec",
            "range": "stddev: 0.000009232207781648723",
            "extra": "mean: 9.450092117900642 usec\nrounds: 37452"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 65.13370900178026,
            "unit": "iter/sec",
            "range": "stddev: 0.0005024743161493536",
            "extra": "mean: 15.353033249997594 msec\nrounds: 12"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 15333.131629350017,
            "unit": "iter/sec",
            "range": "stddev: 0.00003265292220539102",
            "extra": "mean: 65.21824922482524 usec\nrounds: 5160"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 227821.081212139,
            "unit": "iter/sec",
            "range": "stddev: 0.000022390856664756098",
            "extra": "mean: 4.389409420231989 usec\nrounds: 552"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 84.6018726116808,
            "unit": "iter/sec",
            "range": "stddev: 0.0004896389168410002",
            "extra": "mean: 11.820069333334496 msec\nrounds: 6"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 66.01127513368348,
            "unit": "iter/sec",
            "range": "stddev: 0.0010027774138870967",
            "extra": "mean: 15.148927179104456 msec\nrounds: 67"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 163.98240626687956,
            "unit": "iter/sec",
            "range": "stddev: 0.0003115408955861347",
            "extra": "mean: 6.098215185185849 msec\nrounds: 135"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 9308.607375611142,
            "unit": "iter/sec",
            "range": "stddev: 0.00005476954264714086",
            "extra": "mean: 107.4274550047124 usec\nrounds: 6334"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 4468.360051239046,
            "unit": "iter/sec",
            "range": "stddev: 0.00021862519926256775",
            "extra": "mean: 223.79575247583435 usec\nrounds: 101"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 275.4414141607426,
            "unit": "iter/sec",
            "range": "stddev: 0.00036952568562350973",
            "extra": "mean: 3.6305361089106882 msec\nrounds: 202"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 9557.910134195361,
            "unit": "iter/sec",
            "range": "stddev: 0.00013230362514358613",
            "extra": "mean: 104.62538211384697 usec\nrounds: 3444"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.09868519627780875,
            "unit": "iter/sec",
            "range": "stddev: 0.06787669085583753",
            "extra": "mean: 10.133232113000002 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 38.225790643220854,
            "unit": "iter/sec",
            "range": "stddev: 0.0013559904949814873",
            "extra": "mean: 26.160348371429823 msec\nrounds: 35"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 141.6343518702822,
            "unit": "iter/sec",
            "range": "stddev: 0.00048507310781192164",
            "extra": "mean: 7.060434045801715 msec\nrounds: 131"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 23415.51851803307,
            "unit": "iter/sec",
            "range": "stddev: 0.000017444607620993984",
            "extra": "mean: 42.706720298756856 usec\nrounds: 1877"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 72.09312537244745,
            "unit": "iter/sec",
            "range": "stddev: 0.0004742514696876409",
            "extra": "mean: 13.87094809434049 msec\nrounds: 53"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 139.6118011498065,
            "unit": "iter/sec",
            "range": "stddev: 0.0004046190179210484",
            "extra": "mean: 7.162718278571439 msec\nrounds: 140"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 128.16721798519458,
            "unit": "iter/sec",
            "range": "stddev: 0.0009670635700448187",
            "extra": "mean: 7.802307140001403 msec\nrounds: 100"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 33.03722621648392,
            "unit": "iter/sec",
            "range": "stddev: 0.0017536750844092744",
            "extra": "mean: 30.268884967741332 msec\nrounds: 31"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1446981.7046625318,
            "unit": "iter/sec",
            "range": "stddev: 6.324620884843497e-7",
            "extra": "mean: 691.0937413911543 nsec\nrounds: 70423"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 684.6990853043799,
            "unit": "iter/sec",
            "range": "stddev: 0.00013245575733118232",
            "extra": "mean: 1.4604955979391363 msec\nrounds: 485"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 788427.0564921692,
            "unit": "iter/sec",
            "range": "stddev: 0.000004066320749775875",
            "extra": "mean: 1.2683481518875708 usec\nrounds: 55246"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1341.0374977309482,
            "unit": "iter/sec",
            "range": "stddev: 0.0000845286796020874",
            "extra": "mean: 745.6913037047899 usec\nrounds: 135"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 640.6846183902948,
            "unit": "iter/sec",
            "range": "stddev: 0.00037322809767343414",
            "extra": "mean: 1.560830354430042 msec\nrounds: 79"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 72.66348385062867,
            "unit": "iter/sec",
            "range": "stddev: 0.0010884158482969855",
            "extra": "mean: 13.762070671641053 msec\nrounds: 67"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 9262343.908124836,
            "unit": "iter/sec",
            "range": "stddev: 1.1596941490761845e-7",
            "extra": "mean: 107.96403263784737 nsec\nrounds: 93450"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 6741605.883823828,
            "unit": "iter/sec",
            "range": "stddev: 1.3957980013785538e-7",
            "extra": "mean: 148.33261054316353 nsec\nrounds: 56494"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 5144392.843680521,
            "unit": "iter/sec",
            "range": "stddev: 3.121234093561211e-7",
            "extra": "mean: 194.38639901484058 nsec\nrounds: 56815"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "committer": {
            "email": "417981+cpcloud@users.noreply.github.com",
            "name": "Phillip Cloud",
            "username": "cpcloud"
          },
          "distinct": true,
          "id": "29030562917e7cd9bc4fe79b3cfa7e39cf29d540",
          "message": "ci: fetch just before pushing",
          "timestamp": "2022-04-02T13:03:29-04:00",
          "tree_id": "413181c956c34de07117569fa7964bd7c4c30da7",
          "url": "https://github.com/ibis-project/ibis/commit/29030562917e7cd9bc4fe79b3cfa7e39cf29d540"
        },
        "date": 1648919167453,
        "tool": "pytest",
        "benches": [
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_argnames",
            "value": 5254559.5639229575,
            "unit": "iter/sec",
            "range": "stddev: 1.2544895093604697e-8",
            "extra": "mean: 190.3109076668976 nsec\nrounds: 196079"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[large]",
            "value": 78.6447215318605,
            "unit": "iter/sec",
            "range": "stddev: 0.00016389721640296074",
            "extra": "mean: 12.715411543480137 msec\nrounds: 46"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort_projection]",
            "value": 54.85776971187541,
            "unit": "iter/sec",
            "range": "stddev: 0.0009392687842548271",
            "extra": "mean: 18.228958363641304 msec\nrounds: 11"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-sqlite]",
            "value": 81.57587349212316,
            "unit": "iter/sec",
            "range": "stddev: 0.00077480853240648",
            "extra": "mean: 12.25852641463359 msec\nrounds: 41"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort_projection]",
            "value": 32.34125317439866,
            "unit": "iter/sec",
            "range": "stddev: 0.0007821700846125455",
            "extra": "mean: 30.920261333336335 msec\nrounds: 33"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-medium]",
            "value": 1453.934002065326,
            "unit": "iter/sec",
            "range": "stddev: 0.00006306430999505403",
            "extra": "mean: 687.7891283782421 usec\nrounds: 148"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-large]",
            "value": 827358.6308240616,
            "unit": "iter/sec",
            "range": "stddev: 1.770569376916463e-7",
            "extra": "mean: 1.2086657016002662 usec\nrounds: 53192"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_parse",
            "value": 3112531.248805307,
            "unit": "iter/sec",
            "range": "stddev: 2.8373607665481145e-7",
            "extra": "mean: 321.28191496353116 nsec\nrounds: 188"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[hash]",
            "value": 4320975.342708206,
            "unit": "iter/sec",
            "range": "stddev: 1.686086750191771e-8",
            "extra": "mean: 231.42923083011064 nsec\nrounds: 188680"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates_from_strings]",
            "value": 65.67883162419064,
            "unit": "iter/sec",
            "range": "stddev: 0.00026003836819929124",
            "extra": "mean: 15.225605804346905 msec\nrounds: 46"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-small]",
            "value": 25909.690632316895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021327036297339742",
            "extra": "mean: 38.59559784757562 usec\nrounds: 2509"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_tpc_h02",
            "value": 1353.6676021676592,
            "unit": "iter/sec",
            "range": "stddev: 0.00006854091858843827",
            "extra": "mean: 738.7337913670068 usec\nrounds: 139"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_repr_huge_union",
            "value": 482.34040066083753,
            "unit": "iter/sec",
            "range": "stddev: 0.000011103144535149381",
            "extra": "mean: 2.0732246327073893 msec\nrounds: 373"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[complex]",
            "value": 371306.3641895899,
            "unit": "iter/sec",
            "range": "stddev: 6.859115780085337e-7",
            "extra": "mean: 2.6931938055588986 usec\nrounds: 14045"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-medium]",
            "value": 1515470.174765913,
            "unit": "iter/sec",
            "range": "stddev: 2.7148279628143714e-8",
            "extra": "mean: 659.8612210593491 nsec\nrounds: 74075"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_eq_datatypes[singletons]",
            "value": 10420.935053580008,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015098694927824499",
            "extra": "mean: 95.96067865872172 usec\nrounds: 3280"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-sqlite]",
            "value": 5743.28995105697,
            "unit": "iter/sec",
            "range": "stddev: 0.00005225282238784924",
            "extra": "mean: 174.1162310316519 usec\nrounds: 2346"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_window_analytics_udf]",
            "value": 0.12749945391754383,
            "unit": "iter/sec",
            "range": "stddev: 0.047398339384778776",
            "extra": "mean: 7.843170847200002 sec\nrounds: 5"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[small]",
            "value": 10565.55684550311,
            "unit": "iter/sec",
            "range": "stddev: 0.000004321044295592733",
            "extra": "mean: 94.64716480377635 usec\nrounds: 3131"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[high_card_group_by]",
            "value": 127.72726550271672,
            "unit": "iter/sec",
            "range": "stddev: 0.0001128696560886713",
            "extra": "mean: 7.82918193749893 msec\nrounds: 16"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_complex_datatype_builtins[str]",
            "value": 120916.76106197965,
            "unit": "iter/sec",
            "range": "stddev: 4.802827669852306e-7",
            "extra": "mean: 8.270152055159821 usec\nrounds: 29851"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-impala]",
            "value": 297.9886342874006,
            "unit": "iter/sec",
            "range": "stddev: 0.00011947921971298978",
            "extra": "mean: 3.3558326893620096 msec\nrounds: 235"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[medium-sqlite]",
            "value": 158.1733230204813,
            "unit": "iter/sec",
            "range": "stddev: 0.00031456828291663645",
            "extra": "mean: 6.322178613333638 msec\nrounds: 150"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[large-impala]",
            "value": 176.76075067119913,
            "unit": "iter/sec",
            "range": "stddev: 0.00009484918806867984",
            "extra": "mean: 5.657364523531282 msec\nrounds: 170"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[cast_to_dates]",
            "value": 719.158283219982,
            "unit": "iter/sec",
            "range": "stddev: 0.000019593189912978298",
            "extra": "mean: 1.3905144713380322 msec\nrounds: 471"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_sort]",
            "value": 34.092244442649715,
            "unit": "iter/sec",
            "range": "stddev: 0.0007751674788727549",
            "extra": "mean: 29.33218438235151 msec\nrounds: 34"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[simple_sort]",
            "value": 51.49966439403258,
            "unit": "iter/sec",
            "range": "stddev: 0.0008651084466439962",
            "extra": "mean: 19.417602265304726 msec\nrounds: 49"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_op_args",
            "value": 6985167.482976325,
            "unit": "iter/sec",
            "range": "stddev: 6.341903738399022e-9",
            "extra": "mean: 143.1604900580398 nsec\nrounds: 56177"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_large_expr_equals",
            "value": 354212.9639386899,
            "unit": "iter/sec",
            "range": "stddev: 1.1596529640190391e-7",
            "extra": "mean: 2.8231603634165356 usec\nrounds: 661"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[hash-small]",
            "value": 841093.7772346251,
            "unit": "iter/sec",
            "range": "stddev: 1.4620971905365485e-7",
            "extra": "mean: 1.1889280685060255 usec\nrounds: 172414"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[low_card_window_analytics_udf]",
            "value": 36.137442448927665,
            "unit": "iter/sec",
            "range": "stddev: 0.0009365101576545925",
            "extra": "mean: 27.67212985294353 msec\nrounds: 34"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_execute[multikey_group_by_with_mutate]",
            "value": 81.07124339750528,
            "unit": "iter/sec",
            "range": "stddev: 0.0003748913494922443",
            "extra": "mean: 12.334829935897737 msec\nrounds: 78"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_compile[small-impala]",
            "value": 18697.60193528481,
            "unit": "iter/sec",
            "range": "stddev: 0.00004251722614187385",
            "extra": "mean: 53.48279439583479 usec\nrounds: 5603"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_construction[medium]",
            "value": 166.67898245006344,
            "unit": "iter/sec",
            "range": "stddev: 0.00006208807214310291",
            "extra": "mean: 5.999556664557856 msec\nrounds: 158"
          },
          {
            "name": "ibis/tests/benchmarks/test_benchmarks.py::test_builtins[str-large]",
            "value": 710.3322580008767,
            "unit": "iter/sec",
            "range": "stddev: 0.000022517058261623937",
            "extra": "mean: 1.4077919012355564 msec\nrounds: 81"
          }
        ]
      }
    ]
  }
}