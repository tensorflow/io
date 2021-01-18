window.BENCHMARK_DATA = {
  "lastUpdate": 1611003392141,
  "repoUrl": "https://github.com/tensorflow/io",
  "entries": {
    "Tensorflow-IO Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "vikoth18@in.ibm.com",
            "name": "Vignesh Kothapalli",
            "username": "kvignesh1420"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3e16038f8ce6bf76c927176d4d1fc8f4a73c2771",
          "message": "handle missing dependencies while benchmarking (#1271)\n\n* handle missing dependencies while benchmarking\r\n\r\n* setup test_sql\r\n\r\n* job name change\r\n\r\n* set auto-push to true\r\n\r\n* remove auto-push\r\n\r\n* add personal access token\r\n\r\n* use alternate method to push to gh-pages\r\n\r\n* add name to the action\r\n\r\n* use different id\r\n\r\n* modify creds\r\n\r\n* use github_token\r\n\r\n* change repo name\r\n\r\n* set auto-push\r\n\r\n* set origin and push results\r\n\r\n* set env\r\n\r\n* use PERSONAL_GITHUB_TOKEN\r\n\r\n* use push changes action\r\n\r\n* use github.head_ref to push the changes\r\n\r\n* try using fetch-depth\r\n\r\n* modify branch name\r\n\r\n* use alternative push approach\r\n\r\n* git switch -\r\n\r\n* test by merging with forked master",
          "timestamp": "2021-01-18T12:47:47-08:00",
          "tree_id": "08e90708e7a2b56ce5ee09ae6475345ecca503a5",
          "url": "https://github.com/tensorflow/io/commit/3e16038f8ce6bf76c927176d4d1fc8f4a73c2771"
        },
        "date": 1611003276387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.304679994883793,
            "unit": "iter/sec",
            "range": "stddev: 0.04178211856572626",
            "extra": "mean: 232.30530520004322 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 33.73616537924919,
            "unit": "iter/sec",
            "range": "stddev: 0.0009380559352526182",
            "extra": "mean: 29.641780230751742 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4558491864445848,
            "unit": "iter/sec",
            "range": "stddev: 0.053385406140713715",
            "extra": "mean: 686.8843347999245 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4595872604308204,
            "unit": "iter/sec",
            "range": "stddev: 0.05253451834136777",
            "extra": "mean: 685.1251906000016 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4411195097800904,
            "unit": "iter/sec",
            "range": "stddev: 0.05314464370214676",
            "extra": "mean: 693.9049768000132 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.6008237154949962,
            "unit": "iter/sec",
            "range": "stddev: 0.04875915525437668",
            "extra": "mean: 1.6643817050000052 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.46461061589205505,
            "unit": "iter/sec",
            "range": "stddev: 0.05371091401431979",
            "extra": "mean: 2.1523399719999814 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8020236367824926,
            "unit": "iter/sec",
            "range": "stddev: 0.00749891863595247",
            "extra": "mean: 1.246846045600023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.8716799631934227,
            "unit": "iter/sec",
            "range": "stddev: 0.05850214480012198",
            "extra": "mean: 258.2858111999485 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.2969044619485253,
            "unit": "iter/sec",
            "range": "stddev: 0.06442334909934348",
            "extra": "mean: 435.36856519999674 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.2498744825397408,
            "unit": "iter/sec",
            "range": "stddev: 0.0626726246238478",
            "extra": "mean: 444.46923940004126 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.239904457072719,
            "unit": "iter/sec",
            "range": "stddev: 0.06240588378338161",
            "extra": "mean: 446.44761379995543 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 29.131760845155497,
            "unit": "iter/sec",
            "range": "stddev: 0.0012068292108734942",
            "extra": "mean: 34.32679560000906 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5969.604003544291,
            "unit": "iter/sec",
            "range": "stddev: 0.000008094597187428939",
            "extra": "mean: 167.5152990728157 usec\nrounds: 2588"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4326.561367343299,
            "unit": "iter/sec",
            "range": "stddev: 0.000006529570915804745",
            "extra": "mean: 231.13043248339375 usec\nrounds: 2666"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1046.4884644569722,
            "unit": "iter/sec",
            "range": "stddev: 0.00000950120872042345",
            "extra": "mean: 955.5767062553382 usec\nrounds: 960"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 558.4202864408825,
            "unit": "iter/sec",
            "range": "stddev: 0.000012580490141841481",
            "extra": "mean: 1.7907658877752208 msec\nrounds: 499"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1751.803047545617,
            "unit": "iter/sec",
            "range": "stddev: 0.000009214456558069606",
            "extra": "mean: 570.8404271821887 usec\nrounds: 1442"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 756.8018826019883,
            "unit": "iter/sec",
            "range": "stddev: 0.00001520540824503138",
            "extra": "mean: 1.3213497785733082 msec\nrounds: 420"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1259.2594325442471,
            "unit": "iter/sec",
            "range": "stddev: 0.000012764260008010975",
            "extra": "mean: 794.1175377813678 usec\nrounds: 794"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vikoth18@in.ibm.com",
            "name": "Vignesh Kothapalli",
            "username": "kvignesh1420"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3e16038f8ce6bf76c927176d4d1fc8f4a73c2771",
          "message": "handle missing dependencies while benchmarking (#1271)\n\n* handle missing dependencies while benchmarking\r\n\r\n* setup test_sql\r\n\r\n* job name change\r\n\r\n* set auto-push to true\r\n\r\n* remove auto-push\r\n\r\n* add personal access token\r\n\r\n* use alternate method to push to gh-pages\r\n\r\n* add name to the action\r\n\r\n* use different id\r\n\r\n* modify creds\r\n\r\n* use github_token\r\n\r\n* change repo name\r\n\r\n* set auto-push\r\n\r\n* set origin and push results\r\n\r\n* set env\r\n\r\n* use PERSONAL_GITHUB_TOKEN\r\n\r\n* use push changes action\r\n\r\n* use github.head_ref to push the changes\r\n\r\n* try using fetch-depth\r\n\r\n* modify branch name\r\n\r\n* use alternative push approach\r\n\r\n* git switch -\r\n\r\n* test by merging with forked master",
          "timestamp": "2021-01-18T12:47:47-08:00",
          "tree_id": "08e90708e7a2b56ce5ee09ae6475345ecca503a5",
          "url": "https://github.com/tensorflow/io/commit/3e16038f8ce6bf76c927176d4d1fc8f4a73c2771"
        },
        "date": 1611003391692,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.291225781624101,
            "unit": "iter/sec",
            "range": "stddev: 0.03515597144107129",
            "extra": "mean: 233.03364839999858 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 33.201100246668744,
            "unit": "iter/sec",
            "range": "stddev: 0.0010848756378380958",
            "extra": "mean: 30.119483769226463 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4975548970432138,
            "unit": "iter/sec",
            "range": "stddev: 0.04762134807869421",
            "extra": "mean: 667.7551533999917 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.495963298917228,
            "unit": "iter/sec",
            "range": "stddev: 0.047597073848552184",
            "extra": "mean: 668.4655972000087 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4833546076199464,
            "unit": "iter/sec",
            "range": "stddev: 0.04943309097102761",
            "extra": "mean: 674.1476346000013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.6075462648640505,
            "unit": "iter/sec",
            "range": "stddev: 0.05105520937978464",
            "extra": "mean: 1.6459651846000043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.46813303994726785,
            "unit": "iter/sec",
            "range": "stddev: 0.04703333564479941",
            "extra": "mean: 2.136144887599994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8222328245819328,
            "unit": "iter/sec",
            "range": "stddev: 0.006226010202048463",
            "extra": "mean: 1.2162005336000221 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 4.039193963172943,
            "unit": "iter/sec",
            "range": "stddev: 0.049113333161569274",
            "extra": "mean: 247.57414699997753 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.34913833053974,
            "unit": "iter/sec",
            "range": "stddev: 0.058516690652003725",
            "extra": "mean: 425.6880010000259 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.3157606969205253,
            "unit": "iter/sec",
            "range": "stddev: 0.05930214942026509",
            "extra": "mean: 431.8235477999906 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.2887013040512443,
            "unit": "iter/sec",
            "range": "stddev: 0.05658930704243954",
            "extra": "mean: 436.9290122000166 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 28.864014343176034,
            "unit": "iter/sec",
            "range": "stddev: 0.0008764749606082044",
            "extra": "mean: 34.64521559997138 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6138.950916134795,
            "unit": "iter/sec",
            "range": "stddev: 0.000006574361987692262",
            "extra": "mean: 162.8942817203073 usec\nrounds: 2396"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4382.747111592647,
            "unit": "iter/sec",
            "range": "stddev: 0.000006740440391520411",
            "extra": "mean: 228.16739696318228 usec\nrounds: 2766"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1052.3148505638233,
            "unit": "iter/sec",
            "range": "stddev: 0.000010911277661695742",
            "extra": "mean: 950.2859334012123 usec\nrounds: 961"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 555.86983447307,
            "unit": "iter/sec",
            "range": "stddev: 0.00004171074297066913",
            "extra": "mean: 1.7989823120154338 msec\nrounds: 516"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1741.8977027282392,
            "unit": "iter/sec",
            "range": "stddev: 0.000009456391143903642",
            "extra": "mean: 574.0865255369214 usec\nrounds: 1351"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 761.9024603759048,
            "unit": "iter/sec",
            "range": "stddev: 0.000029726492385136535",
            "extra": "mean: 1.312503964755047 msec\nrounds: 454"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1269.6042714628938,
            "unit": "iter/sec",
            "range": "stddev: 0.000012961471499867283",
            "extra": "mean: 787.6470034617606 usec\nrounds: 867"
          }
        ]
      }
    ]
  }
}