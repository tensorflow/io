window.BENCHMARK_DATA = {
  "lastUpdate": 1612078427592,
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
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c652bee06fbe3d86120a49d7f823491a85234db",
          "message": "Disable s3 macOS for now as docker is not working on GitHub Actions for macOS (#1277)\n\n* Revert \"[s3] add support for testing on macOS (#1253)\"\r\n\r\nThis reverts commit 81789bde99e62523ca4d9f460bb345c666902acd.\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>\r\n\r\n* Update\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-19T08:23:38-08:00",
          "tree_id": "1f4ebd0d670b0eac026c20b6f881707acc9b0a05",
          "url": "https://github.com/tensorflow/io/commit/5c652bee06fbe3d86120a49d7f823491a85234db"
        },
        "date": 1611073760051,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.255031939712024,
            "unit": "iter/sec",
            "range": "stddev: 0.03778937947304685",
            "extra": "mean: 235.01586220000945 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 33.26321762631914,
            "unit": "iter/sec",
            "range": "stddev: 0.001074748280751596",
            "extra": "mean: 30.063237153845314 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4555661093575591,
            "unit": "iter/sec",
            "range": "stddev: 0.05220422840404052",
            "extra": "mean: 687.0179194000116 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4615482314937203,
            "unit": "iter/sec",
            "range": "stddev: 0.050070668232670486",
            "extra": "mean: 684.2059525999957 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4390824232713808,
            "unit": "iter/sec",
            "range": "stddev: 0.053527942579283797",
            "extra": "mean: 694.8872307999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.6003612693389956,
            "unit": "iter/sec",
            "range": "stddev: 0.05266461458382216",
            "extra": "mean: 1.665663744599999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.45875239951004637,
            "unit": "iter/sec",
            "range": "stddev: 0.05849560239517374",
            "extra": "mean: 2.179825110599995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8209834399484565,
            "unit": "iter/sec",
            "range": "stddev: 0.004202830139726732",
            "extra": "mean: 1.2180513653999925 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.9545466214601803,
            "unit": "iter/sec",
            "range": "stddev: 0.05306372182504714",
            "extra": "mean: 252.8734886000052 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.3282919527720067,
            "unit": "iter/sec",
            "range": "stddev: 0.059911782016372046",
            "extra": "mean: 429.499401399994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.3020966240948404,
            "unit": "iter/sec",
            "range": "stddev: 0.058715683901597766",
            "extra": "mean: 434.3866324000146 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.2649925640338466,
            "unit": "iter/sec",
            "range": "stddev: 0.05716465644477851",
            "extra": "mean: 441.5025532000186 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 28.476967541028813,
            "unit": "iter/sec",
            "range": "stddev: 0.000539056948969044",
            "extra": "mean: 35.11609860000817 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6142.647416381999,
            "unit": "iter/sec",
            "range": "stddev: 0.000007238232886319312",
            "extra": "mean: 162.79625578591276 usec\nrounds: 2506"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4395.542595104788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072645599462775355",
            "extra": "mean: 227.50319860707896 usec\nrounds: 2729"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1050.4273470123194,
            "unit": "iter/sec",
            "range": "stddev: 0.000008939661784085216",
            "extra": "mean: 951.9934937378131 usec\nrounds: 958"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 558.4193874344173,
            "unit": "iter/sec",
            "range": "stddev: 0.000012777537388486969",
            "extra": "mean: 1.7907687707519708 msec\nrounds: 506"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1753.3126790115502,
            "unit": "iter/sec",
            "range": "stddev: 0.000008097427557599297",
            "extra": "mean: 570.3489240514482 usec\nrounds: 1422"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 767.3392568328779,
            "unit": "iter/sec",
            "range": "stddev: 0.000014793018542079358",
            "extra": "mean: 1.3032045357973838 msec\nrounds: 433"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1278.404196713598,
            "unit": "iter/sec",
            "range": "stddev: 0.000012358788586115806",
            "extra": "mean: 782.2252168529378 usec\nrounds: 807"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c652bee06fbe3d86120a49d7f823491a85234db",
          "message": "Disable s3 macOS for now as docker is not working on GitHub Actions for macOS (#1277)\n\n* Revert \"[s3] add support for testing on macOS (#1253)\"\r\n\r\nThis reverts commit 81789bde99e62523ca4d9f460bb345c666902acd.\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>\r\n\r\n* Update\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-19T08:23:38-08:00",
          "tree_id": "1f4ebd0d670b0eac026c20b6f881707acc9b0a05",
          "url": "https://github.com/tensorflow/io/commit/5c652bee06fbe3d86120a49d7f823491a85234db"
        },
        "date": 1611073914804,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.252359926235029,
            "unit": "iter/sec",
            "range": "stddev: 0.03899494419884176",
            "extra": "mean: 235.1635367999961 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 31.910087923834816,
            "unit": "iter/sec",
            "range": "stddev: 0.0005162282522892502",
            "extra": "mean: 31.33805216666493 msec\nrounds: 12"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4407835033711964,
            "unit": "iter/sec",
            "range": "stddev: 0.04836149413131772",
            "extra": "mean: 694.0668030000097 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4332244042942441,
            "unit": "iter/sec",
            "range": "stddev: 0.050069826147766866",
            "extra": "mean: 697.7274437999995 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4200060901845601,
            "unit": "iter/sec",
            "range": "stddev: 0.05129116661601383",
            "extra": "mean: 704.2223318000197 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.596601263854984,
            "unit": "iter/sec",
            "range": "stddev: 0.05414707410563692",
            "extra": "mean: 1.6761613838000016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.46217430239984864,
            "unit": "iter/sec",
            "range": "stddev: 0.05572074674595085",
            "extra": "mean: 2.163685853599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8156998653618865,
            "unit": "iter/sec",
            "range": "stddev: 0.005129472715181833",
            "extra": "mean: 1.2259411119999981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.90575670597923,
            "unit": "iter/sec",
            "range": "stddev: 0.050792618769902535",
            "extra": "mean: 256.0323326000116 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.294259156666729,
            "unit": "iter/sec",
            "range": "stddev: 0.060179297796585575",
            "extra": "mean: 435.87054979999493 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.149559009994218,
            "unit": "iter/sec",
            "range": "stddev: 0.05669529136825332",
            "extra": "mean: 465.2116994000039 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.228966571817372,
            "unit": "iter/sec",
            "range": "stddev: 0.05957539441289142",
            "extra": "mean: 448.6384016000102 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 27.3471018359872,
            "unit": "iter/sec",
            "range": "stddev: 0.001164078142774967",
            "extra": "mean: 36.566946142865426 msec\nrounds: 7"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6076.43892265256,
            "unit": "iter/sec",
            "range": "stddev: 0.000006926538746814151",
            "extra": "mean: 164.57007348038445 usec\nrounds: 2368"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4353.981261215201,
            "unit": "iter/sec",
            "range": "stddev: 0.00000692508403042643",
            "extra": "mean: 229.67485159109276 usec\nrounds: 2702"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1052.650134832757,
            "unit": "iter/sec",
            "range": "stddev: 0.000008908540059140956",
            "extra": "mean: 949.9832536086437 usec\nrounds: 970"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 563.5537771913278,
            "unit": "iter/sec",
            "range": "stddev: 0.00003900390569202796",
            "extra": "mean: 1.7744535490186197 msec\nrounds: 510"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1791.586100694125,
            "unit": "iter/sec",
            "range": "stddev: 0.000008321641100601255",
            "extra": "mean: 558.1646339032012 usec\nrounds: 1404"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 756.7301223002269,
            "unit": "iter/sec",
            "range": "stddev: 0.000013965589809707072",
            "extra": "mean: 1.32147508144688 msec\nrounds: 442"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1270.991734798704,
            "unit": "iter/sec",
            "range": "stddev: 0.000012268065791548523",
            "extra": "mean: 786.7871777768696 usec\nrounds: 855"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "chren@linkedin.com",
            "name": "Cheng Ren",
            "username": "burgerkingeater"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4d3aa3eab6517d28c07a12ded4cd46bb3a49948f",
          "message": "rename testing data files (#1278)",
          "timestamp": "2021-01-20T00:07:14+05:30",
          "tree_id": "483e52b8c0e2b59d5b1b47aa4fc7493770d1f647",
          "url": "https://github.com/tensorflow/io/commit/4d3aa3eab6517d28c07a12ded4cd46bb3a49948f"
        },
        "date": 1611081917678,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.28260224102601,
            "unit": "iter/sec",
            "range": "stddev: 0.04015897433526243",
            "extra": "mean: 233.50288999998838 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 33.182630333663006,
            "unit": "iter/sec",
            "range": "stddev: 0.0012966740060090473",
            "extra": "mean: 30.13624869230223 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4786944679412632,
            "unit": "iter/sec",
            "range": "stddev: 0.050096800833479393",
            "extra": "mean: 676.272226399999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4820813423775443,
            "unit": "iter/sec",
            "range": "stddev: 0.0504575731090235",
            "extra": "mean: 674.7267989999841 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.466488811607531,
            "unit": "iter/sec",
            "range": "stddev: 0.049950882709782596",
            "extra": "mean: 681.90087239999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5960557646779981,
            "unit": "iter/sec",
            "range": "stddev: 0.0486112524688005",
            "extra": "mean: 1.6776953756000013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.46219056297554045,
            "unit": "iter/sec",
            "range": "stddev: 0.051674132577297485",
            "extra": "mean: 2.1636097318 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8222184932969135,
            "unit": "iter/sec",
            "range": "stddev: 0.007626275372994539",
            "extra": "mean: 1.2162217319999968 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 4.024816509863019,
            "unit": "iter/sec",
            "range": "stddev: 0.049363989535797986",
            "extra": "mean: 248.45853159999933 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.3324727196300445,
            "unit": "iter/sec",
            "range": "stddev: 0.05810859791801429",
            "extra": "mean: 428.72955879998926 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.2945266484316185,
            "unit": "iter/sec",
            "range": "stddev: 0.057361727772907906",
            "extra": "mean: 435.81973679997645 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.268693861938939,
            "unit": "iter/sec",
            "range": "stddev: 0.05683421202925893",
            "extra": "mean: 440.7822565999936 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 28.778313731036395,
            "unit": "iter/sec",
            "range": "stddev: 0.0008814545651174483",
            "extra": "mean: 34.7483875999842 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6134.563099021071,
            "unit": "iter/sec",
            "range": "stddev: 0.000007012746809444979",
            "extra": "mean: 163.01079373681495 usec\nrounds: 2395"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4389.625880194023,
            "unit": "iter/sec",
            "range": "stddev: 0.000006913446606308615",
            "extra": "mean: 227.80984696486246 usec\nrounds: 2751"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1055.926643606572,
            "unit": "iter/sec",
            "range": "stddev: 0.000009174531453744479",
            "extra": "mean: 947.0354840033664 usec\nrounds: 969"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 551.552751797476,
            "unit": "iter/sec",
            "range": "stddev: 0.000018589413433737387",
            "extra": "mean: 1.8130632051803972 msec\nrounds: 502"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1720.739442449022,
            "unit": "iter/sec",
            "range": "stddev: 0.000009221402571140056",
            "extra": "mean: 581.1455095007074 usec\nrounds: 1421"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 764.5158810651453,
            "unit": "iter/sec",
            "range": "stddev: 0.000024925401861750666",
            "extra": "mean: 1.3080173018862231 msec\nrounds: 318"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1278.0707794472335,
            "unit": "iter/sec",
            "range": "stddev: 0.000015414119396354645",
            "extra": "mean: 782.4292801940912 usec\nrounds: 828"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "chren@linkedin.com",
            "name": "Cheng Ren",
            "username": "burgerkingeater"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4d3aa3eab6517d28c07a12ded4cd46bb3a49948f",
          "message": "rename testing data files (#1278)",
          "timestamp": "2021-01-20T00:07:14+05:30",
          "tree_id": "483e52b8c0e2b59d5b1b47aa4fc7493770d1f647",
          "url": "https://github.com/tensorflow/io/commit/4d3aa3eab6517d28c07a12ded4cd46bb3a49948f"
        },
        "date": 1611082049924,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.685553398819845,
            "unit": "iter/sec",
            "range": "stddev: 0.051782958496475455",
            "extra": "mean: 271.32967339998686 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 32.778317250051344,
            "unit": "iter/sec",
            "range": "stddev: 0.001017786202455431",
            "extra": "mean: 30.507972461534266 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.43143914430635,
            "unit": "iter/sec",
            "range": "stddev: 0.0519565306094775",
            "extra": "mean: 698.5976344000164 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.450319581066903,
            "unit": "iter/sec",
            "range": "stddev: 0.05953554814967176",
            "extra": "mean: 689.5032053999898 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4349504264849098,
            "unit": "iter/sec",
            "range": "stddev: 0.059476476001616145",
            "extra": "mean: 696.8881862000103 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5920668131237585,
            "unit": "iter/sec",
            "range": "stddev: 0.065845601347651",
            "extra": "mean: 1.688998568799991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4676242384710408,
            "unit": "iter/sec",
            "range": "stddev: 0.0443501887528977",
            "extra": "mean: 2.1384691333999966 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8122249721975802,
            "unit": "iter/sec",
            "range": "stddev: 0.008495388552088883",
            "extra": "mean: 1.2311859819999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.9770898624885436,
            "unit": "iter/sec",
            "range": "stddev: 0.054731311097692734",
            "extra": "mean: 251.4401320000047 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.3279668239734974,
            "unit": "iter/sec",
            "range": "stddev: 0.06318819512800093",
            "extra": "mean: 429.5593861999919 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.286245452386338,
            "unit": "iter/sec",
            "range": "stddev: 0.060720317147140684",
            "extra": "mean: 437.39835499999344 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.2659388063957793,
            "unit": "iter/sec",
            "range": "stddev: 0.05856358361384559",
            "extra": "mean: 441.31818439996096 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 27.968221321079152,
            "unit": "iter/sec",
            "range": "stddev: 0.0008896976920070607",
            "extra": "mean: 35.754865800004154 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6126.032836291362,
            "unit": "iter/sec",
            "range": "stddev: 0.000006554353424919431",
            "extra": "mean: 163.2377799341653 usec\nrounds: 2422"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4395.147580207181,
            "unit": "iter/sec",
            "range": "stddev: 0.000007140855275396703",
            "extra": "mean: 227.52364550927356 usec\nrounds: 2694"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1053.710475817798,
            "unit": "iter/sec",
            "range": "stddev: 0.000008871117326114988",
            "extra": "mean: 949.0272925529069 usec\nrounds: 940"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 563.1525880550522,
            "unit": "iter/sec",
            "range": "stddev: 0.00004016326530472552",
            "extra": "mean: 1.775717667308745 msec\nrounds: 517"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1795.8042864496601,
            "unit": "iter/sec",
            "range": "stddev: 0.000007825151508666196",
            "extra": "mean: 556.853554446637 usec\nrounds: 1405"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 764.2222062414669,
            "unit": "iter/sec",
            "range": "stddev: 0.00001411272811962357",
            "extra": "mean: 1.3085199459436223 msec\nrounds: 444"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1277.2301924915162,
            "unit": "iter/sec",
            "range": "stddev: 0.000012265441031454067",
            "extra": "mean: 782.944222489199 usec\nrounds: 836"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "chren@linkedin.com",
            "name": "Cheng Ren",
            "username": "burgerkingeater"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "33ad81f0dfbf1ab4b8ce08b29bc9ccea5a54e38e",
          "message": "Add tutorial for avro dataset API (#1250)",
          "timestamp": "2021-01-19T15:02:21-08:00",
          "tree_id": "9e71f18d6910d8e2ae667ff3fdd54dd407a8adb0",
          "url": "https://github.com/tensorflow/io/commit/33ad81f0dfbf1ab4b8ce08b29bc9ccea5a54e38e"
        },
        "date": 1611097674801,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.222063364367316,
            "unit": "iter/sec",
            "range": "stddev: 0.04507548324080852",
            "extra": "mean: 236.85101659999646 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 32.79405527529656,
            "unit": "iter/sec",
            "range": "stddev: 0.0010898145562340967",
            "extra": "mean: 30.49333153845387 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4265828137038183,
            "unit": "iter/sec",
            "range": "stddev: 0.05960143292625085",
            "extra": "mean: 700.9757796000031 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4322903906227633,
            "unit": "iter/sec",
            "range": "stddev: 0.05916641958510352",
            "extra": "mean: 698.1824401999916 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4180475909534225,
            "unit": "iter/sec",
            "range": "stddev: 0.05960209065703695",
            "extra": "mean: 705.1949499999864 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5829585999085518,
            "unit": "iter/sec",
            "range": "stddev: 0.0661950691709888",
            "extra": "mean: 1.7153876796000076 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.44954181551381467,
            "unit": "iter/sec",
            "range": "stddev: 0.06420439059071958",
            "extra": "mean: 2.2244871678000093 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.759944882407512,
            "unit": "iter/sec",
            "range": "stddev: 0.011519420433909194",
            "extra": "mean: 1.3158849057999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.8022818112158245,
            "unit": "iter/sec",
            "range": "stddev: 0.06557958170843056",
            "extra": "mean: 262.9999694000162 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.210827039930488,
            "unit": "iter/sec",
            "range": "stddev: 0.07599908830798588",
            "extra": "mean: 452.3194179999905 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.1794039683908215,
            "unit": "iter/sec",
            "range": "stddev: 0.0752263074226225",
            "extra": "mean: 458.8410475999808 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.135943203031804,
            "unit": "iter/sec",
            "range": "stddev: 0.07754869118568845",
            "extra": "mean: 468.177242999991 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 24.175292648799715,
            "unit": "iter/sec",
            "range": "stddev: 0.00033194066648933163",
            "extra": "mean: 41.364545800013275 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6120.270387067691,
            "unit": "iter/sec",
            "range": "stddev: 0.00000832631014529124",
            "extra": "mean: 163.39147402915873 usec\nrounds: 2137"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4415.8336093984535,
            "unit": "iter/sec",
            "range": "stddev: 0.000008061778931031407",
            "extra": "mean: 226.45780807312278 usec\nrounds: 2725"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1052.1584078429935,
            "unit": "iter/sec",
            "range": "stddev: 0.000011552653469404465",
            "extra": "mean: 950.4272289664801 usec\nrounds: 939"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 555.1626164633007,
            "unit": "iter/sec",
            "range": "stddev: 0.00004172074169006323",
            "extra": "mean: 1.80127402376364 msec\nrounds: 505"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1752.4190678371237,
            "unit": "iter/sec",
            "range": "stddev: 0.00000926246019360223",
            "extra": "mean: 570.6397621170735 usec\nrounds: 1341"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 769.8958681692162,
            "unit": "iter/sec",
            "range": "stddev: 0.000016149559315791203",
            "extra": "mean: 1.2988769538119003 msec\nrounds: 433"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1272.8750211750262,
            "unit": "iter/sec",
            "range": "stddev: 0.000014143373939521619",
            "extra": "mean: 785.6230842497579 usec\nrounds: 819"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "chren@linkedin.com",
            "name": "Cheng Ren",
            "username": "burgerkingeater"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "33ad81f0dfbf1ab4b8ce08b29bc9ccea5a54e38e",
          "message": "Add tutorial for avro dataset API (#1250)",
          "timestamp": "2021-01-19T15:02:21-08:00",
          "tree_id": "9e71f18d6910d8e2ae667ff3fdd54dd407a8adb0",
          "url": "https://github.com/tensorflow/io/commit/33ad81f0dfbf1ab4b8ce08b29bc9ccea5a54e38e"
        },
        "date": 1611097756962,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.092620805377086,
            "unit": "iter/sec",
            "range": "stddev: 0.05287912797134335",
            "extra": "mean: 244.34220700001106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.25709813223348,
            "unit": "iter/sec",
            "range": "stddev: 0.00034830034593168066",
            "extra": "mean: 34.17973975000166 msec\nrounds: 12"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4195476823390694,
            "unit": "iter/sec",
            "range": "stddev: 0.06705886991163983",
            "extra": "mean: 704.4497429999979 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4147181731340064,
            "unit": "iter/sec",
            "range": "stddev: 0.06775460018344351",
            "extra": "mean: 706.8545657999948 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.396281969202143,
            "unit": "iter/sec",
            "range": "stddev: 0.06770415107071781",
            "extra": "mean: 716.1877200000049 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5772425256231559,
            "unit": "iter/sec",
            "range": "stddev: 0.07518031809427844",
            "extra": "mean: 1.7323740985999962 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.44911953839046853,
            "unit": "iter/sec",
            "range": "stddev: 0.06666325904105164",
            "extra": "mean: 2.2265787046000014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7454185839222175,
            "unit": "iter/sec",
            "range": "stddev: 0.013293820267444776",
            "extra": "mean: 1.3415281313999912 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.796597951189736,
            "unit": "iter/sec",
            "range": "stddev: 0.06826151403634845",
            "extra": "mean: 263.3937048000121 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.1833400651165276,
            "unit": "iter/sec",
            "range": "stddev: 0.0791382284761269",
            "extra": "mean: 458.0138550000129 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.019170862683516,
            "unit": "iter/sec",
            "range": "stddev: 0.08130165724865358",
            "extra": "mean: 495.2527883999778 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.114344452490243,
            "unit": "iter/sec",
            "range": "stddev: 0.079986750412901",
            "extra": "mean: 472.9598334000002 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 24.509718532319717,
            "unit": "iter/sec",
            "range": "stddev: 0.0012206543014187325",
            "extra": "mean: 40.80014214285451 msec\nrounds: 7"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5989.054092256668,
            "unit": "iter/sec",
            "range": "stddev: 0.000009084843750706472",
            "extra": "mean: 166.97127536265103 usec\nrounds: 2208"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4378.232164302139,
            "unit": "iter/sec",
            "range": "stddev: 0.000008002661460482683",
            "extra": "mean: 228.40268913866365 usec\nrounds: 2670"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1047.5137038488242,
            "unit": "iter/sec",
            "range": "stddev: 0.00001038999892604951",
            "extra": "mean: 954.6414489144655 usec\nrounds: 920"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 545.0907451696094,
            "unit": "iter/sec",
            "range": "stddev: 0.000015055571388517012",
            "extra": "mean: 1.834556922607156 msec\nrounds: 491"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1703.923669238419,
            "unit": "iter/sec",
            "range": "stddev: 0.00001100849826384062",
            "extra": "mean: 586.8807494451656 usec\nrounds: 1353"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 760.2243930429518,
            "unit": "iter/sec",
            "range": "stddev: 0.000016590169197092526",
            "extra": "mean: 1.3154010962438312 msec\nrounds: 426"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1270.209414830905,
            "unit": "iter/sec",
            "range": "stddev: 0.0000177583177104742",
            "extra": "mean: 787.271758754145 usec\nrounds: 771"
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
          "id": "171b826db86c7ea3792beb4ebde34cd5f1040521",
          "message": "remove docker based mongodb tests in macos (#1279)",
          "timestamp": "2021-01-20T08:40:36-08:00",
          "tree_id": "9efab47cc944423e5f301267aaaa1484f2fbadbd",
          "url": "https://github.com/tensorflow/io/commit/171b826db86c7ea3792beb4ebde34cd5f1040521"
        },
        "date": 1611162002721,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.227392388067871,
            "unit": "iter/sec",
            "range": "stddev: 0.04009915851522333",
            "extra": "mean: 309.84766640001453 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 24.64533365855172,
            "unit": "iter/sec",
            "range": "stddev: 0.003117135925691325",
            "extra": "mean: 40.5756324444408 msec\nrounds: 9"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.2700745403598486,
            "unit": "iter/sec",
            "range": "stddev: 0.05910303520850433",
            "extra": "mean: 787.3553623999669 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.2661453746108222,
            "unit": "iter/sec",
            "range": "stddev: 0.059810890161288606",
            "extra": "mean: 789.7987230000126 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.226440810651344,
            "unit": "iter/sec",
            "range": "stddev: 0.05059458876339837",
            "extra": "mean: 815.3675182000143 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.4433313882091816,
            "unit": "iter/sec",
            "range": "stddev: 0.12044405927213124",
            "extra": "mean: 2.2556489944000075 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.35902357178616523,
            "unit": "iter/sec",
            "range": "stddev: 0.17585069937725462",
            "extra": "mean: 2.785332436600015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7061241365246212,
            "unit": "iter/sec",
            "range": "stddev: 0.009233634076940524",
            "extra": "mean: 1.4161815866000098 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.459463128809259,
            "unit": "iter/sec",
            "range": "stddev: 0.05786197035666411",
            "extra": "mean: 289.062193399991 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.005645741938971,
            "unit": "iter/sec",
            "range": "stddev: 0.06719139078219823",
            "extra": "mean: 498.59253760000684 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.7401195816997872,
            "unit": "iter/sec",
            "range": "stddev: 0.06994109462833023",
            "extra": "mean: 574.6731491999981 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.8019123424683348,
            "unit": "iter/sec",
            "range": "stddev: 0.07302928383285938",
            "extra": "mean: 554.9659527999893 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 21.262648352154564,
            "unit": "iter/sec",
            "range": "stddev: 0.0027310359134360062",
            "extra": "mean: 47.03083000000182 msec\nrounds: 6"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5298.67136360949,
            "unit": "iter/sec",
            "range": "stddev: 0.00000864039987650291",
            "extra": "mean: 188.72655640956629 usec\nrounds: 2216"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3783.7256316653525,
            "unit": "iter/sec",
            "range": "stddev: 0.000008314705228733105",
            "extra": "mean: 264.2897760955951 usec\nrounds: 2510"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 883.7472881820798,
            "unit": "iter/sec",
            "range": "stddev: 0.000010506968149323942",
            "extra": "mean: 1.131545197787321 msec\nrounds: 814"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 470.87934939766944,
            "unit": "iter/sec",
            "range": "stddev: 0.000027273359915997377",
            "extra": "mean: 2.1236862505844885 msec\nrounds: 431"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1436.4727583752856,
            "unit": "iter/sec",
            "range": "stddev: 0.0000437062260683349",
            "extra": "mean: 696.1496444464734 usec\nrounds: 1170"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 655.24902749891,
            "unit": "iter/sec",
            "range": "stddev: 0.00003733400170790281",
            "extra": "mean: 1.5261373279972759 msec\nrounds: 375"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1093.5175270415552,
            "unit": "iter/sec",
            "range": "stddev: 0.000026124053976047454",
            "extra": "mean: 914.4800840142352 usec\nrounds: 738"
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
          "id": "171b826db86c7ea3792beb4ebde34cd5f1040521",
          "message": "remove docker based mongodb tests in macos (#1279)",
          "timestamp": "2021-01-20T08:40:36-08:00",
          "tree_id": "9efab47cc944423e5f301267aaaa1484f2fbadbd",
          "url": "https://github.com/tensorflow/io/commit/171b826db86c7ea3792beb4ebde34cd5f1040521"
        },
        "date": 1611162066015,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.568623793714776,
            "unit": "iter/sec",
            "range": "stddev: 0.052788827524655996",
            "extra": "mean: 280.2200674000005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 25.542494070958202,
            "unit": "iter/sec",
            "range": "stddev: 0.001178950035870877",
            "extra": "mean: 39.15044463637555 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.2563631675515077,
            "unit": "iter/sec",
            "range": "stddev: 0.06428232710509948",
            "extra": "mean: 795.9481985999901 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.2908305922713168,
            "unit": "iter/sec",
            "range": "stddev: 0.0480773641471506",
            "extra": "mean: 774.6949956000208 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.255865668482878,
            "unit": "iter/sec",
            "range": "stddev: 0.05739473536829998",
            "extra": "mean: 796.2635057999705 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.49908630548207444,
            "unit": "iter/sec",
            "range": "stddev: 0.09568503964299635",
            "extra": "mean: 2.003661469000008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.40350596418949397,
            "unit": "iter/sec",
            "range": "stddev: 0.06838502247429842",
            "extra": "mean: 2.4782781142000205 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7149529134513661,
            "unit": "iter/sec",
            "range": "stddev: 0.06571782966418843",
            "extra": "mean: 1.3986935099999755 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.641229261404558,
            "unit": "iter/sec",
            "range": "stddev: 0.007427754280481903",
            "extra": "mean: 274.63252879997526 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.021847439676933,
            "unit": "iter/sec",
            "range": "stddev: 0.06555094816439547",
            "extra": "mean: 494.5971591999978 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.9263074852026887,
            "unit": "iter/sec",
            "range": "stddev: 0.06515698854749447",
            "extra": "mean: 519.1279209999948 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.9109655935782612,
            "unit": "iter/sec",
            "range": "stddev: 0.06390775141575991",
            "extra": "mean: 523.2956591999709 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 20.554370205961856,
            "unit": "iter/sec",
            "range": "stddev: 0.0014639679124157754",
            "extra": "mean: 48.65145416666413 msec\nrounds: 6"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5264.302465066802,
            "unit": "iter/sec",
            "range": "stddev: 0.000009103281856271525",
            "extra": "mean: 189.95868999470767 usec\nrounds: 2129"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3735.2418804666895,
            "unit": "iter/sec",
            "range": "stddev: 0.000009909378730138902",
            "extra": "mean: 267.7202794361092 usec\nrounds: 2344"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 877.4935268830267,
            "unit": "iter/sec",
            "range": "stddev: 0.000015219357839636094",
            "extra": "mean: 1.1396095462403382 msec\nrounds: 811"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 457.4260961261787,
            "unit": "iter/sec",
            "range": "stddev: 0.000016566380673322343",
            "extra": "mean: 2.1861454964391775 msec\nrounds: 421"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1436.5065950326175,
            "unit": "iter/sec",
            "range": "stddev: 0.00004041967835365196",
            "extra": "mean: 696.1332467654239 usec\nrounds: 1159"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 658.531249609449,
            "unit": "iter/sec",
            "range": "stddev: 0.000032384469194429654",
            "extra": "mean: 1.518530822330852 msec\nrounds: 394"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1103.7638551197317,
            "unit": "iter/sec",
            "range": "stddev: 0.00001603747730499637",
            "extra": "mean: 905.9908923104972 usec\nrounds: 715"
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
          "id": "5a507a337137c14b26e52cd94b7d59e3eed6587d",
          "message": "trigger benchmarks workflow only on commits (#1282)",
          "timestamp": "2021-01-25T17:39:30-08:00",
          "tree_id": "a8d73beb997452f9d6dc38f394c382d166ff567f",
          "url": "https://github.com/tensorflow/io/commit/5a507a337137c14b26e52cd94b7d59e3eed6587d"
        },
        "date": 1611625565141,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.025283410503964,
            "unit": "iter/sec",
            "range": "stddev: 0.04079260438644628",
            "extra": "mean: 248.42971240000224 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 30.06671028794909,
            "unit": "iter/sec",
            "range": "stddev: 0.0017688829131217835",
            "extra": "mean: 33.259375250002186 msec\nrounds: 12"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4176267992371057,
            "unit": "iter/sec",
            "range": "stddev: 0.05968433423958773",
            "extra": "mean: 705.4042717999891 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.404360548829018,
            "unit": "iter/sec",
            "range": "stddev: 0.0546095690767885",
            "extra": "mean: 712.0678524000255 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.400278428450159,
            "unit": "iter/sec",
            "range": "stddev: 0.057468424197361395",
            "extra": "mean: 714.1436871999872 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5433328844787939,
            "unit": "iter/sec",
            "range": "stddev: 0.06764509433361944",
            "extra": "mean: 1.8404923179999968 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4179011804644528,
            "unit": "iter/sec",
            "range": "stddev: 0.06984873265116572",
            "extra": "mean: 2.392910206400006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7755602236008337,
            "unit": "iter/sec",
            "range": "stddev: 0.05567409055823743",
            "extra": "mean: 1.2893905200000062 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 4.252665503757994,
            "unit": "iter/sec",
            "range": "stddev: 0.003056292950925888",
            "extra": "mean: 235.1466390000155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.2194371885925723,
            "unit": "iter/sec",
            "range": "stddev: 0.06441678385746026",
            "extra": "mean: 450.56467699999985 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.025222495216551,
            "unit": "iter/sec",
            "range": "stddev: 0.0633939299305674",
            "extra": "mean: 493.77290759999823 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.1027842196654762,
            "unit": "iter/sec",
            "range": "stddev: 0.057612075382976316",
            "extra": "mean: 475.55996979998554 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 24.411029341069455,
            "unit": "iter/sec",
            "range": "stddev: 0.0013500277215144536",
            "extra": "mean: 40.9650894285554 msec\nrounds: 7"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5802.639257381739,
            "unit": "iter/sec",
            "range": "stddev: 0.000015456416087437107",
            "extra": "mean: 172.335372861214 usec\nrounds: 2513"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4037.012660059306,
            "unit": "iter/sec",
            "range": "stddev: 0.00001934296881865247",
            "extra": "mean: 247.70791776147396 usec\nrounds: 2590"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 956.3669515344106,
            "unit": "iter/sec",
            "range": "stddev: 0.00007428274867103747",
            "extra": "mean: 1.0456237518408429 msec\nrounds: 951"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 502.40331607105395,
            "unit": "iter/sec",
            "range": "stddev: 0.00011155156365362546",
            "extra": "mean: 1.9904327221012446 msec\nrounds: 457"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1604.944395385139,
            "unit": "iter/sec",
            "range": "stddev: 0.00004981463580741062",
            "extra": "mean: 623.0745456823317 usec\nrounds: 1204"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 683.1558684064366,
            "unit": "iter/sec",
            "range": "stddev: 0.00007229285422684864",
            "extra": "mean: 1.4637947886368157 msec\nrounds: 440"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1143.0591111486247,
            "unit": "iter/sec",
            "range": "stddev: 0.00004639198856325617",
            "extra": "mean: 874.8453953489168 usec\nrounds: 774"
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
          "id": "5a507a337137c14b26e52cd94b7d59e3eed6587d",
          "message": "trigger benchmarks workflow only on commits (#1282)",
          "timestamp": "2021-01-25T17:39:30-08:00",
          "tree_id": "a8d73beb997452f9d6dc38f394c382d166ff567f",
          "url": "https://github.com/tensorflow/io/commit/5a507a337137c14b26e52cd94b7d59e3eed6587d"
        },
        "date": 1611625671001,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.7035859461310565,
            "unit": "iter/sec",
            "range": "stddev: 0.040391208134315765",
            "extra": "mean: 270.0085847999958 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.300992382520235,
            "unit": "iter/sec",
            "range": "stddev: 0.0014390021150474512",
            "extra": "mean: 34.12853690909659 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.325305229755303,
            "unit": "iter/sec",
            "range": "stddev: 0.06258538757458752",
            "extra": "mean: 754.5431629999939 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.3385197153058543,
            "unit": "iter/sec",
            "range": "stddev: 0.055691526634973496",
            "extra": "mean: 747.0939639999983 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.3182158633234946,
            "unit": "iter/sec",
            "range": "stddev: 0.0519100942992526",
            "extra": "mean: 758.6010969999961 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5085420705091803,
            "unit": "iter/sec",
            "range": "stddev: 0.057772872429911776",
            "extra": "mean: 1.9664056486000163 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.3887655452098019,
            "unit": "iter/sec",
            "range": "stddev: 0.0586570551051079",
            "extra": "mean: 2.572244408800009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7434265967549694,
            "unit": "iter/sec",
            "range": "stddev: 0.06024569066409846",
            "extra": "mean: 1.345122712000034 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.987370368007586,
            "unit": "iter/sec",
            "range": "stddev: 0.002362767787284188",
            "extra": "mean: 250.79185219999545 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.106105336655337,
            "unit": "iter/sec",
            "range": "stddev: 0.06207319268264486",
            "extra": "mean: 474.8100594000107 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.0451887171239735,
            "unit": "iter/sec",
            "range": "stddev: 0.06851673779215506",
            "extra": "mean: 488.95243340000434 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.009610199538889,
            "unit": "iter/sec",
            "range": "stddev: 0.06271949928964034",
            "extra": "mean: 497.6089394000155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 25.941279223987635,
            "unit": "iter/sec",
            "range": "stddev: 0.0009147095345985145",
            "extra": "mean: 38.548600142868445 msec\nrounds: 7"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5499.888661073788,
            "unit": "iter/sec",
            "range": "stddev: 0.000012768783231352153",
            "extra": "mean: 181.82186251835174 usec\nrounds: 2153"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3901.325837423209,
            "unit": "iter/sec",
            "range": "stddev: 0.000013981472982445399",
            "extra": "mean: 256.32311723557325 usec\nrounds: 2576"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 922.3083868994569,
            "unit": "iter/sec",
            "range": "stddev: 0.000047275246610692246",
            "extra": "mean: 1.0842360475130457 msec\nrounds: 884"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 488.23269673280737,
            "unit": "iter/sec",
            "range": "stddev: 0.00009554585018243606",
            "extra": "mean: 2.0482036674148945 msec\nrounds: 445"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1551.074595578866,
            "unit": "iter/sec",
            "range": "stddev: 0.0000427445671281326",
            "extra": "mean: 644.7143179640544 usec\nrounds: 1258"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 689.5575802224608,
            "unit": "iter/sec",
            "range": "stddev: 0.00006826146132586164",
            "extra": "mean: 1.4502052166222088 msec\nrounds: 397"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1139.119222405833,
            "unit": "iter/sec",
            "range": "stddev: 0.000035501458101574094",
            "extra": "mean: 877.8712362416187 usec\nrounds: 745"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4e8fa4fe9b213e9eb1007710748b5e2cd03eb173",
          "message": "Bump Apache Arrow to 3.0.0 (#1285)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-27T11:18:15+05:30",
          "tree_id": "e1d68828e7bd83e2575db40840d0d83f39249fad",
          "url": "https://github.com/tensorflow/io/commit/4e8fa4fe9b213e9eb1007710748b5e2cd03eb173"
        },
        "date": 1611726900439,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.333917280144807,
            "unit": "iter/sec",
            "range": "stddev: 0.04294244718847738",
            "extra": "mean: 230.73813720011458 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 31.86122160975409,
            "unit": "iter/sec",
            "range": "stddev: 0.0012376277102473716",
            "extra": "mean: 31.38611608331606 msec\nrounds: 12"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4727624168391975,
            "unit": "iter/sec",
            "range": "stddev: 0.05159230054328557",
            "extra": "mean: 678.9961425999536 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4856570175923072,
            "unit": "iter/sec",
            "range": "stddev: 0.04790798215740573",
            "extra": "mean: 673.1028684000194 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4546264591488276,
            "unit": "iter/sec",
            "range": "stddev: 0.05266846867953408",
            "extra": "mean: 687.4617148000652 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.6043205741113884,
            "unit": "iter/sec",
            "range": "stddev: 0.05510789046959572",
            "extra": "mean: 1.6547508770000605 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.46424041779102887,
            "unit": "iter/sec",
            "range": "stddev: 0.056804272626925806",
            "extra": "mean: 2.154056307200153 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.8237818587849407,
            "unit": "iter/sec",
            "range": "stddev: 0.0564040272237694",
            "extra": "mean: 1.2139135978000013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 4.378090509412815,
            "unit": "iter/sec",
            "range": "stddev: 0.00025592692052194536",
            "extra": "mean: 228.41007920005723 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.3076333348798634,
            "unit": "iter/sec",
            "range": "stddev: 0.06274239606209946",
            "extra": "mean: 433.3444073999999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.156901851165851,
            "unit": "iter/sec",
            "range": "stddev: 0.06146499603593638",
            "extra": "mean: 463.62795760014706 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.2336785600149445,
            "unit": "iter/sec",
            "range": "stddev: 0.06141951547544438",
            "extra": "mean: 447.6919901999281 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 25.794154101346727,
            "unit": "iter/sec",
            "range": "stddev: 0.0005720333069642013",
            "extra": "mean: 38.76847428572156 msec\nrounds: 7"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6121.770907519643,
            "unit": "iter/sec",
            "range": "stddev: 0.000007516349301415567",
            "extra": "mean: 163.35142479305387 usec\nrounds: 2380"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4373.915434376442,
            "unit": "iter/sec",
            "range": "stddev: 0.000007237894770526874",
            "extra": "mean: 228.62810564204767 usec\nrounds: 2783"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1050.1974100274965,
            "unit": "iter/sec",
            "range": "stddev: 0.000010585572400160674",
            "extra": "mean: 952.201929324714 usec\nrounds: 962"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 557.4374889944188,
            "unit": "iter/sec",
            "range": "stddev: 0.000048604539770342894",
            "extra": "mean: 1.7939231209654296 msec\nrounds: 496"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1756.0255531652058,
            "unit": "iter/sec",
            "range": "stddev: 0.000009135409816837654",
            "extra": "mean: 569.4677951568058 usec\nrounds: 1362"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 769.6075523668126,
            "unit": "iter/sec",
            "range": "stddev: 0.000024922611650761188",
            "extra": "mean: 1.2993635482456611 msec\nrounds: 456"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1291.2641367976992,
            "unit": "iter/sec",
            "range": "stddev: 0.000012943288614085842",
            "extra": "mean: 774.4348901999041 usec\nrounds: 847"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4e8fa4fe9b213e9eb1007710748b5e2cd03eb173",
          "message": "Bump Apache Arrow to 3.0.0 (#1285)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-27T11:18:15+05:30",
          "tree_id": "e1d68828e7bd83e2575db40840d0d83f39249fad",
          "url": "https://github.com/tensorflow/io/commit/4e8fa4fe9b213e9eb1007710748b5e2cd03eb173"
        },
        "date": 1611727048312,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.2554275003261446,
            "unit": "iter/sec",
            "range": "stddev: 0.04375345879805819",
            "extra": "mean: 307.17931820008744 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 27.040355929326726,
            "unit": "iter/sec",
            "range": "stddev: 0.0026986201092207072",
            "extra": "mean: 36.98176172730945 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 0.7954029542450997,
            "unit": "iter/sec",
            "range": "stddev: 0.03298428126025155",
            "extra": "mean: 1.25722439759993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 0.8444761097462444,
            "unit": "iter/sec",
            "range": "stddev: 0.04234933632024118",
            "extra": "mean: 1.1841661220001698 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 0.843303640310789,
            "unit": "iter/sec",
            "range": "stddev: 0.061136597043434465",
            "extra": "mean: 1.18581250240004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.39648787233233873,
            "unit": "iter/sec",
            "range": "stddev: 0.11328855817479348",
            "extra": "mean: 2.522145240199916 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.30496948852607236,
            "unit": "iter/sec",
            "range": "stddev: 0.08737069769548923",
            "extra": "mean: 3.2790165495998735 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.67185618398315,
            "unit": "iter/sec",
            "range": "stddev: 0.07663467984536601",
            "extra": "mean: 1.4884137763999206 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 2.4103491135193815,
            "unit": "iter/sec",
            "range": "stddev: 0.011883522671149468",
            "extra": "mean: 414.8776600000019 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 1.5557218598717077,
            "unit": "iter/sec",
            "range": "stddev: 0.06228717800143188",
            "extra": "mean: 642.7884223999172 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.4845997449032176,
            "unit": "iter/sec",
            "range": "stddev: 0.07270124854285176",
            "extra": "mean: 673.5822253998776 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.4529989019576104,
            "unit": "iter/sec",
            "range": "stddev: 0.05002128535686897",
            "extra": "mean: 688.2317658001739 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 23.48173013181853,
            "unit": "iter/sec",
            "range": "stddev: 0.0016065345337132144",
            "extra": "mean: 42.58629983337414 msec\nrounds: 6"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 3726.6105193347216,
            "unit": "iter/sec",
            "range": "stddev: 0.00010288554272301781",
            "extra": "mean: 268.34035776256036 usec\nrounds: 1951"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3057.189401195,
            "unit": "iter/sec",
            "range": "stddev: 0.0000823628158010849",
            "extra": "mean: 327.0978237753664 usec\nrounds: 1935"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 896.9925897551237,
            "unit": "iter/sec",
            "range": "stddev: 0.00015278482472383915",
            "extra": "mean: 1.114836411606251 msec\nrounds: 724"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 550.9625969232104,
            "unit": "iter/sec",
            "range": "stddev: 0.00021277406941447972",
            "extra": "mean: 1.8150052391657603 msec\nrounds: 485"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1477.5430165126445,
            "unit": "iter/sec",
            "range": "stddev: 0.00012067826878512595",
            "extra": "mean: 676.7992463327664 usec\nrounds: 1433"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 517.2813163800303,
            "unit": "iter/sec",
            "range": "stddev: 0.00017796976352461128",
            "extra": "mean: 1.9331840689667044 msec\nrounds: 377"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 905.6428289936067,
            "unit": "iter/sec",
            "range": "stddev: 0.00019527866956685477",
            "extra": "mean: 1.1041880617674051 msec\nrounds: 680"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d63a0b64f780eae74de8076d1d7f3ba1cf20d80",
          "message": "Add bazel cache (#1287)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-28T18:54:15-08:00",
          "tree_id": "3949b52771d3d2ff10300cef8a58bd917b5560be",
          "url": "https://github.com/tensorflow/io/commit/8d63a0b64f780eae74de8076d1d7f3ba1cf20d80"
        },
        "date": 1611889771570,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5895.885174185953,
            "unit": "iter/sec",
            "range": "stddev: 0.00003049363672459609",
            "extra": "mean: 169.60981607618749 usec\nrounds: 1381"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4272.333246158171,
            "unit": "iter/sec",
            "range": "stddev: 0.00003962843302690557",
            "extra": "mean: 234.06413834857906 usec\nrounds: 2725"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 998.6564049120869,
            "unit": "iter/sec",
            "range": "stddev: 0.00017543383465075087",
            "extra": "mean: 1.0013454027644588 msec\nrounds: 941"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 492.75207673740596,
            "unit": "iter/sec",
            "range": "stddev: 0.0002992996884700036",
            "extra": "mean: 2.0294181338030426 msec\nrounds: 568"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1732.034232942347,
            "unit": "iter/sec",
            "range": "stddev: 0.00008334661500662252",
            "extra": "mean: 577.355794118006 usec\nrounds: 1156"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 707.3622602217127,
            "unit": "iter/sec",
            "range": "stddev: 0.00018597732064439218",
            "extra": "mean: 1.4137027888462188 msec\nrounds: 251"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1232.4544629524194,
            "unit": "iter/sec",
            "range": "stddev: 0.00014014863874805931",
            "extra": "mean: 811.3890046731944 usec\nrounds: 642"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.086075010463643,
            "unit": "iter/sec",
            "range": "stddev: 0.03227417275238259",
            "extra": "mean: 244.7336373999974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 31.70047904788682,
            "unit": "iter/sec",
            "range": "stddev: 0.001660668312521688",
            "extra": "mean: 31.54526461538318 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4689000729392985,
            "unit": "iter/sec",
            "range": "stddev: 0.04582704531072623",
            "extra": "mean: 680.78150340001 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.418018103124201,
            "unit": "iter/sec",
            "range": "stddev: 0.04353341727430121",
            "extra": "mean: 705.2096145999712 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.3681613912685615,
            "unit": "iter/sec",
            "range": "stddev: 0.05293117267124334",
            "extra": "mean: 730.9079224000016 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5318768208556908,
            "unit": "iter/sec",
            "range": "stddev: 0.04902513685290087",
            "extra": "mean: 1.8801345740000215 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4034104031155706,
            "unit": "iter/sec",
            "range": "stddev: 0.05326165247471786",
            "extra": "mean: 2.47886517620002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7808830528044011,
            "unit": "iter/sec",
            "range": "stddev: 0.07901605522268157",
            "extra": "mean: 1.2806014887999937 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.893369568497964,
            "unit": "iter/sec",
            "range": "stddev: 0.05064403543897523",
            "extra": "mean: 256.8469246000177 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.27022102920077,
            "unit": "iter/sec",
            "range": "stddev: 0.05607144275964983",
            "extra": "mean: 440.48574439998447 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.1644423621776254,
            "unit": "iter/sec",
            "range": "stddev: 0.061364498911095446",
            "extra": "mean: 462.01276480003344 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.0510080711026513,
            "unit": "iter/sec",
            "range": "stddev: 0.06540400038984828",
            "extra": "mean: 487.56512180002574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 28.077348068090163,
            "unit": "iter/sec",
            "range": "stddev: 0.0016593053572774529",
            "extra": "mean: 35.61589924998998 msec\nrounds: 8"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8d63a0b64f780eae74de8076d1d7f3ba1cf20d80",
          "message": "Add bazel cache (#1287)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-28T18:54:15-08:00",
          "tree_id": "3949b52771d3d2ff10300cef8a58bd917b5560be",
          "url": "https://github.com/tensorflow/io/commit/8d63a0b64f780eae74de8076d1d7f3ba1cf20d80"
        },
        "date": 1611889886944,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5371.51520663421,
            "unit": "iter/sec",
            "range": "stddev: 0.00001017008947708267",
            "extra": "mean: 186.16721009463544 usec\nrounds: 1466"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3857.140171260802,
            "unit": "iter/sec",
            "range": "stddev: 0.000017334433505448644",
            "extra": "mean: 259.2594397919237 usec\nrounds: 2699"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 911.1881977096855,
            "unit": "iter/sec",
            "range": "stddev: 0.000025615016428978086",
            "extra": "mean: 1.0974681218584121 msec\nrounds: 796"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 484.67229437640185,
            "unit": "iter/sec",
            "range": "stddev: 0.00005444354543677985",
            "extra": "mean: 2.0632497702115176 msec\nrounds: 470"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1403.9001290092258,
            "unit": "iter/sec",
            "range": "stddev: 0.00004390996279344608",
            "extra": "mean: 712.3013805160982 usec\nrounds: 1201"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 672.2006936980843,
            "unit": "iter/sec",
            "range": "stddev: 0.000047849759878823746",
            "extra": "mean: 1.4876509491511254 msec\nrounds: 236"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1100.1504649972346,
            "unit": "iter/sec",
            "range": "stddev: 0.000022447412920754288",
            "extra": "mean: 908.966574860752 usec\nrounds: 708"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.628667720349503,
            "unit": "iter/sec",
            "range": "stddev: 0.03532010344212563",
            "extra": "mean: 275.58323800000153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.32909093898989,
            "unit": "iter/sec",
            "range": "stddev: 0.00047854594242053986",
            "extra": "mean: 34.09584027272414 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.283071963824548,
            "unit": "iter/sec",
            "range": "stddev: 0.05106118526701774",
            "extra": "mean: 779.3795111999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.297027486078087,
            "unit": "iter/sec",
            "range": "stddev: 0.051058841369300945",
            "extra": "mean: 770.9936841999934 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.2519575135067034,
            "unit": "iter/sec",
            "range": "stddev: 0.05921108412725552",
            "extra": "mean: 798.7491502000125 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.4822042691906347,
            "unit": "iter/sec",
            "range": "stddev: 0.07676346411043863",
            "extra": "mean: 2.0738099264000085 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.3895103774689985,
            "unit": "iter/sec",
            "range": "stddev: 0.05514902730768759",
            "extra": "mean: 2.567325693600014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7283153016754184,
            "unit": "iter/sec",
            "range": "stddev: 0.016488088956433604",
            "extra": "mean: 1.3730317044000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.4548117153767692,
            "unit": "iter/sec",
            "range": "stddev: 0.05517899126807633",
            "extra": "mean: 289.45137460000296 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 1.9914726891943102,
            "unit": "iter/sec",
            "range": "stddev: 0.059653367018792716",
            "extra": "mean: 502.1409560000393 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.886799876141649,
            "unit": "iter/sec",
            "range": "stddev: 0.059629308366263185",
            "extra": "mean: 529.9979148000148 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.8925457927543463,
            "unit": "iter/sec",
            "range": "stddev: 0.06044457398764239",
            "extra": "mean: 528.3887997999955 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 23.831565075556014,
            "unit": "iter/sec",
            "range": "stddev: 0.001479273123760496",
            "extra": "mean: 41.96115516667002 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72e19f3eddd512a691e080a7d1bb1dcac5aa6a24",
          "message": "Add initial bigtable stub test (#1286)\n\n* Add initial bigtable stub test\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>\r\n\r\n* Fix kokoro test\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-28T19:19:27-08:00",
          "tree_id": "f355df19545ebad9758a80963c7dd67358466ff3",
          "url": "https://github.com/tensorflow/io/commit/72e19f3eddd512a691e080a7d1bb1dcac5aa6a24"
        },
        "date": 1611890912373,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 3568.4460132729378,
            "unit": "iter/sec",
            "range": "stddev: 0.00006559857412388654",
            "extra": "mean: 280.2340279999953 usec\nrounds: 1250"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 2838.4171823971965,
            "unit": "iter/sec",
            "range": "stddev: 0.00007950407315745216",
            "extra": "mean: 352.30902849715915 usec\nrounds: 1544"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 812.0527230439685,
            "unit": "iter/sec",
            "range": "stddev: 0.0002248084478735081",
            "extra": "mean: 1.2314471359094934 msec\nrounds: 802"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 467.61507378805146,
            "unit": "iter/sec",
            "range": "stddev: 0.00018552780711231985",
            "extra": "mean: 2.138511044777086 msec\nrounds: 469"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1282.8121307926435,
            "unit": "iter/sec",
            "range": "stddev: 0.00010456171500589705",
            "extra": "mean: 779.5373741766105 usec\nrounds: 1216"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 474.72117964673504,
            "unit": "iter/sec",
            "range": "stddev: 0.00019948172716175344",
            "extra": "mean: 2.10649965258376 msec\nrounds: 213"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 815.6200778133755,
            "unit": "iter/sec",
            "range": "stddev: 0.00015378152124271408",
            "extra": "mean: 1.2260610389593831 msec\nrounds: 616"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 2.969312281686005,
            "unit": "iter/sec",
            "range": "stddev: 0.03427301486136752",
            "extra": "mean: 336.7783194000026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 24.976394559742523,
            "unit": "iter/sec",
            "range": "stddev: 0.0009032455134130853",
            "extra": "mean: 40.03780439999218 msec\nrounds: 10"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 0.7375296545204375,
            "unit": "iter/sec",
            "range": "stddev: 0.055496997507312976",
            "extra": "mean: 1.3558776842000042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 0.746952326907724,
            "unit": "iter/sec",
            "range": "stddev: 0.05840400396166687",
            "extra": "mean: 1.3387735254000177 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 0.736551378463494,
            "unit": "iter/sec",
            "range": "stddev: 0.05668077282703914",
            "extra": "mean: 1.3576785398000084 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.3599264735749872,
            "unit": "iter/sec",
            "range": "stddev: 0.07898742155334561",
            "extra": "mean: 2.7783452272000204 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.2766007898309999,
            "unit": "iter/sec",
            "range": "stddev: 0.1616760684733711",
            "extra": "mean: 3.615318671399996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.5457718446820642,
            "unit": "iter/sec",
            "range": "stddev: 0.013919849273023673",
            "extra": "mean: 1.8322674753999877 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 2.064631464514186,
            "unit": "iter/sec",
            "range": "stddev: 0.005489480662619648",
            "extra": "mean: 484.3479415999809 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 1.2350786660346884,
            "unit": "iter/sec",
            "range": "stddev: 0.07310921085137197",
            "extra": "mean: 809.6650257999954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.189642460714893,
            "unit": "iter/sec",
            "range": "stddev: 0.07783193100511954",
            "extra": "mean: 840.5886920000057 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.197044057422753,
            "unit": "iter/sec",
            "range": "stddev: 0.07905619864837037",
            "extra": "mean: 835.3911401999767 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 19.496156613453216,
            "unit": "iter/sec",
            "range": "stddev: 0.0006327319428839682",
            "extra": "mean: 51.29216080003971 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72e19f3eddd512a691e080a7d1bb1dcac5aa6a24",
          "message": "Add initial bigtable stub test (#1286)\n\n* Add initial bigtable stub test\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>\r\n\r\n* Fix kokoro test\r\n\r\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-28T19:19:27-08:00",
          "tree_id": "f355df19545ebad9758a80963c7dd67358466ff3",
          "url": "https://github.com/tensorflow/io/commit/72e19f3eddd512a691e080a7d1bb1dcac5aa6a24"
        },
        "date": 1611890925047,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5601.022786264571,
            "unit": "iter/sec",
            "range": "stddev: 0.000011783636140818022",
            "extra": "mean: 178.538820169114 usec\nrounds: 1418"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3914.5509112150244,
            "unit": "iter/sec",
            "range": "stddev: 0.0000116890943726943",
            "extra": "mean: 255.45714506740524 usec\nrounds: 2585"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 915.0371403879458,
            "unit": "iter/sec",
            "range": "stddev: 0.000040283953296992696",
            "extra": "mean: 1.092851815365694 msec\nrounds: 872"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 467.2455299359371,
            "unit": "iter/sec",
            "range": "stddev: 0.00006393190182152445",
            "extra": "mean: 2.1402023902446055 msec\nrounds: 451"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1512.9440216449996,
            "unit": "iter/sec",
            "range": "stddev: 0.00004501621815690067",
            "extra": "mean: 660.9629871914997 usec\nrounds: 1015"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 669.2522905071471,
            "unit": "iter/sec",
            "range": "stddev: 0.00003495665178305611",
            "extra": "mean: 1.494204822582256 msec\nrounds: 248"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1127.5416522036746,
            "unit": "iter/sec",
            "range": "stddev: 0.00003042271912097599",
            "extra": "mean: 886.8851966981385 usec\nrounds: 727"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.8051264285752606,
            "unit": "iter/sec",
            "range": "stddev: 0.0347262571948223",
            "extra": "mean: 262.80335719999357 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.812551495598395,
            "unit": "iter/sec",
            "range": "stddev: 0.0014091929310386156",
            "extra": "mean: 33.542918999993766 msec\nrounds: 12"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.3359884393500214,
            "unit": "iter/sec",
            "range": "stddev: 0.04480323279163516",
            "extra": "mean: 748.5094709999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.322827083298136,
            "unit": "iter/sec",
            "range": "stddev: 0.05076048305587164",
            "extra": "mean: 755.956702599974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.2739110541110652,
            "unit": "iter/sec",
            "range": "stddev: 0.05483358759747336",
            "extra": "mean: 784.9841609999999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5368634617606144,
            "unit": "iter/sec",
            "range": "stddev: 0.04899407773465712",
            "extra": "mean: 1.8626709977999893 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4149145456764406,
            "unit": "iter/sec",
            "range": "stddev: 0.05247472705914677",
            "extra": "mean: 2.4101348347999876 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7379231787368316,
            "unit": "iter/sec",
            "range": "stddev: 0.012788732253329362",
            "extra": "mean: 1.3551546133999863 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.5059831232278746,
            "unit": "iter/sec",
            "range": "stddev: 0.05729232119881468",
            "extra": "mean: 285.2267010000105 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.028424467851055,
            "unit": "iter/sec",
            "range": "stddev: 0.06307668260038295",
            "extra": "mean: 492.9934615999855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.9482330757870587,
            "unit": "iter/sec",
            "range": "stddev: 0.07008688601928074",
            "extra": "mean: 513.2856085999947 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.9203861031817773,
            "unit": "iter/sec",
            "range": "stddev: 0.06464549807266722",
            "extra": "mean: 520.7286172000295 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 23.907306155412005,
            "unit": "iter/sec",
            "range": "stddev: 0.0011422963515584938",
            "extra": "mean: 41.828217428571534 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "10176247e2bf8d495eedd209fae92751c7cfef2a",
          "message": "Update azure lite v0.3.0 (#1288)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-29T20:05:10-08:00",
          "tree_id": "fe594a63db921cb0a7b9e706173bffc6d1549bf2",
          "url": "https://github.com/tensorflow/io/commit/10176247e2bf8d495eedd209fae92751c7cfef2a"
        },
        "date": 1611979933420,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5459.574915040278,
            "unit": "iter/sec",
            "range": "stddev: 0.000010809598122440002",
            "extra": "mean: 183.16444330586174 usec\nrounds: 1464"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3897.232614795533,
            "unit": "iter/sec",
            "range": "stddev: 0.00001243165921916681",
            "extra": "mean: 256.59233072298014 usec\nrounds: 2809"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 923.2181612526241,
            "unit": "iter/sec",
            "range": "stddev: 0.000038896858588714856",
            "extra": "mean: 1.0831675999995474 msec\nrounds: 820"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 470.4984900137082,
            "unit": "iter/sec",
            "range": "stddev: 0.00006639569444475718",
            "extra": "mean: 2.1254053333324503 msec\nrounds: 483"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1535.168210987007,
            "unit": "iter/sec",
            "range": "stddev: 0.000047675454985824834",
            "extra": "mean: 651.3944158321707 usec\nrounds: 998"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 674.748631647858,
            "unit": "iter/sec",
            "range": "stddev: 0.00005314150265847535",
            "extra": "mean: 1.482033387096791 msec\nrounds: 248"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1120.3934844400026,
            "unit": "iter/sec",
            "range": "stddev: 0.00006174815183046642",
            "extra": "mean: 892.543569636896 usec\nrounds: 718"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.754436319860166,
            "unit": "iter/sec",
            "range": "stddev: 0.041947105872326695",
            "extra": "mean: 266.35156779999534 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.502451138739815,
            "unit": "iter/sec",
            "range": "stddev: 0.001195956274296324",
            "extra": "mean: 33.895488727270354 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.353852379200983,
            "unit": "iter/sec",
            "range": "stddev: 0.04811977210105411",
            "extra": "mean: 738.6329671999988 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.3370372659893281,
            "unit": "iter/sec",
            "range": "stddev: 0.03969146133195405",
            "extra": "mean: 747.9223095999942 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.311378807051754,
            "unit": "iter/sec",
            "range": "stddev: 0.04762951857022405",
            "extra": "mean: 762.5561695999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5260913946589376,
            "unit": "iter/sec",
            "range": "stddev: 0.05912902799165893",
            "extra": "mean: 1.9008104107999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4038180769157628,
            "unit": "iter/sec",
            "range": "stddev: 0.05652084170298928",
            "extra": "mean: 2.4763626423999883 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7491524423389454,
            "unit": "iter/sec",
            "range": "stddev: 0.012069516911466509",
            "extra": "mean: 1.3348418071999846 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.593797845241587,
            "unit": "iter/sec",
            "range": "stddev: 0.05211033153081381",
            "extra": "mean: 278.2571649999909 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.104072670034359,
            "unit": "iter/sec",
            "range": "stddev: 0.05771522129136032",
            "extra": "mean: 475.2687557999934 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.0161044520426064,
            "unit": "iter/sec",
            "range": "stddev: 0.06111696013600219",
            "extra": "mean: 496.0060472000123 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.9989128544640666,
            "unit": "iter/sec",
            "range": "stddev: 0.06010811651971742",
            "extra": "mean: 500.2719342000091 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 24.84351090228703,
            "unit": "iter/sec",
            "range": "stddev: 0.0017309850197645832",
            "extra": "mean: 40.251959714274626 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yong.tang.github@outlook.com",
            "name": "Yong Tang",
            "username": "yongtang"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "10176247e2bf8d495eedd209fae92751c7cfef2a",
          "message": "Update azure lite v0.3.0 (#1288)\n\nSigned-off-by: Yong Tang <yong.tang.github@outlook.com>",
          "timestamp": "2021-01-29T20:05:10-08:00",
          "tree_id": "fe594a63db921cb0a7b9e706173bffc6d1549bf2",
          "url": "https://github.com/tensorflow/io/commit/10176247e2bf8d495eedd209fae92751c7cfef2a"
        },
        "date": 1611980002158,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 5566.663504689343,
            "unit": "iter/sec",
            "range": "stddev: 0.000011747940714223063",
            "extra": "mean: 179.64082060243132 usec\nrounds: 1427"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4002.931688271047,
            "unit": "iter/sec",
            "range": "stddev: 0.000013290677497114943",
            "extra": "mean: 249.8169036783942 usec\nrounds: 2855"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 950.7929949139811,
            "unit": "iter/sec",
            "range": "stddev: 0.00004595021675924657",
            "extra": "mean: 1.0517536470601265 msec\nrounds: 833"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 497.28236052857847,
            "unit": "iter/sec",
            "range": "stddev: 0.00007221045589844877",
            "extra": "mean: 2.010929965295905 msec\nrounds: 461"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1484.8910544908927,
            "unit": "iter/sec",
            "range": "stddev: 0.00004421384273978502",
            "extra": "mean: 673.4500803783604 usec\nrounds: 1269"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 685.1507780841627,
            "unit": "iter/sec",
            "range": "stddev: 0.00005122351752031172",
            "extra": "mean: 1.4595327510190201 msec\nrounds: 245"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1141.237657568669,
            "unit": "iter/sec",
            "range": "stddev: 0.00003146281024720099",
            "extra": "mean: 876.2416779432548 usec\nrounds: 739"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 3.7177877079985744,
            "unit": "iter/sec",
            "range": "stddev: 0.037917730558577896",
            "extra": "mean: 268.9771656000062 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 29.427759086363203,
            "unit": "iter/sec",
            "range": "stddev: 0.0007504142610136071",
            "extra": "mean: 33.98152054545666 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.371616919418574,
            "unit": "iter/sec",
            "range": "stddev: 0.04540726168734373",
            "extra": "mean: 729.0665387999866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.364157260017293,
            "unit": "iter/sec",
            "range": "stddev: 0.05985544195436815",
            "extra": "mean: 733.0533137999964 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.3718663467420862,
            "unit": "iter/sec",
            "range": "stddev: 0.049122053591255474",
            "extra": "mean: 728.9339828000038 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.5195985001367753,
            "unit": "iter/sec",
            "range": "stddev: 0.06750240426534167",
            "extra": "mean: 1.9245629072000157 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4131963096788694,
            "unit": "iter/sec",
            "range": "stddev: 0.06959974509554284",
            "extra": "mean: 2.420157142200003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7711477346570779,
            "unit": "iter/sec",
            "range": "stddev: 0.007168366610577747",
            "extra": "mean: 1.2967683818000069 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.7073699355640737,
            "unit": "iter/sec",
            "range": "stddev: 0.059211772607479965",
            "extra": "mean: 269.73299599999336 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.137909764639722,
            "unit": "iter/sec",
            "range": "stddev: 0.0563541092079292",
            "extra": "mean: 467.7465889999894 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.0545863630857673,
            "unit": "iter/sec",
            "range": "stddev: 0.057048660555140185",
            "extra": "mean: 486.7159726000068 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.0148725260653535,
            "unit": "iter/sec",
            "range": "stddev: 0.057000789004091106",
            "extra": "mean: 496.3093133999905 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 25.24875879717444,
            "unit": "iter/sec",
            "range": "stddev: 0.000857608432263729",
            "extra": "mean: 39.605907285704234 msec\nrounds: 7"
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
          "id": "e986ea51d0f15c1349a666c9cbda93b8a7a4ce1b",
          "message": "Add reference to github-pages benchmarks in README (#1289)\n\n* add reference to github-pages benchmarks\r\n\r\n* minor grammar change\r\n\r\n* Update README.md\r\n\r\nCo-authored-by: Yuan Tang <terrytangyuan@gmail.com>\r\n\r\nCo-authored-by: Yuan Tang <terrytangyuan@gmail.com>",
          "timestamp": "2021-01-30T19:38:29-08:00",
          "tree_id": "58cd5ef2e9be619891237e0507ef4e28ad61228f",
          "url": "https://github.com/tensorflow/io/commit/e986ea51d0f15c1349a666c9cbda93b8a7a4ce1b"
        },
        "date": 1612064675062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6155.399816406661,
            "unit": "iter/sec",
            "range": "stddev: 0.000008030264333687106",
            "extra": "mean: 162.45898395333973 usec\nrounds: 1371"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4411.697326207488,
            "unit": "iter/sec",
            "range": "stddev: 0.000012241487855060303",
            "extra": "mean: 226.67012853750083 usec\nrounds: 2933"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1061.3394818844165,
            "unit": "iter/sec",
            "range": "stddev: 0.000009284744513801829",
            "extra": "mean: 942.2055968600097 usec\nrounds: 955"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 544.6244610282749,
            "unit": "iter/sec",
            "range": "stddev: 0.00002413789401967814",
            "extra": "mean: 1.8361275916839213 msec\nrounds: 529"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1789.3841064680955,
            "unit": "iter/sec",
            "range": "stddev: 0.000008703441818860578",
            "extra": "mean: 558.8515044843056 usec\nrounds: 1338"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 766.3164613317321,
            "unit": "iter/sec",
            "range": "stddev: 0.00001715239725027263",
            "extra": "mean: 1.304943910851353 msec\nrounds: 258"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1277.0492820646937,
            "unit": "iter/sec",
            "range": "stddev: 0.000013460724215509775",
            "extra": "mean: 783.0551365905245 usec\nrounds: 798"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.135820076547112,
            "unit": "iter/sec",
            "range": "stddev: 0.04485762323131705",
            "extra": "mean: 241.79001540001082 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 33.29719059897788,
            "unit": "iter/sec",
            "range": "stddev: 0.0007131215895210038",
            "extra": "mean: 30.032563769229732 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.446556675610164,
            "unit": "iter/sec",
            "range": "stddev: 0.05413895821191851",
            "extra": "mean: 691.2967993999928 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.4524927482716248,
            "unit": "iter/sec",
            "range": "stddev: 0.053731277771659675",
            "extra": "mean: 688.4715956000036 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.4206576869923062,
            "unit": "iter/sec",
            "range": "stddev: 0.06138429837336318",
            "extra": "mean: 703.8993342000026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.6004593566198084,
            "unit": "iter/sec",
            "range": "stddev: 0.11117106219739988",
            "extra": "mean: 1.6653916521999803 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4499959656331708,
            "unit": "iter/sec",
            "range": "stddev: 0.08571183830811152",
            "extra": "mean: 2.2222421451999934 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7856282965535926,
            "unit": "iter/sec",
            "range": "stddev: 0.005396169820790505",
            "extra": "mean: 1.2728665761999878 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.8637272068881425,
            "unit": "iter/sec",
            "range": "stddev: 0.05783812439530789",
            "extra": "mean: 258.8174439999875 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.259271602530232,
            "unit": "iter/sec",
            "range": "stddev: 0.0690284205479992",
            "extra": "mean: 442.62053260000584 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.161731751788611,
            "unit": "iter/sec",
            "range": "stddev: 0.07693098994723048",
            "extra": "mean: 462.5920858000086 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.201337681585603,
            "unit": "iter/sec",
            "range": "stddev: 0.06494058293663797",
            "extra": "mean: 454.26924200003214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 26.035625006897938,
            "unit": "iter/sec",
            "range": "stddev: 0.0036296470922999",
            "extra": "mean: 38.408910857145074 msec\nrounds: 7"
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
          "id": "e986ea51d0f15c1349a666c9cbda93b8a7a4ce1b",
          "message": "Add reference to github-pages benchmarks in README (#1289)\n\n* add reference to github-pages benchmarks\r\n\r\n* minor grammar change\r\n\r\n* Update README.md\r\n\r\nCo-authored-by: Yuan Tang <terrytangyuan@gmail.com>\r\n\r\nCo-authored-by: Yuan Tang <terrytangyuan@gmail.com>",
          "timestamp": "2021-01-30T19:38:29-08:00",
          "tree_id": "58cd5ef2e9be619891237e0507ef4e28ad61228f",
          "url": "https://github.com/tensorflow/io/commit/e986ea51d0f15c1349a666c9cbda93b8a7a4ce1b"
        },
        "date": 1612064870999,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 3765.080810939151,
            "unit": "iter/sec",
            "range": "stddev: 0.0000643676263586285",
            "extra": "mean: 265.5985489327553 usec\nrounds: 1124"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 3294.42749173218,
            "unit": "iter/sec",
            "range": "stddev: 0.000060038707631897235",
            "extra": "mean: 303.5428773313839 usec\nrounds: 2038"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 904.4149522791947,
            "unit": "iter/sec",
            "range": "stddev: 0.0002125392655769146",
            "extra": "mean: 1.105687159947902 msec\nrounds: 794"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 527.4893515227171,
            "unit": "iter/sec",
            "range": "stddev: 0.00033734763769984345",
            "extra": "mean: 1.895772866529484 msec\nrounds: 487"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1485.2139269349675,
            "unit": "iter/sec",
            "range": "stddev: 0.0001136763232400258",
            "extra": "mean: 673.3036782544166 usec\nrounds: 1467"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 495.6775067816054,
            "unit": "iter/sec",
            "range": "stddev: 0.0003950658429448316",
            "extra": "mean: 2.0174407479026444 msec\nrounds: 238"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 848.6244145341042,
            "unit": "iter/sec",
            "range": "stddev: 0.00020919079150133354",
            "extra": "mean: 1.178377598939339 msec\nrounds: 566"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 2.8466414682608154,
            "unit": "iter/sec",
            "range": "stddev: 0.047150905197581944",
            "extra": "mean: 351.2911658000121 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 25.99985091685626,
            "unit": "iter/sec",
            "range": "stddev: 0.0017528662474910832",
            "extra": "mean: 38.461758999997905 msec\nrounds: 11"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 0.8791682481650526,
            "unit": "iter/sec",
            "range": "stddev: 0.03958715086174467",
            "extra": "mean: 1.1374387122000145 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 0.8640505122215288,
            "unit": "iter/sec",
            "range": "stddev: 0.04112618231719883",
            "extra": "mean: 1.1573397455999839 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 0.8330698391747171,
            "unit": "iter/sec",
            "range": "stddev: 0.04679611992113174",
            "extra": "mean: 1.2003795515999627 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.39097463141610006,
            "unit": "iter/sec",
            "range": "stddev: 0.05594685745619882",
            "extra": "mean: 2.557710704600004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.2991131095000184,
            "unit": "iter/sec",
            "range": "stddev: 0.03796748317243593",
            "extra": "mean: 3.34321689100002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.6151520554276383,
            "unit": "iter/sec",
            "range": "stddev: 0.04633898392677014",
            "extra": "mean: 1.6256143357999917 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 2.3721686786218865,
            "unit": "iter/sec",
            "range": "stddev: 0.04313404689182345",
            "extra": "mean: 421.5551823999931 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 1.520975755348354,
            "unit": "iter/sec",
            "range": "stddev: 0.047454811880265665",
            "extra": "mean: 657.4726759999976 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 1.5324714029251347,
            "unit": "iter/sec",
            "range": "stddev: 0.05324709309135856",
            "extra": "mean: 652.5407247999738 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 1.499921891417451,
            "unit": "iter/sec",
            "range": "stddev: 0.06492318643940194",
            "extra": "mean: 666.7013834000272 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 20.900577270659326,
            "unit": "iter/sec",
            "range": "stddev: 0.0035173161145262725",
            "extra": "mean: 47.845568428572605 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "rencheng311@gmail.com",
            "name": "Cheng Ren",
            "username": "burgerkingeater"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b1b49d9bf941ba913eb3763deffa08064448f4c",
          "message": "Update _toc.yaml (#1290)",
          "timestamp": "2021-01-31T12:58:07+05:30",
          "tree_id": "38d7b7fc759a362086d512a98395f3a701368bc2",
          "url": "https://github.com/tensorflow/io/commit/4b1b49d9bf941ba913eb3763deffa08064448f4c"
        },
        "date": 1612078427189,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav]]",
            "value": 6147.735338911903,
            "unit": "iter/sec",
            "range": "stddev: 0.00000827996985843615",
            "extra": "mean: 162.6615241014899 usec\nrounds: 1307"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[wav|s24]]",
            "value": 4406.336938823977,
            "unit": "iter/sec",
            "range": "stddev: 0.000007527727488975422",
            "extra": "mean: 226.94587678691985 usec\nrounds: 2938"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[flac]]",
            "value": 1080.0734929807695,
            "unit": "iter/sec",
            "range": "stddev: 0.0000089409107987159",
            "extra": "mean: 925.8629218278619 usec\nrounds: 985"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[vorbis]]",
            "value": 573.4418513382473,
            "unit": "iter/sec",
            "range": "stddev: 0.000011632409773640668",
            "extra": "mean: 1.7438559771427382 msec\nrounds: 525"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[audio[mp3]]",
            "value": 1675.1157556714368,
            "unit": "iter/sec",
            "range": "stddev: 0.000021169203009063366",
            "extra": "mean: 596.9736697982223 usec\nrounds: 1384"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[hdf5]",
            "value": 759.7387954650064,
            "unit": "iter/sec",
            "range": "stddev: 0.000027215968196853917",
            "extra": "mean: 1.3162418530804907 msec\nrounds: 211"
          },
          {
            "name": "tests/test_io_tensor_eager.py::test_io_tensor_benchmark[arrow]",
            "value": 1262.5299680353332,
            "unit": "iter/sec",
            "range": "stddev: 0.000018137233627476336",
            "extra": "mean: 792.0604067372237 usec\nrounds: 772"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[mnist]",
            "value": 4.071679822284544,
            "unit": "iter/sec",
            "range": "stddev: 0.043942481673293815",
            "extra": "mean: 245.59887899999922 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[lmdb]",
            "value": 31.89401060903366,
            "unit": "iter/sec",
            "range": "stddev: 0.0010032967244180103",
            "extra": "mean: 31.353849230763092 msec\nrounds: 13"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav]]",
            "value": 1.4391881418569374,
            "unit": "iter/sec",
            "range": "stddev: 0.0573072611188428",
            "extra": "mean: 694.8361864000162 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[wav|s24]]",
            "value": 1.44179619126374,
            "unit": "iter/sec",
            "range": "stddev: 0.06104261272509333",
            "extra": "mean: 693.579304800005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[flac]]",
            "value": 1.417716458105676,
            "unit": "iter/sec",
            "range": "stddev: 0.06378671186508632",
            "extra": "mean: 705.3596608000021 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[vorbis]]",
            "value": 0.586961194578029,
            "unit": "iter/sec",
            "range": "stddev: 0.07079771776647156",
            "extra": "mean: 1.7036901404000104 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[audio[mp3]]",
            "value": 0.4571144495956045,
            "unit": "iter/sec",
            "range": "stddev: 0.07318328981733815",
            "extra": "mean: 2.1876359429999868 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[hdf5]",
            "value": 0.7732317120710703,
            "unit": "iter/sec",
            "range": "stddev: 0.10421930774364072",
            "extra": "mean: 1.293273393200002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy]",
            "value": 3.8540436070542117,
            "unit": "iter/sec",
            "range": "stddev: 0.06372795785979307",
            "extra": "mean: 259.4677440000055 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[structure]]",
            "value": 2.2445677440573055,
            "unit": "iter/sec",
            "range": "stddev: 0.07819814139129426",
            "extra": "mean: 445.52007959999855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/tuple]]",
            "value": 2.219543358669172,
            "unit": "iter/sec",
            "range": "stddev: 0.07653932054227132",
            "extra": "mean: 450.54312460000574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[numpy[file/dict]]",
            "value": 2.1839330584367196,
            "unit": "iter/sec",
            "range": "stddev: 0.07535848402142013",
            "extra": "mean: 457.8894926000203 msec\nrounds: 5"
          },
          {
            "name": "tests/test_io_dataset_eager.py::test_io_dataset_benchmark[sql]",
            "value": 27.06468937097817,
            "unit": "iter/sec",
            "range": "stddev: 0.001376389206512443",
            "extra": "mean: 36.9485120000052 msec\nrounds: 7"
          }
        ]
      }
    ]
  }
}