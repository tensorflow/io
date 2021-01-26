window.BENCHMARK_DATA = {
  "lastUpdate": 1611625565581,
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
      }
    ]
  }
}