window.BENCHMARK_DATA = {
  "lastUpdate": 1611082055470,
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
      }
    ]
  }
}