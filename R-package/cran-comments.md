## Test environments

* local OS X install, R 3.5
* ubuntu 14.04 (on travis-ci), R 3.5
* ubuntu 18.04 (on travis-ci), R 3.5
* win-builder (devel)

## R CMD check results

```
0 errors | 0 warnings | 0 note
```

## Comments

The examples are wrapped in `\dontrun{}` block and most of the tests are skipped via `skip_on_cran()` since they can only be run when both Python and TensorFlow are installed but this is currently not viable on CRAN test machines.
