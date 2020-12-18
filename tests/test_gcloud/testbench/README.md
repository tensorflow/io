# GCS Testbench

This is a minimal testbench for GCS. It only supports data operation and creating/listing/deleteing bucket.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Test Bench

```bash
gunicorn --bind "0.0.0.0:9099" --worker-class gevent --chdir "tests/test_gcs/testbench" testbench:application
```
