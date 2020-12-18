set -e
set -o pipefail

if [ "$#" -eq 1 ]; then
  container=$1
  docker pull python:3.8
  docker run -d --rm --net=host --name=$container -v $PWD:/v -w /v python:3.8 bash -x -c 'python3 -m pip install -r /v/tests/test_gcloud/testbench/requirements.txt && gunicorn --bind "0.0.0.0:9099" --worker-class gevent --chdir "/v/tests/test_gcloud/testbench" testbench:application'
  echo wait 30 secs until gcs emulator is up and running
  sleep 30
  exit 0
fi

export PATH=$(python3 -m site --user-base)/bin:$PATH

python3 -m pip install -r tests/test_gcloud/testbench/requirements.txt
echo starting gcs-testbench
gunicorn --bind "0.0.0.0:9099" \
    --worker-class gevent \
    --chdir "tests/test_gcloud/testbench" \
    testbench:application &
sleep 30 # Wait for storage emulator to start
echo gcs-testbench started successfully
