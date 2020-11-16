set -e
set -o pipefail

if [ "$#" -eq 1 ]; then
  container=$1
  docker pull python:3.8
  docker run -d --rm --net=host --name=$container -v $PWD:/v -w /v python:3.8 bash -x -c 'python3 -m pip install gcloud-storage-emulator==0.3.0 && gcloud-storage-emulator start --port=9099'
  echo wait 30 secs until gcs emulator is up and running
  sleep 30
  exit 0
fi

export PATH=$(python3 -m site --user-base)/bin:$PATH

python3 -m pip install gcloud-storage-emulator==0.3.0

gcloud-storage-emulator start --port=9099 &

sleep 30 # Wait for storage emulator to start
echo gcs emulator started successfully
