set -e
set -o pipefail

export PATH=$(python3 -m site --user-base)/bin:$PATH

python3 -m pip install gcloud-storage-emulator==0.3.0

gcloud-storage-emulator start --port=9090 &

sleep 30 # Wait for storage emulator to start
echo gcs emulator started successfully
