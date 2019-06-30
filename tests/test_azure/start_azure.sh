set -e
set -o pipefail

npm install -g azurite@2.7.0
echo starting azurite-blob
azurite-blob &
sleep 10 # Wait for storage emulator to start
echo azurite-blob started successfully
