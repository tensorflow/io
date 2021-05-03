set -e
set -o pipefail


npm install azurite@3.12.0
echo starting azurite-blob
$(npm bin)/azurite-blob &
sleep 10 # Wait for storage emulator to start
echo azurite-blob started successfully
