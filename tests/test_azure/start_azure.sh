set -e
set -o pipefail

npm install -global azurite
echo starting azurite-blob
azurite-blob &
echo azurite-blob started successfully
