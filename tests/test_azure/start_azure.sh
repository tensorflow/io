set -e
set -o pipefail

nvm use 8
npm install -global azurite
echo starting azurite-blob
azurite-blob &
echo azurite-blob started successfully
