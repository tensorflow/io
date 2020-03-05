set -e
set -o pipefail


if [[ "$(uname)" == "Linux" ]]; then
  docker run -t -d --rm --net=host node bash -c 'npm install -g azurite@2.7.0 && azurite-blob'
else
  npm install -g azurite@2.7.0
  echo starting azurite-blob
  azurite-blob &
fi
sleep 10 # Wait for storage emulator to start
echo azurite-blob started successfully
