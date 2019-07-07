set -e
set -o pipefail


if [[ "$(uname)" == "Darwin" ]]; then
  npm install -g azurite@2.7.0
  echo starting azurite-blob
  azurite-blob &
else
  docker run -t -d --rm --net=host node bash -c 'npm install -g azurite@2.7.0 && azurite-blob'
fi
sleep 10 # Wait for storage emulator to start
echo azurite-blob started successfully
