set -e
set -o pipefail


npm install azurite@3.12.0
echo starting azurite-blob
$(npm bin)/azurite-blob --loose &
sleep 10 # Wait for storage emulator to start
echo azurite-blob started successfully

path_to_az=$(which az)
if [ -x "$path_to_az" ] ; then
    echo "az is already installed: $path_to_az"
else
    npm install -g azure-cli  
fi
