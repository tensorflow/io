set -e
set -o pipefail

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing ffmpeg"
    brew install ffmpeg
fi
