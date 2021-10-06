#!/usr/bin/env bash
OUTPUT=$(dmg -i pool create -s 2G)
echo "$OUTPUT"
export POOL_ID=`echo -e $OUTPUT | cut -d':' -f 3 | cut -d ' ' -f 2 | xargs`
echo "$POOL_ID"
OUTPUT=$(daos cont create --pool=$POOL_ID --type=POSIX)
export CONT_ID=`echo -e $OUTPUT | cut -d ' ' -f 4 | xargs`
echo "$CONT_ID"