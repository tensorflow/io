#!/usr/bin/env bash
OUTPUT=$(dmg -i pool create -s 2G TEST_POOL)
export POOL_UUID=`echo -e $OUTPUT | cut -d':' -f 3 | cut -d ' ' -f 2 | xargs`
echo "$POOL_UUID"
export POOL_LABEL='TEST_POOL'
echo "$POOL_LABEL"
OUTPUT=$(daos cont create --pool=TEST_POOL --type=POSIX TEST_CONT)
export CONT_UUID=`echo -e $OUTPUT | cut -d ' ' -f 4 | xargs`
echo "$CONT_UUID"
export CONT_LABEL='TEST_CONT'
echo "$CONT_LABEL"

