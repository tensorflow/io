#!/usr/bin/env bash
dmg -i pool create -s 2G TEST_POOL
daos cont create --pool=TEST_POOL --type=POSIX TEST_CONT