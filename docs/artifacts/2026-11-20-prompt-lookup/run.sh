#!/bin/bash
# Mirrored arms, a fresh server each, with DFly2 then without a drafter (the
# file edit only). Arms name binaries $BIN_DIR/gufo-<arm>: E33 compared base
# (830596c) and lookup (8-row verify); E34 compared l8 (= lookup) and l16.
# usage: run.sh MODEL.gguf BIN_DIR DFLY2_DIR OUT_DIR SOURCE_FILE A B
M=$1; BIN_DIR=$2; DFLY=$3; OUT=$4; SRC=$5; A=$6; B=$7
PORT=18441
for arm in ${A}1 ${B}1 ${B}2 ${A}2 n${A}1 n${B}1 n${B}2 n${A}2; do
  case $arm in
    n*) name=${arm#n}; EXTRA=""; ONLY=edit;;
    *) name=$arm; EXTRA="--dfly-model $DFLY"; ONLY="";;
  esac
  HY4_DFLY_PROFILE=1 $BIN_DIR/gufo-${name%?} serve llm --model $M -c 32768 --port $PORT $EXTRA > $OUT/server-$arm.log 2>&1 &
  spid=$!
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; do
    kill -0 $spid 2>/dev/null || { echo "server $arm died"; exit 1; }; sleep 2
  done
  echo "== $arm"
  python3 "$(dirname "$0")/arm.py" $PORT $OUT/$arm.json $SRC $ONLY
  kill $spid; wait $spid 2>/dev/null
done
