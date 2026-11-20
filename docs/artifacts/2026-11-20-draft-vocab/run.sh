#!/bin/bash
# Mirrored arms, a fresh server each, with DFly2. The sweep ran one build
# with a temporary switch for the head prefix (full, 32768, 65536 rows); the
# retained build fixes 65536. BIN is that sweep build.
# usage: run.sh MODEL.gguf BIN DFLY2_DIR OUT_DIR
M=$1; BIN=$2; DFLY=$3; OUT=$4
PORT=18443
for arm in full1 v32k1 v64k1 v64k2 v32k2 full2; do
  case ${arm%?} in full) V="";; v32k) V=32768;; v64k) V=65536;; esac
  HY4_DFLY_VOCAB=$V $BIN serve llm --model $M -c 32768 --port $PORT --dfly-model $DFLY > $OUT/server-$arm.log 2>&1 &
  spid=$!
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; do
    kill -0 $spid 2>/dev/null || { echo "server $arm died"; exit 1; }; sleep 2
  done
  python3 "$(dirname "$0")/arm.py" $PORT $OUT/$arm.json
  kill $spid; wait $spid 2>/dev/null
done
