#!/bin/bash
# Mirrored arms, a fresh server each: base (e6bb904), tail (E31) and new (E32,
# stochastic proposals); binaries $BIN_DIR/gufo-{base,tail,new}.
# usage: run.sh MODEL.gguf BIN_DIR DFLY2_DIR OUT_DIR LONG_TEXT
M=$1; BIN_DIR=$2; DFLY=$3; OUT=$4
DOC=$5  # any long text; the runs used this repository's docs/PLAN.md
PORT=18437
for arm in base1 tail1 new1 new2 tail2 base2; do
  G=$BIN_DIR/gufo-${arm%?}
  $G serve llm --model $M -c 32768 --port $PORT --dfly-model $DFLY > $OUT/server-$arm.log 2>&1 &
  spid=$!
  until curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; do
    kill -0 $spid 2>/dev/null || { echo "server $arm died"; tail -5 $OUT/server-$arm.log; exit 1; }; sleep 2
  done
  echo "== $arm"
  python3 "$(dirname "$0")/arm.py" $PORT $OUT/$arm.json $DOC
  kill $spid; wait $spid 2>/dev/null
done
