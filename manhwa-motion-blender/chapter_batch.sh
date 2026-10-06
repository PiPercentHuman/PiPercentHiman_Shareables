#!/bin/bash
# Every panel of a comic through the multiplane rig: depth -> layers (LaMa fill) -> Blender camera move.
# The move is chosen by the panel's shape, not by eye: wide (w/h > 1.8) pans, tall drifts;
# MOVE[...] overrides (a figure near the top edge crops on a push -> "rise").
# Resumable: panels that already have a clip are skipped. Set the four paths below.
set -u
PANELS=./panels                  # your flat panels, *.png
WORK=./work                      # depth maps and layers
OUT=./clips                      # one 4 s mp4 per panel
BLENDER=blender                  # path to blender(.exe) - the portable zip works
PY=python
declare -A MOVE=( )              # e.g. MOVE=( [34_c19_lirazel]=rise [36_c21_eyes]=push )

mkdir -p $WORK/depth $WORK/layers $OUT
$PY depth_probe.py $PANELS $WORK/depth
T0=$(date +%s)
for f in $PANELS/*.png; do
  n=$(basename $f .png)
  [ -f $OUT/$n.mp4 ] && continue
  read W H < <($PY -c "from PIL import Image;print(*Image.open('$f').size)")
  move=${MOVE[$n]:-$( awk -v w=$W -v h=$H 'BEGIN{print (w/h>1.8)?"pan":"drift"}' )}
  $PY prep_layers.py $f $WORK/depth/${n}_depth.png $WORK/layers/$n --fill lama || { echo "!! layers $n"; continue; }
  $BLENDER -b --factory-startup --python multiplane.py -- $WORK/layers/$n $OUT/$n.mp4 --move $move --seconds 4 --fps 24 --scale 1.0 \
    2>&1 | grep -E "^MULTIPLANE|Error|Traceback"
  echo "== $n ($W x $H, $move)"
done
echo "done in $(( $(date +%s) - T0 )) s"
