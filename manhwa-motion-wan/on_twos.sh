#!/bin/bash
# Does holding drawings make generated motion read as anime?
# TV anime holds each drawing for several frames: 12 drawings a second ("on twos") or
# 8 ("on threes", Mushi Production 1963). A video model draws a new frame every 1/16 s.
# Same clip, three timings, all delivered at 24 fps so only the number of distinct
# drawings changes:
#   ones   - every generated frame, interpolated to 24 fps by frame duplication only
#   twos   - 12 distinct drawings a second, each held 2 frames
#   threes -  8 distinct drawings a second, each held 3 frames
# usage: on_twos.sh <clip.mp4> <outdir>
set -eu
IN=$1; OUT=$2; mkdir -p "$OUT"; b=$(basename "$IN" .mp4)
ffmpeg -v error -y -i "$IN" -vf "fps=24" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUT/${b}_ones.mp4"
ffmpeg -v error -y -i "$IN" -vf "fps=12,fps=24" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUT/${b}_twos.mp4"
ffmpeg -v error -y -i "$IN" -vf "fps=8,fps=24" -c:v libx264 -crf 16 -pix_fmt yuv420p "$OUT/${b}_threes.mp4"
# side by side, labelled, for looking at (not a deliverable)
ffmpeg -v error -y -i "$OUT/${b}_ones.mp4" -i "$OUT/${b}_twos.mp4" -i "$OUT/${b}_threes.mp4" -filter_complex \
 "[0]drawtext=text='16 drawings/s':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[a];\
  [1]drawtext=text='12 (on twos)':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[b];\
  [2]drawtext=text='8 (on threes)':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[c];\
  [a][b][c]hstack=inputs=3" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/${b}_compare.mp4"
for t in ones twos threes; do
  n=$(ffmpeg -v error -i "$OUT/${b}_$t.mp4" -vf "mpdecimate" -f null - -loglevel info 2>&1 | grep -c "drop" || true)
  echo "$t: $(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/${b}_$t.mp4") frames"
done
