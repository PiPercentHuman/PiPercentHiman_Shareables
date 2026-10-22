#!/bin/bash
# Play a clip backwards, so it builds up INTO the panel as drawn instead of relaxing away from it.
# Why: in 3 of 3 of our shots that started on a peak expression (a gasp, a scream, a roar), Wan
# calmed the face within ~2.5-4 s. Reversed, the last frame is the panel (mean difference 7.6/255
# on the dragon). Cost: everything else runs backwards too - birds, sparks, falling leaves.
# usage: reverse.sh <clip.mp4> <out.mp4>
set -eu
ffmpeg -v error -y -i "$1" -vf reverse -c:v libx264 -crf 16 -pix_fmt yuv420p -an "$2"
