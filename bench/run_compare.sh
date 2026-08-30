#!/usr/bin/env bash
# Baseline (main) vs optimised (perf/batch-hprod), same data, same ranks.
# Uses a git worktree so the working repo never changes branch.
set -u
REPO=/home/prawns/SOP/HadNet-HadDNN
BASE=/tmp/claude-1000/-home-prawns-SOP/5db9bb95-6436-4ed3-af01-66acd9906fe5/scratchpad/baseline
PY=$REPO/.venv/bin/python
OUT=$REPO/bench
mkdir -p "$OUT"

run () {                       # run <tree> <folder> <script> <tag>
  local tree=$1 folder=$2 script=$3 tag=$4
  echo "=== $tag : $folder/$script ===" 
  ( cd "$tree/$folder" && timeout 5400 "$PY" "$script" 2>&1 ) > "$OUT/${tag}.log" 2>&1
  echo "    exit=$? lines=$(wc -l < "$OUT/${tag}.log")"
}

echo "### IMAGES - full sweep, both trees"
run "$BASE" application-images main_image.py images_baseline
run "$REPO" application-images main_image.py images_optimised

echo "### VIDEO - reduced maxIter so AGD/IGD do not dominate"
for t in "$BASE:video_baseline" "$REPO:video_optimised"; do
  tree=${t%%:*}; tag=${t##*:}
  cp "$tree/application-videos/main_video.py" /tmp/mv_$tag.bak
  sed -i 's/maxIter = 40000/maxIter = 2000/' "$tree/application-videos/main_video.py"
  run "$tree" application-videos main_video.py "$tag"
  cp /tmp/mv_$tag.bak "$tree/application-videos/main_video.py"
done
echo "### DONE"
