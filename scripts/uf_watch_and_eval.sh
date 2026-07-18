#!/bin/bash
# Detached watcher: wait until the UF judge run is complete (all 10 judges ×
# 2000 votes = 20000), then run the H2b evaluator automatically. Robust to a
# judge process dying early (it will run the eval once no runner remains).
cd /home/elias/elias_projects/corrfilter || exit 1
LOG=outputs/synthetic_poisoned_ultrafeedback/uf_watcher.log
echo "watcher started $(date)" >> "$LOG"
while true; do
  total=$(python -c "
from pathlib import Path; import pandas as pd
print(sum(len(pd.read_parquet(f)) for f in Path('outputs/synthetic_poisoned_ultrafeedback/judge_votes').glob('*.parquet')))
" 2>/dev/null)
  alive=$(pgrep -f "scripts/13_run_uf_judges" | wc -l)
  echo "$(date) total=${total:-NA}/20000 runners_alive=${alive}" >> "$LOG"
  [ "${total:-0}" -ge 20000 ] && { echo "COMPLETE (full coverage)" >> "$LOG"; break; }
  [ "${alive:-0}" -eq 0 ] && { echo "runners exited (total=${total:-0})" >> "$LOG"; break; }
  sleep 120
done
echo "running evaluator $(date)" >> "$LOG"
python scripts/14_eval_poisoned_uf.py --project-root . >> "$LOG" 2>&1
echo "EVAL DONE $(date)" >> "$LOG"
