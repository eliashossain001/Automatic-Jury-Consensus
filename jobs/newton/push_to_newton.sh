#!/usr/bin/env bash
# Push the ENTIRE local corrfilter folder -> Newton (nothing excluded).
# Newton has publickey auth DISABLED, so this prompts for your Newton password.
# ControlMaster keeps one connection open so you only type it once.
set -euo pipefail

SRC="/home/elias/elias_projects/corrfilter/"
DEST="md686119@newton.ist.ucf.edu:/home/md686119/elias/final_backup/ICLR-26/corrfilter/"

CM=(-o ControlMaster=auto -o ControlPath=/tmp/ssh-newton-%r@%h:%p -o ControlPersist=30m
    -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive)

echo ">> opening master connection (enter Newton password)"
ssh "${CM[@]}" md686119@newton.ist.ucf.edu \
    "mkdir -p /home/md686119/elias/final_backup/ICLR-26/corrfilter"

# -a archive  -H hardlinks  -z compress  --partial+--append-verify = resumable
rsync -aHz --partial --append-verify --info=progress2,stats2 \
      -e "ssh ${CM[*]}" "$SRC" "$DEST"

echo ">> done"
