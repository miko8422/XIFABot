#!/bin/sh
# Regenerate synthetic benchmark utterances (macOS `say`) into runs/asr_fixtures, 16 kHz mono.
set -e
cd "$(dirname "$0")/../runs" && mkdir -p asr_fixtures && cd asr_fixtures
say -v Samantha -o en1.aiff "Did my background task finish?"
say -v Samantha -o en2.aiff "What should I eat for dinner tonight?"
say -v Kyoko -o ja1.aiff "今日はちょっと疲れたよ。"
say -v Kyoko -o ja2.aiff "明日の天気ってどうかな？"
say -v Samantha -o mix1.aiff "Can you check if the Steam download finished, the one called Kissa Stella?"
for f in *.aiff; do afconvert -f WAVE -d LEI16@16000 -c 1 "$f" "${f%.aiff}.wav" && rm "$f"; done
cp ../../voices/natsume/clips/NA_sys_05.wav ja_natsume05.wav 2>/dev/null || true
