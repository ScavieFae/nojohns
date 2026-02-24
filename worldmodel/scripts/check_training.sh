#!/bin/bash
# Quick check on training status across both machines.
# Usage: bash worldmodel/scripts/check_training.sh

echo "=== LOCAL (Scav) ==="
PID=$(pgrep -f "worldmodel.scripts.train" 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Training running: PID $PID"
    # Memory usage
    ps -o rss= -p $PID 2>/dev/null | awk '{printf "  RAM: %.1f GB\n", $1/1024/1024}'
    # Latest epoch from wandb output log
    LOGFILE=$(ls -t ~/claude-projects/nojohns/wandb/*/files/output.log 2>/dev/null | head -1)
    if [ -n "$LOGFILE" ]; then
        echo "  Latest:"
        grep "Epoch" "$LOGFILE" | tail -2 | sed 's/^/  /'
    fi
    # Check for NaN or errors
    if [ -n "$LOGFILE" ]; then
        NANS=$(grep -ci "nan" "$LOGFILE" 2>/dev/null)
        ERRORS=$(grep -ci "error\|exception\|traceback" "$LOGFILE" 2>/dev/null)
        [ "$NANS" -gt 0 ] && echo "  WARNING: $NANS lines with NaN!"
        [ "$ERRORS" -gt 0 ] && echo "  WARNING: $ERRORS error lines!"
    fi
else
    echo "No training running"
    # Check if it finished
    MANIFEST=$(ls -t ~/claude-projects/nojohns-training/checkpoints/*/manifest.json 2>/dev/null | head -1)
    [ -n "$MANIFEST" ] && echo "  Latest manifest: $MANIFEST"
fi

echo ""
echo "=== SCAVIEFAE ==="
ssh queenmab@100.93.8.111 "
PID=\$(pgrep -f 'worldmodel.scripts.train' 2>/dev/null)
if [ -n \"\$PID\" ]; then
    echo 'Training running: PID '\$PID
    ps -o rss= -p \$PID 2>/dev/null | awk '{printf \"  RAM: %.1f GB\n\", \$1/1024/1024}'
    LOGFILE=\$(ls -t ~/claude-projects/nojohns-training/train.log 2>/dev/null | head -1)
    if [ -n \"\$LOGFILE\" ]; then
        echo '  Latest:'
        grep 'Epoch' \"\$LOGFILE\" | tail -2 | sed 's/^/  /'
    fi
else
    echo 'No training running'
    MANIFEST=\$(ls -t ~/claude-projects/nojohns-training/checkpoints/*/manifest.json 2>/dev/null | head -1)
    [ -n \"\$MANIFEST\" ] && echo \"  Latest manifest: \$MANIFEST\"
fi
" 2>/dev/null || echo "Can't reach ScavieFae"
