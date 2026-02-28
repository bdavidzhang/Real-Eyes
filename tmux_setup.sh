#!/bin/bash

# Configuration
SESSION_NAME="claude-dev"
CLAUDE_WINDOW="claude"
BASH_WINDOW="bash-commands"

# 1. Kill the session if it already exists (start fresh)
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# 2. Start the main session and the Claude window
# Note: --dangerously-allow-permissions allows Claude to run bash/files without asking
tmux new-session -d -s "$SESSION_NAME" -n "$CLAUDE_WINDOW"
tmux send-keys -t "$SESSION_NAME:$CLAUDE_WINDOW" "claude --dangerously-skip-permissions" Enter

# 3. Create the second window for your manual bash commands
tmux new-window -t "$SESSION_NAME" -n "$BASH_WINDOW"

# 4. Instruction for the user
echo "--------------------------------------------------------"
echo "Tmux setup complete!"
echo ""
echo "To view CLAUDE (Window 0):"
echo "  Run this in Terminal 1: tmux attach -t $SESSION_NAME"
echo ""
echo "To view BASH (Window 1) independently:"
echo "  Run this in Terminal 2: tmux new-session -t $SESSION_NAME -s bash-view"
echo "  Then press: Ctrl+b then 1"
echo "--------------------------------------------------------"
