export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export LVSM_PORT=${LVSM_PORT:-12200}

session_name="lvsm_server"

# Create a detached tmux session
tmux new-session -d -s ${session_name}

tmux send-keys -t ${session_name} "${VLFM_PYTHON} LVSM_server.py --port ${LVSM_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"