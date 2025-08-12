#!/bin/bash

APP_URL="http://192.168.1.243:30915"
SCRIPT="/home/ubuntu/DeathStarBench/socialNetwork/wrk2/scripts/social-network/sa/mixed-workload.lua"
NAMESPACE="sa"
APP_LABEL="app=customer-feedback"

LOG_FILE="experiment.log"
NODE_STATS_LOG="node_stats.log"

# Print log with timestamp
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a $LOG_FILE
}

run_wrk_cycles() {
  RATE=$1
  LABEL=$2
  THREADS=$3
  CONNECTIONS=$4

  for i in {1..3}; do
    log "🚀 Starting $LABEL load - Cycle $i"
    /home/ubuntu/DeathStarBench/wrk2/wrk \
      -t$THREADS -c$CONNECTIONS -d300s -R $RATE \
      -s $SCRIPT $APP_URL > wrk_${LABEL}_cycle${i}.log
    log "🛑 Resting for 2 minutes"
    sleep 120
  done
}

redeploy_app() {
  YAML=$1
  log "📦 Applying $YAML"
  kubectl apply -f $YAML | tee -a $LOG_FILE

  log "🧹 Deleting old pod"
  kubectl delete pod -n $NAMESPACE -l $APP_LABEL | tee -a $LOG_FILE

  log "⏳ Waiting for pod to become Ready..."
  while [[ $(kubectl get pods -n $NAMESPACE -l $APP_LABEL -o jsonpath='{.items[0].status.phase}') != "Running" ]]; do
    log "⌛ Pod not ready yet..."
    sleep 5
  done
  log "✅ Pod is running!"
}

# Start background resource monitoring
log "📊 Starting node resource tracking..."
(
  while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> $NODE_STATS_LOG
    kubectl top node >> $NODE_STATS_LOG
    echo "-----------------------------" >> $NODE_STATS_LOG
    sleep 30
  done
) &
MONITOR_PID=$!

# === LOW LOAD ===
redeploy_app "deployment-low.yaml"
run_wrk_cycles 400 "low" 8 300

log "🛑 Final rest before switching to MID load"
sleep 120

# === MID LOAD ===
redeploy_app "deployment-mid.yaml"
run_wrk_cycles 850 "mid" 12 600

log "🛑 Final rest before switching to HIGH load"
sleep 120

# === HIGH LOAD ===
redeploy_app "deployment-high.yaml"
run_wrk_cycles 3000 "high" 24 1000

# Stop background monitoring
log "🛑 Stopping node resource tracking..."
kill $MONITOR_PID

log "📁 All logs collected — node_stats.log and wrk_* logs are ready."
log "🎉 Experiment complete!"
