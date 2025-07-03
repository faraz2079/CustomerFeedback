#!/bin/bash

APP_URL="http://192.168.1.243:30915"
SCRIPT="/home/ubuntu/DeathStarBench/socialNetwork/wrk2/scripts/social-network/sa/mixed-workload.lua"
NAMESPACE="sa"
APP_LABEL="app=customer-feedback"
LOG_FILE="test_switch.log"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a $LOG_FILE
}

run_wrk() {
  RATE=$1
  LABEL=$2
  DURATION=$3
  log "▶️ Running $LABEL workload for ${DURATION}s at rate $RATE"
  /home/ubuntu/DeathStarBench/wrk2/wrk -t4 -c100 -d${DURATION}s -R $RATE -s $SCRIPT $APP_URL > wrk_${LABEL}_test.log
  log "✅ Finished $LABEL workload"
}

redeploy_app() {
  YAML=$1
  log "📦 Applying $YAML"
  kubectl apply -f $YAML | tee -a $LOG_FILE

  log "🧹 Deleting current pod"
  kubectl delete pod -n $NAMESPACE -l $APP_LABEL | tee -a $LOG_FILE

  log "⏳ Waiting for new pod to be Ready..."
  while [[ $(kubectl get pods -n $NAMESPACE -l $APP_LABEL -o jsonpath='{.items[0].status.phase}') != "Running" ]]; do
    log "⌛ Pod not ready yet..."
    sleep 3
  done
  log "✅ New pod is running"
}

log "🔽 TEST STARTED: Deploy LOW and run workload"
redeploy_app "deployment-low.yaml"
run_wrk 300 "low" 30

log "🛑 Short rest for 30 seconds"
sleep 30

log "🔼 Switching to MEDIUM deployment"
redeploy_app "deployment-mid.yaml"
run_wrk 700 "mid" 30

log "✅ TEST COMPLETE"
