#!/usr/bin/env bash
set -euo pipefail

# -------- Config (override via env or flags) --------
NAMESPACE="${NAMESPACE:-sa}"
SERVICE_NAME="${SERVICE_NAME:-customer-feedback-service}"
APP_LABEL="${APP_LABEL:-app=customer-feedback}"

# Paths relative to this script by default
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT:-$HERE/mixed-workload.lua}"
WRK="${WRK:-$HERE/../wrk2/wrk}"

DURATION="${DURATION:-300s}"
CYCLES="${CYCLES:-3}"
SLEEP_BETWEEN_CYCLES="${SLEEP_BETWEEN_CYCLES:-120}"

LOG_FILE="${LOG_FILE:-experiment.log}"
NODE_STATS_LOG="${NODE_STATS_LOG:-node_stats.log}"

# -------- Helpers --------
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG_FILE"; }
require(){ command -v "$1" >/dev/null 2>&1 || { echo "Missing '$1' in PATH"; exit 1; }; }
kill_if_set(){ [[ -n "${1:-}" ]] && kill -9 "$1" 2>/dev/null || true; }

# Discover APP_URL:
# - If Service is NodePort, use <node-internal-ip>:<nodePort>
# - If ClusterIP, start port-forward to 8000 locally
discover_app_url() {
  local svcjson type nodePort nodeIP portNumber
  svcjson="$(kubectl -n "$NAMESPACE" get svc "$SERVICE_NAME" -o json)"
  type="$(jq -r '.spec.type' <<<"$svcjson")"

  if [[ "$type" == "NodePort" ]]; then
    nodePort="$(jq -r '
      .spec.ports
      | ( map(select(.name=="http")) + map(select(.port==80)) + . )[0].nodePort // .[0].nodePort
    ' <<<"$svcjson")"

    if [[ -n "${NODE_IP_OVERRIDE:-}" ]]; then
      nodeIP="$NODE_IP_OVERRIDE"
    else
      nodeIP="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
    fi

    APP_URL="http://$nodeIP:$nodePort"
    PF_PID=""
    log "🔎 Discovered NodePort service → APP_URL=$APP_URL"
  else
    portNumber="$(jq -r '.spec.ports[0].port' <<<"$svcjson")"
    log "🔎 ClusterIP service on port $portNumber → starting port-forward to 8000"
    kubectl -n "$NAMESPACE" port-forward "svc/$SERVICE_NAME" 8000:"$portNumber" >/dev/null 2>&1 &
    PF_PID=$!
    APP_URL="http://127.0.0.1:8000"
    sleep 2
  fi
}

# Improved: truly backgrounded, failure-proof monitor
start_node_monitor() {
  log "📊 Starting node resource tracking (every 30s)…"
  nohup bash -c '
    while true; do
      {
        echo "===== $(date "+%F %T") ====="
        kubectl top node || echo "kubectl top node failed (metrics-server missing?)"
        echo
      } >> "'"$NODE_STATS_LOG"'"
      sleep 30
    done
  ' >/dev/null 2>&1 &
  echo $!
}

redeploy_app() {
  local yaml="$1"
  log "📦 Applying $yaml"
  kubectl apply -f "$yaml" | tee -a "$LOG_FILE"

  log "🧹 Deleting old pod(s) with label $APP_LABEL (if any)"
  kubectl -n "$NAMESPACE" delete pod -l "$APP_LABEL" --ignore-not-found | tee -a "$LOG_FILE"

  log "⏳ Waiting for pod to be Ready…"
  kubectl -n "$NAMESPACE" wait --for=condition=ready pod -l "$APP_LABEL" --timeout=5m
  NEW_POD="$(kubectl -n "$NAMESPACE" get pod -l "$APP_LABEL" -o jsonpath='{.items[0].metadata.name}')"
  log "✅ Pod Ready: $NEW_POD"
}

run_wrk_cycles() {
  local rate="$1" label="$2" threads="$3" conns="$4"
  for i in $(seq 1 "$CYCLES"); do
    local out="wrk_${label}_cycle${i}_$(date +%Y%m%d_%H%M%S).log"
    log "🚀 $label load - Cycle $i  (rate=$rate, threads=$threads, conns=$conns, dur=$DURATION)"
    "$WRK" -D exp -L \
      -t"$threads" -c"$conns" -d"$DURATION" -R "$rate" \
      -s "$SCRIPT" "$APP_URL" | tee "$out"
    log "🛑 Resting for ${SLEEP_BETWEEN_CYCLES}s"
    sleep "$SLEEP_BETWEEN_CYCLES"
  done
}

# -------- Main --------
require kubectl
require jq
[[ -x "$WRK" ]] || { echo "WRK binary not found/executable at $WRK"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Lua script not found at $SCRIPT"; exit 1; }

log "=== CustomerFeedback experiment (3-cycle) ==="
log "Namespace=$NAMESPACE Service=$SERVICE_NAME Label=$APP_LABEL"
log "WRK=$WRK  SCRIPT=$SCRIPT  DURATION=$DURATION  CYCLES=$CYCLES"

discover_app_url
log "APP_URL=$APP_URL"

MONITOR_PID="$(start_node_monitor)"
trap 'log "🛑 Stopping monitors"; kill_if_set "$MONITOR_PID"; kill_if_set "${PF_PID:-}";' EXIT

# LOW
redeploy_app "$HERE/deployment-low.yaml"
run_wrk_cycles 400 "low" 8 300
log "🛑 Rest 120s before MID"; sleep 120

# MID
redeploy_app "$HERE/deployment-mid.yaml"
run_wrk_cycles 850 "mid" 12 600
log "🛑 Rest 120s before HIGH"; sleep 120

# HIGH
redeploy_app "$HERE/deployment-high.yaml"
run_wrk_cycles 3000 "high" 24 1000

log "📁 Done. Logs: $LOG_FILE, $NODE_STATS_LOG, wrk_*.log"
