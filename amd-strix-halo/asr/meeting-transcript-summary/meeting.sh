#!/usr/bin/env bash
# meeting.sh — capture both ends of a call and transcribe them with gufo.
#
#   ./meeting.sh devices              list sinks, sources and playback streams
#   ./meeting.sh                      transcribe mic + loudspeaker monitor
#   ./meeting.sh --mic N --peer N     explicit PipeWire node/source names
#   ./meeting.sh --isolate            route the call into its own sink first
#   ./meeting.sh --dry-run            segment the audio, don't call the model
#   ./meeting.sh --lang it            force a language for both tracks
#
# Needs: pipewire-pulse (pactl/parec) or ffmpeg built with libpulse, python3 +
# numpy + websockets, and `gufo serve asr` running.  Writes <stem>.md,
# <stem>.jsonl, <stem>.events.jsonl and, unless --no-record, the two wav tracks.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIC="" PEER="" OUT=""
URL="ws://127.0.0.1:8081/v1/realtime?intent=transcription"
HEALTH=""            # derived from --url unless given
MODEL="qwen3-asr-1.7b" LANG_CODE="" RECORD=1 ISOLATE=0 RATE=24000
DRY=() MODE=run

die() { printf 'meeting.sh: %b\n' "$*" >&2; exit 1; }
usage() { sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --mic)    MIC=$2; shift 2;;
    --peer)   PEER=$2; shift 2;;
    --out)    OUT=$2; shift 2;;
    --url)    URL=$2; shift 2;;
    --model)  MODEL=$2; shift 2;;
    --lang)   LANG_CODE=$2; shift 2;;
    --no-record) RECORD=0; shift;;
    --no-check) HEALTH="none"; shift;;
    --dry-run)  DRY=(--dry-run); shift;;
    --isolate)  ISOLATE=1; shift;;
    -h|--help)  usage;;
    devices|ls) MODE=devices; shift;;
    *) die "unknown argument: $1 (see --help)";;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
have pactl || die "pactl not found: install pipewire-pulse (or pulseaudio-utils)"

default_sink()   { pactl get-default-sink 2>/dev/null || true; }
default_source() { pactl get-default-source 2>/dev/null || true; }

list_devices() {
  echo "== sinks (what the laptop plays; each has a <sink>.monitor source)"
  pactl list short sinks
  echo "\n== sources (microphones and monitors)"
  pactl list short sources
  echo "\n== playback streams (stream-id  client  sink)"
  pactl list short sink-inputs 2>/dev/null || echo "(none)"
  echo "\nfar end of the default output: $(default_sink).monitor"
}

# Give the call a private sink so music and notifications stay out of the
# transcript, and so we capture exactly what the meeting app plays.
isolate() {
  local sink=asr_call master; master=$(default_sink)
  [[ -n $master ]] || die "no default sink"
  if ! pactl list short sinks | awk '{print $2}' | grep -qx "$sink"; then
    # A null sink has no hardware: loop its monitor to the real output so the
    # call is still audible.
    pactl load-module module-null-sink "sink_name=$sink" \
      sink_properties=device.description=ASR_Call >/dev/null
    pactl load-module module-loopback "source=$sink.monitor" "sink=$master" \
      latency_msec=20 >/dev/null
    echo "created $sink, looped back to $master"
  fi
  local ids; ids=$(pactl list short sink-inputs | awk '{print $1}')
  if [[ -z $ids ]]; then
    echo "no playback stream running yet — start the call, then:"
    echo "  for i in \$(pactl list short sink-inputs | cut -f1); do"
    echo "      pactl move-sink-input \$i $sink; done"
  else
    for i in $ids; do pactl move-sink-input "$i" "$sink" 2>/dev/null || true; done
    echo "moved stream(s) $(tr '\n' ' ' <<<"$ids")-> $sink"
  fi
  PEER="$sink.monitor"
}

[[ $MODE == devices ]] && { list_devices; exit 0; }
[[ $ISOLATE == 1 ]] && isolate

[[ -n $MIC ]] || MIC=$(default_source)
[[ -n $MIC ]] || die "no default source; pass --mic <source>"
if [[ -z $PEER ]]; then
  s=$(default_sink); [[ -n $s ]] || die "no default sink; pass --peer <sink>"
  PEER=$s
fi
# A sink name becomes the source to record from by appending .monitor.
if ! pactl list short sources | awk '{print $2}' | grep -qx "$PEER"; then
  case $PEER in *.monitor) ;; *) PEER="$PEER.monitor";; esac
  pactl list short sources | awk '{print $2}' | grep -qx "$PEER" ||
    die "$PEER is neither a source nor a sink I can monitor. Sources:\n  $(pactl list short sources | awk '{print $2}' | paste -sd' ')"
fi

# Pick the capture tool before anything else: without one there is nothing to
# record, and that is the more useful error than "server not running".
if have parec; then CAPTURER=parec
elif have ffmpeg && ffmpeg -hide_banner -devices 2>/dev/null | grep -q ' pulse'; then
  CAPTURER=ffmpeg
elif have pw-record; then CAPTURER=pw-record
else die "need parec (pulseaudio-utils), ffmpeg with libpulse, or pw-record"; fi

# Check the server is up, at the same host:port the WebSocket will use.
if [[ -z $HEALTH ]]; then
  hp=$(sed -n 's#^wss\?://\([^/]*\).*#\1#p' <<<"$URL")
  [[ -n $hp ]] && HEALTH="http://$hp/ready"
fi
if [[ ${#DRY[@]} == 0 && -n $HEALTH && $HEALTH != none ]] && have curl; then
  curl -fsS -m 3 "$HEALTH" >/dev/null 2>&1 ||
    die "no ASR server at $HEALTH — start it with:\n  gufo serve asr --model models/Qwen3-ASR-1.7B --port 8081"
fi

runtime=${XDG_RUNTIME_DIR:-/tmp/run-$(id -u)}
# A private directory per run, so two meetings (or a leftover from a crashed
# one) never share FIFOs, and cleanup cannot remove someone else's pipes.
dir=$(mktemp -d "$runtime/asr.XXXXXX") || die "cannot make a work dir in $runtime"
mic_fifo=$dir/mic.raw peer_fifo=$dir/peer.raw
mkfifo "$mic_fifo" "$peer_fifo" || die "cannot create FIFOs in $dir"

# Each writes mono s16le at $RATE to stdout. PipeWire resamples for us, so the
# script never has to.
capture() { # capture <source-name> <fifo>
  case $CAPTURER in
    parec)
      parec --raw --format=s16le --rate=$RATE --channels=1 --channel-map=mono \
            --device="$1" --latency-msec=100 >"$2" ;;
    ffmpeg)
      ffmpeg -hide_banner -loglevel error -f pulse -i "$1" \
             -ac 1 -ar $RATE -f s16le - >"$2" ;;
    pw-record)
      # -a keeps the bytes headerless; needs pipewire >= 1.2, and --target only
      # takes a capture *node*, so for the far end pass --peer <source>, e.g. a
      # virtual source made with pw-loopback.  parec/ffmpeg are better here.
      pw-record --rate=$RATE --channels=1 --channel-map=mono --format=s16 -a \
                --target="$1" --latency=100ms - >"$2" ;;
  esac
}

printf 'capturer %s\n  MIC  <= %s\n  PEER <= %s\n' "$CAPTURER" "$MIC" "$PEER"
capture "$PEER" "$peer_fifo" & peer_pid=$!
capture "$MIC"  "$mic_fifo"  & mic_pid=$!

cleanup() {
  kill $mic_pid $peer_pid 2>/dev/null || true
  wait $mic_pid $peer_pid 2>/dev/null || true
  rm -f "$mic_fifo" "$peer_fifo" "$dir"/client.* 2>/dev/null || true
  rmdir "$dir" 2>/dev/null || true
}
trap cleanup EXIT

args=(--source "$mic_fifo" --source "$peer_fifo" --layout MIC,PEER
      --sample-width 2 --rate $RATE --url "$URL" --model "$MODEL"
      --calibrate "${CALIBRATE:-2}")
[[ -n $OUT ]] && args+=(--out "$OUT")
[[ -n $LANG_CODE ]] && args+=(--language "$LANG_CODE")
[[ $RECORD == 1 ]] && args+=(--record "${OUT:-meeting}-$(date +%Y%m%d-%H%M%S)")
[[ ${#DRY[@]} -gt 0 ]] && args+=("${DRY[@]}")

# The client runs inside a subshell that records its real exit status, and a
# signal to *this* script is forwarded to it: Ctrl-C in a terminal reaches both
# processes anyway, but `kill`/systemd-stop signalled only the wrapper would
# otherwise orphan a client still holding the meeting open.  bash's own `wait`
# reports the trap's signal number rather than the child's status, which is why
# the status travels through a file.
pid_file=$dir/client.pid rc_file=$dir/client.rc
(
  trap '' INT TERM          # this subshell's only job is to record the status
  python3 "$here/asr_meeting.py" "${args[@]}" &
  echo "$!" > "$pid_file"
  wait
  printf '%s' "$?" > "$rc_file"
) &
runner=$!

forward() {
  local pid
  pid=$(cat "$pid_file" 2>/dev/null) || return 0
  [[ -n $pid ]] && kill -"$1" "$pid" 2>/dev/null || true
}
trap 'forward INT'  INT
trap 'forward TERM' TERM

while [[ ! -f $rc_file ]] && kill -0 "$runner" 2>/dev/null; do sleep 0.1; done
for _ in 1 2 3 4 5; do [[ -f $rc_file ]] && break; sleep 0.2; done
wait "$runner" 2>/dev/null; wrc=$?
rc=$(cat "$rc_file" 2>/dev/null || true)
[[ -z $rc ]] && rc=$wrc
exit "$rc"
