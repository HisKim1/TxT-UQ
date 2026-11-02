#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "--execute" ]; then
  printf '%s\n' \
    "Refusing to start the compute pipeline without an explicit --execute flag." \
    "This script trains models and generates multi-terabyte artifacts." \
    "Usage: scripts/run_txt_uq.sh --execute"
  exit 2
fi

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
PATH_VALUES=$(uv run python -c 'from config import DATA_ROOT, CACHE_ROOT, LOGS_DIR; print("\n".join([DATA_ROOT, CACHE_ROOT, LOGS_DIR]))')
mapfile -t CONFIG_PATHS <<< "$PATH_VALUES"
export TXT_DATA_ROOT="${CONFIG_PATHS[0]}"
export TXT_CACHE_ROOT="${CONFIG_PATHS[1]}"
readonly LOGS_DIR="${CONFIG_PATHS[2]}"

require_recipe_value() {
  local name="$1" actual="$2" expected="$3"
  if [ "$actual" != "$expected" ]; then
    printf 'TxT-UQ recipe mismatch: %s=%q, expected %q\n' \
      "$name" "$actual" "$expected" >&2
    exit 2
  fi
}

require_recipe_value VERSION "${VERSION:-txt}" txt
require_recipe_value RESID_TAG "${RESID_TAG:-txt_uq}" txt_uq
require_recipe_value ARCH "${ARCH:-sp_full}" sp_full
require_recipe_value NORM "${NORM:-doy_full}" doy_full
require_recipe_value TEST_YEARS "${TEST_YEARS:-2024 2025 2026}" "2024 2025 2026"
require_recipe_value TARGET_RELATIVE_TOLERANCE \
  "${TARGET_RELATIVE_TOLERANCE:-0.01}" 0.01

readonly VERSION=txt
readonly RESID_TAG=txt_uq
readonly ARCH=sp_full
readonly NORM=doy_full
readonly TEST_YEARS="2024 2025 2026"
readonly TARGET_RELATIVE_TOLERANCE=0.01
GPUS_OVERRIDE="${GPUS:-}"
GPU_FREE_MEM_MIB="${GPU_FREE_MEM_MIB:-4000}"
KEEP_CACHE="${KEEP_CACHE:-0}"

DONE_MARK="${LOGS_DIR}/txt_uq.done"
mkdir -p "$LOGS_DIR"
RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
MASTER_LOG="${LOGS_DIR}/txt_uq_${RUN_ID}.log"
RUN_MANIFEST="${LOGS_DIR}/txt_uq_${RUN_ID}.manifest.txt"
if [ -e "$DONE_MARK" ]; then
  mv -- "$DONE_MARK" "${LOGS_DIR}/txt_uq_${RUN_ID}.previous.done"
fi
printf '%s\n' \
  "run_id=${RUN_ID}" \
  "version=${VERSION}" \
  "resid_tag=${RESID_TAG}" \
  "architecture=${ARCH}" \
  "normalizer=${NORM}" \
  "test_years=${TEST_YEARS}" \
  "target_relative_tolerance=${TARGET_RELATIVE_TOLERANCE}" \
  "data_root=${TXT_DATA_ROOT}" \
  "cache_root=${TXT_CACHE_ROOT}" > "$RUN_MANIFEST"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }
fail() {
  log "FATAL: $*"
  printf 'exit_code=1 run_id=%s stage=%s\n' "$RUN_ID" "$*" > "$DONE_MARK"
  exit 1
}

run_logged() {
  local name="$1"; shift
  log "START ${name}: $*"
  if "$@" 2>&1 | tee "${LOGS_DIR}/${name}.log"; then
    log "DONE ${name}"
  else
    log "ERROR ${name}; see ${LOGS_DIR}/${name}.log"
    return 1
  fi
}

run_gpu_jobs() {
  local group="$1"; shift
  [ "$#" -gt 0 ] || return 0
  log "START ${group}: $# FIFO jobs on GPUs ${GPUS}"
  if uv run python scripts/run_gpu_jobs.py \
      --gpus "$GPUS" --group "$group" --log-dir "$LOGS_DIR" -- "$@" \
      2>&1 | tee -a "$MASTER_LOG"; then
    log "DONE ${group}"
  else
    log "ERROR ${group}; see ${LOGS_DIR}/${group}_job*_g*.log"
    return 1
  fi
}

detect_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v t="$GPU_FREE_MEM_MIB" -F'[, ]+' '$2 < t {print $1}' | paste -sd, -
}

build_ens_ranges() {
  local total=10 lo=1 i hi count
  local -a gpu_array
  IFS=',' read -r -a gpu_array <<< "$GPUS"
  local n=${#gpu_array[@]}
  for ((i=0; i<n; i++)); do
    count=$(( (total - lo + 1) / (n - i) ))
    [ "$count" -lt 1 ] && continue
    hi=$((lo + count - 1))
    echo "$lo $hi"
    lo=$((hi + 1))
  done
}

remove_cache_if_requested() {
  local cache_dir="$1"
  [ "$KEEP_CACHE" = "1" ] && return 0
  case "$cache_dir" in
    "${TXT_CACHE_ROOT}"/txt_reference_cache_*)
      log "Removing temporary cache ${cache_dir}"
      rm -rf -- "$cache_dir"
      ;;
    *) fail "refusing to remove unexpected cache path ${cache_dir}" ;;
  esac
}

train_reference() {
  local var="$1"
  local cache_dir="${TXT_CACHE_ROOT}/txt_reference_cache_${var}_hres_${NORM}"
  local cache_quoted
  printf -v cache_quoted '%q' "$cache_dir"
  run_logged "txt_referencecache_${var}" \
    uv run training/build_reference_cache.py --var "$var" \
      --cache-dir "$cache_dir"
  export TXT_CACHE_VALIDATED_MANIFEST_SHA256
  TXT_CACHE_VALIDATED_MANIFEST_SHA256=$(
    sha256sum "$cache_dir/manifest.json" | awk '{print $1}'
  )

  local -a commands=()
  for weight in 0; do
    if uv run training/validate_reference_checkpoint.py --var "$var" \
      --version "${VERSION}_reference" \
      --expected-cache-manifest-sha256 "$TXT_CACHE_VALIDATED_MANIFEST_SHA256" \
      --weight "$weight" >/dev/null 2>&1; then
      log "skip ${var} control weight ${weight}: completed checkpoint audited"
      continue
    fi
    commands+=("uv run training/train_txt.py --var ${var} --ensemble 0 --weight ${weight} --gpu {gpu} --ckpt-version ${VERSION}_reference --cache-dir ${cache_quoted}")
  done
  run_gpu_jobs "txt_reference_${var}" "${commands[@]}" || return 1
  remove_cache_if_requested "$cache_dir"

  run_logged "txt_reference_audit_${var}" \
    uv run training/validate_reference_checkpoint.py --var "$var" \
      --version "${VERSION}_reference" \
      --expected-cache-manifest-sha256 "$TXT_CACHE_VALIDATED_MANIFEST_SHA256" --weight 0
}

train_model_bank() {
  local var="$1"
  run_logged "txt_uq_training_${var}" \
    uv run training/train_txt_uq.py --var "$var" --ckpt-version "$VERSION" \
      --gpus "$GPUS" --target-relative-tolerance "$TARGET_RELATIVE_TOLERANCE"
  run_logged "txt_uq_weight_validation_${var}" \
    uv run training/train_txt_uq.py --var "$var" --ckpt-version "$VERSION" \
      --target-relative-tolerance "$TARGET_RELATIVE_TOLERANCE" --audit-only
}

append_residual_jobs() {
  local -n target="$1"
  local var="$2" base
  base="uv run uq/compute_residuals.py --var ${var} --ckpt-version ${VERSION} --resid-tag ${RESID_TAG} --repair-incompatible-artifacts"
  while read -r lo hi; do
    target+=("${base} --gpu-id {gpu} --ens-start ${lo} --ens-end ${hi}")
  done < <(build_ens_ranges)
}

append_sample_jobs() {
  local -n target="$1"
  local var="$2" year="$3" tag="txt_uq_k5" base
  base="uv run uq/sample_txt_uq.py --var ${var} --year ${year} --ckpt-version ${VERSION} --resid-tag ${RESID_TAG} --output-tag ${tag} --thursdays-only --require-complete-eval-dates --repair-incompatible-artifacts"
  while read -r lo hi; do
    target+=("${base} --gpu-id {gpu} --ens-start ${lo} --ens-end ${hi}")
  done < <(build_ens_ranges)
}

if [ -n "$GPUS_OVERRIDE" ]; then
  GPUS="$GPUS_OVERRIDE"
else
  GPUS=$(detect_free_gpus)
fi
[ -n "$GPUS" ] || fail "no free GPU detected; set GPUS explicitly"

log "TxT-UQ: VERSION=${VERSION} ARCH=${ARCH} NORM=${NORM} GPUS=${GPUS}"
log "Target tolerance: abs(J-J0)/J0 <= ${TARGET_RELATIVE_TOLERANCE}"
log "No competition download or submission command is part of this pipeline"

run_logged "txt_uq_tests" uv run pytest -q tests
run_logged "txt_climatology" \
  uv run training/build_climatology.py \
    --start-year 1991 --end-year 2020 --smooth-window 15 --overwrite

for var in mslp t2m; do train_reference "$var" || fail "control ${var}"; done
for var in mslp t2m; do train_model_bank "$var"; done
residual_jobs=()
for var in mslp t2m; do append_residual_jobs residual_jobs "$var"; done
run_gpu_jobs txt_uq_residuals "${residual_jobs[@]}" || fail "residuals"

sample_jobs=()
for year in $TEST_YEARS; do
  for var in mslp t2m; do
    append_sample_jobs sample_jobs "$var" "$year"
  done
done
run_gpu_jobs txt_uq_samples "${sample_jobs[@]}" || fail "samples"

printf 'exit_code=0 run_id=%s done=%s\n' \
  "$RUN_ID" "$(date '+%F %T')" > "$DONE_MARK"
log "TxT-UQ compute pipeline complete."
