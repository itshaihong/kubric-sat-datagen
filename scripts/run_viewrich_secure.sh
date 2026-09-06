#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image='kubricdockerhub/kubruntu@sha256:a4152c8066ffbd7bd303e4ea79c3ce250f190368cb07479fb6763308f165a17f'

docker_args=(
  run
  --rm
  --interactive
  --platform linux/amd64
  --user "$(id -u):$(id -g)"
  --volume "${repo_root}:/kubric:rw"
  --workdir /kubric
)

if [[ -n "${FASTSAM_DIR:-}" ]]; then
  docker_args+=(--volume "${FASTSAM_DIR}:/FastSAM:ro")
fi

if [[ $# -eq 0 ]]; then
  set -- \
    generate_spacecraft_orbit_viewrich.py \
    --trajectory fibonacci \
    --num-snapshots 96 \
    --orbit-radius 10.0 \
    --max-abs-elevation 70 \
    --lighting cv_bright \
    --seed 0
fi

script_name="$1"
shift

exec docker "${docker_args[@]}" "${image}" /usr/bin/python3 "/kubric/${script_name}" "$@"
