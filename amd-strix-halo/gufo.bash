gufo() {
  podman run --rm -it --network host \
    --userns="keep-id:uid=$(id -u),gid=$(id -g)" --group-add keep-groups \
    --device /dev/kfd --device /dev/dri --ulimit memlock=-1 \
    -v ./models:/models:ro \
    -v gufo-extensions:/home/gufo/.local/share/gufo/extensions \
    ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo "$@"
}

gufo "$@"
