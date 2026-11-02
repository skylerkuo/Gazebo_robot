#!/usr/bin/env bash
# Run the official image in an unprivileged user/mount/PID namespace.
set -euo pipefail
INSERT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
INSERT_ROOTFS="$INSERT_ROOT/runtime/rootfs"
if [[ ! -f "$INSERT_ROOT/runtime/rootfs.ready" ]]; then
    echo 'Run python3 robot_insert/scripts/prepare_image.py first.' >&2
    exit 1
fi
if [[ ${1:-} != --inside ]]; then
    exec unshare --user --map-root-user --mount --pid --fork --kill-child=SIGKILL \
        "$INSERT_ROOT/scripts/enter.sh" --inside "$@"
fi
shift
mount --make-rprivate /
mkdir -p "$INSERT_ROOT/runtime/home" "$INSERT_ROOT/runtime/tmp/.X11-unix"
mkdir -p "$INSERT_ROOTFS"/{dev,proc,sys,root,tmp,workspace,mnt/wslg,usr/lib/wsl}
mount --rbind /dev "$INSERT_ROOTFS/dev"
mount --rbind /sys "$INSERT_ROOTFS/sys"
mount -t proc proc "$INSERT_ROOTFS/proc"
mount --bind "$INSERT_ROOT/runtime/home" "$INSERT_ROOTFS/root"
mount --bind "$INSERT_ROOT/runtime/tmp" "$INSERT_ROOTFS/tmp"
mount --bind "$INSERT_ROOT" "$INSERT_ROOTFS/workspace"
if [[ -n ${INSERT_RESULTS_DIR:-} ]]; then
    if [[ ! -d "$INSERT_RESULTS_DIR" ]]; then
        echo 'The configured inference results directory does not exist.' >&2
        exit 1
    fi
    mount --bind "$INSERT_RESULTS_DIR" "$INSERT_ROOTFS/workspace/results"
fi
# Optional read-only access to the existing ML environment. ROS stays in Kilted;
# learned policies can use the already verified PyTorch/CUDA wheels without an install.
if [[ ${INSERT_WITH_ML:-false} == true ]]; then
    INSERT_VENV="$INSERT_ROOT/../.venv"
    if [[ ! -d "$INSERT_VENV/lib/python3.12/site-packages/torch" ]]; then
        echo 'The workspace .venv with PyTorch is required for learned policies.' >&2
        exit 1
    fi
    mkdir -p "$INSERT_ROOTFS/host_venv"
    mount --bind "$INSERT_VENV" "$INSERT_ROOTFS/host_venv"
    mount -o remount,bind,ro "$INSERT_ROOTFS/host_venv"
fi
# OCI images omit files normally injected by Docker; Zenoh needs localhost.
for INSERT_ETC_FILE in hosts hostname resolv.conf; do
    if [[ -f "/etc/$INSERT_ETC_FILE" ]]; then
        if [[ -L "$INSERT_ROOTFS/etc/$INSERT_ETC_FILE" ]]; then
            rm -- "$INSERT_ROOTFS/etc/$INSERT_ETC_FILE"
        fi
        touch "$INSERT_ROOTFS/etc/$INSERT_ETC_FILE"
        mount --bind "/etc/$INSERT_ETC_FILE" "$INSERT_ROOTFS/etc/$INSERT_ETC_FILE"
        mount -o remount,bind,ro "$INSERT_ROOTFS/etc/$INSERT_ETC_FILE"
    fi
done
if [[ -d /usr/lib/wsl ]]; then
    mount --rbind /usr/lib/wsl "$INSERT_ROOTFS/usr/lib/wsl"
fi
if [[ -d /mnt/wslg ]]; then
    mount --rbind /mnt/wslg "$INSERT_ROOTFS/mnt/wslg"
fi
if [[ -d /tmp/.X11-unix ]]; then
    mount --bind /tmp/.X11-unix "$INSERT_ROOTFS/tmp/.X11-unix"
fi
INSERT_GRAPHICS_ENV=()
if [[ -e /dev/dxg ]]; then
    # WSLg otherwise selects llvmpipe even when CUDA is available.
    INSERT_GRAPHICS_ENV=("GALLIUM_DRIVER=${GALLIUM_DRIVER:-d3d12}" "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA")
fi
exec chroot "$INSERT_ROOTFS" /usr/bin/env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/root USER=root LOGNAME=root LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    ROS_DISTRO=kilted ROS_DOMAIN_ID=84 \
    DISPLAY="${DISPLAY:-:0}" WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}" \
    XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir QT_X11_NO_MITSHM=1 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    "${INSERT_GRAPHICS_ENV[@]}" \
    "$@"
