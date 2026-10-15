#!/usr/bin/env bash
# Build the release wheel of flashspec-asm-kernel inside a manylinux2014
# container, then `auditwheel repair` it to a manylinux tag.
#
# Usage (from anywhere, needs docker):
#     scripts/build_wheel.sh
# Output:
#     wheelhouse/flashspec_asm_kernel-<ver>-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#
# Environment:
#     MANYLINUX_IMAGE  container image (default: quay.io/pypa/manylinux2014_x86_64)
#     PLAT             auditwheel target platform (default: manylinux2014_x86_64)
#     DOCKER_EXTRA_ARGS  extra `docker run` arguments, word-split (e.g. to
#                      mount a CA bundle behind a TLS-intercepting proxy)
#
# The host part only starts the container; everything else runs inside it
# (`--in-container`). The build happens in a copy of the source tree so that
# no stale host build (e.g. a -march=native libflashspec_asm.so) can leak in,
# and so root-owned files are not written into the checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux2014_x86_64}"
PLAT="${PLAT:-manylinux2014_x86_64}"

if [[ "${1:-}" != "--in-container" ]]; then
    mkdir -p "$ROOT/wheelhouse"
    # shellcheck disable=SC2086
    exec docker run --rm ${DOCKER_EXTRA_ARGS:-} \
        -e PLAT="$PLAT" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -v "$ROOT":/io:ro \
        -v "$ROOT/wheelhouse":/wheelhouse \
        --entrypoint /bin/bash \
        "$IMAGE" /io/scripts/build_wheel.sh --in-container
fi

# ---------------------------------------------------------------- container
PY=/opt/python/cp312-cp312/bin/python

if ! command -v nasm >/dev/null; then
    yum install -y -q nasm
fi
echo "== toolchain"
grep -h . /etc/redhat-release 2>/dev/null || true
ldd --version | sed -n 1p
cc --version | sed -n 1p
nasm -v
auditwheel --version

SRC=$(mktemp -d)
WORK=$(mktemp -d)
# Tracked sources only; build products from the host are left behind.
tar -C /io --exclude=.git --exclude=build --exclude=dist \
    --exclude=wheelhouse --exclude='*.so' --exclude='*.o' -cf - . \
    | tar -C "$SRC" -xf -

echo "== make (portable flags)"
make -C "$SRC" clean
make -C "$SRC" all

echo "== wheel"
"$PY" -m pip wheel --no-deps -w "$WORK/dist" "$SRC"

echo "== auditwheel repair"
# --only-plat: tag exactly $PLAT. The library needs only GLIBC_2.2.5, so
# auditwheel would otherwise also stamp manylinux1/manylinux_2_5 on it, a
# claim about CentOS 5-era systems that nothing here builds or tests on.
auditwheel repair --only-plat --plat "$PLAT" -w "$WORK/repaired" "$WORK"/dist/*.whl
auditwheel show "$WORK"/repaired/*.whl

cp "$WORK"/repaired/*.whl /wheelhouse/
if [[ -n "${HOST_UID:-}" ]]; then
    chown "$HOST_UID:${HOST_GID:-$HOST_UID}" /wheelhouse/*.whl
fi
ls -l /wheelhouse
