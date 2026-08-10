#!/bin/sh
set -e

# Inject host venv packages into the container's Python via PYTHONPATH.
# We deliberately do NOT use the venv's python binary because it is a symlink
# into a snap-managed path that is inaccessible inside Docker. Instead we rely
# on the Triton image's own Python 3.12, which shares the same CPython ABI as
# the host venv, so all compiled extensions (.cpython-312-x86_64-linux-gnu.so)
# load without recompilation.

if [ -d /app/.venv/lib ]; then
  # Find site-packages (e.g. /app/.venv/lib/python3.12/site-packages)
  SITE_PKGS=$(find /app/.venv/lib -maxdepth 3 -type d -name "site-packages" 2>/dev/null | head -1)

  if [ -n "$SITE_PKGS" ]; then
    export PYTHONPATH="${SITE_PKGS}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "==> [entrypoint] PYTHONPATH -> ${SITE_PKGS}"
  else
    echo "==> [entrypoint] WARNING: no site-packages found under /app/.venv/lib"
  fi
else
  echo "==> [entrypoint] WARNING: /app/.venv not mounted or empty"
fi

exec "$@"