
#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# Install a pinned Git LFS release if it is not already available.
if ! command -v git-lfs >/dev/null 2>&1; then
  mkdir -p "$HOME/.local/bin" /tmp/git-lfs-dist

  LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz"

  curl -fL --retry 3 --retry-delay 2 \
    "$LFS_URL" -o /tmp/git-lfs.tar.gz

  tar -xzf /tmp/git-lfs.tar.gz -C /tmp/git-lfs-dist

  LFS_BIN="$(find /tmp/git-lfs-dist -type f -name git-lfs -perm -u+x | head -n 1)"
  test -n "$LFS_BIN"

  install -m 755 "$LFS_BIN" "$HOME/.local/bin/git-lfs"
  export PATH="$HOME/.local/bin:$PATH"
fi

# Download the actual model files from Git LFS.
git lfs install --local
git lfs pull

# Verify that all required model files are present and not LFS pointers.
python - <<'PY'
from pathlib import Path

required = (
    "cnn_feature_extractor.keras",
    "fake_discriminator.keras",
    "lstm_attention_v1_best.keras",
    "sad_neutral_adapter.keras",
)

for name in required:
    path = Path("models") / name

    if not path.is_file():
        raise SystemExit(f"Required model is missing: {path}")

    with path.open("rb") as model_file:
        header = model_file.read(128)

    if b"git-lfs.github.com/spec/v1" in header or path.stat().st_size < 100_000:
        raise SystemExit(
            f"{path} is still a Git LFS pointer or is unexpectedly small. "
            "Check Git LFS access and the repository's LFS objects."
        )

    print(f"Verified {path}: {path.stat().st_size:,} bytes")
PY

python -m pip install -r requirements-render.txt

