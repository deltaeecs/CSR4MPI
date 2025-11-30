#!/usr/bin/env bash
set -euo pipefail
# 下载若干 Matrix Market 矩阵到 tests/matrices 目录
# 依赖: curl / tar
# 使用: bash scripts/download_matrices.sh
# 注意: SuiteSparse 站点近期域名/路径可能发生变化，若出现 404 请手动访问主页检索并更新 URL。
# WSL NAT + 本地代理 (http_proxy=localhost) 会导致访问失败，请在脚本内显式取消相关代理。

# 取消可能的代理环境变量以避免 WSL localhost 代理导致的 404 / 连接失败
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
MAT_DIR="$ROOT_DIR/tests/matrices"
mkdir -p "$MAT_DIR"

# SuiteSparse Matrix Collection (原 UFL)。旧公开说明: 打包文件可能位于
#   https://sparse.tamu.edu/MM/<Group>/<Matrix>.tar.gz
# 若该形式返回 404，可尝试备用镜像或手动下载。
# 这里保留原模式，失败时给出提示。

LIST_FILE="$ROOT_DIR/scripts/matrices.txt"
if [ -f "$LIST_FILE" ]; then
  echo "Using URL list: $LIST_FILE"
  while IFS= read -r url; do
    # skip comments/empty
    [[ -z "$url" || "$url" =~ ^# ]] && continue
    fname="$(basename "$url")"
    base="${fname%.tar.gz}"
    echo "==> Fetching $base from $url"
    tmpTar="${MAT_DIR}/${fname}"
    if curl -fsSL "$url" -o "$tmpTar"; then
      echo "    Extracting $tmpTar"
      tar -xzf "$tmpTar" -C "$MAT_DIR" || { echo "    Extraction failed"; rm -f "$tmpTar"; continue; }
      if [ -f "$MAT_DIR/$base/$base.mtx" ]; then
        mv "$MAT_DIR/$base/$base.mtx" "$MAT_DIR/${base}.mtx"
        echo "    Saved $MAT_DIR/${base}.mtx"
      else
        # try find any .mtx inside extracted dir
        found=$(find "$MAT_DIR/$base" -maxdepth 2 -type f -name '*.mtx' | head -n1)
        if [ -n "$found" ]; then
          tgt="$MAT_DIR/${base}.mtx"
          mv "$found" "$tgt"
          echo "    Saved $tgt"
        else
          echo "    .mtx not found inside archive"
        fi
      fi
      rm -rf "$MAT_DIR/$base" "$tmpTar"
    else
      echo "    Download failed for $base (URL 404 或代理问题)"
    fi
  done < "$LIST_FILE"
else
  echo "No list file found at $LIST_FILE. Add URLs to download."
fi

echo "Done. Available matrices:";
ls -1 "$MAT_DIR" | grep -E '\.mtx$' || true
