#!/usr/bin/env bash
# 下载冠军权重（q2_base_cw 3 种子）到 solution/weights/q2_base_cw/<seed>/best.pt
# 私有仓库：github.com 网页路由不认 token，必须走 api.github.com 资产端点。
# 用法: TOKEN=<你的PAT> bash download_weights.sh
set -e
TOKEN="${TOKEN:?请先 export TOKEN=<你的GitHub PAT，需repo权限>}"
API=https://api.github.com/repos/DaiShuHeng/math-model-e/releases/assets

dl() {  # dl <资产ID> <目标路径>
  mkdir -p "$(dirname "$2")"
  echo "下载 $2 ..."
  curl -fL --retry 3 -H "Authorization: token $TOKEN" \
       -H "Accept: application/octet-stream" -o "$2" "$API/$1"
}

dl 586327300 solution/weights/q2_base_cw/s2026/best.pt
dl 586323833 solution/weights/q2_base_cw/s7/best.pt
dl 586325543 solution/weights/q2_base_cw/s42/best.pt

echo "完成。校验（应为 440MB 级）："
ls -lh solution/weights/q2_base_cw/*/best.pt
echo "SHA256 应为："
echo "  405abcdcecf39defc49f6e33ba041d2fb42ebc681381dd51d1e670d3f7199968  s2026/best.pt"
echo "  afe0dc7d318699e02902167ef63cc9436c18eaded2d7d21fbf1b7f5da6477658  s7/best.pt"
echo "  e2cee7228e7107b61f109237f982ce7179419dd7c38a4d0f06761e4001bb07de  s42/best.pt"
shasum -a 256 solution/weights/q2_base_cw/*/best.pt 2>/dev/null || sha256sum solution/weights/q2_base_cw/*/best.pt
