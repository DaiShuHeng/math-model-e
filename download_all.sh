#!/usr/bin/env bash
# 一键补齐 clone 后缺失的全部大件：原始数据 / bert 预训练 / 全部权重 / 缓存。
# 私有仓库：github.com 网页路由不认 token，必须走 api.github.com 资产端点。
# 用法: 在仓库根目录执行  TOKEN=<你的GitHub PAT，需repo权限> bash download_all.sh
# 下载量约 3.9GB，解压后约 5.6G 数据 + 2.9G 权重。重复执行会重新下载并覆盖。
set -e
TOKEN="${TOKEN:?请先 export TOKEN=<你的GitHub PAT，需repo权限>}"
API=https://api.github.com/repos/DaiShuHeng/math-model-e/releases/assets
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

ck() {  # ck <文件> <期望SHA256>
  if echo "$2  $1" | shasum -a 256 -c - 2>/dev/null; then return 0; fi
  echo "$2  $1" | sha256sum -c - || { echo "SHA256 不匹配: $1"; exit 1; }
}
dl() {  # dl <资产ID> <临时文件名> <期望SHA256>
  echo "下载 $2 ..."
  curl -fL --retry 3 -H "Authorization: token $TOKEN" -H "Accept: application/octet-stream" \
       -o "$STAGE/$2" "$API/$1"
  ck "$STAGE/$2" "$3" >/dev/null && echo "  校验通过 $2"
}

# ===== Release data-v1：原始数据 + 预训练 + 缓存 =====
# 注：组织方原版 E题数据.zip 内 aligned_50.pkl 压缩流损坏，此为从完好解压树重建的 v2（350 文件全量 CRC 自检通过）
dl 586447895 E题数据.zip ffdeae72ef381c037c30f990743c689bc188d4b1d5a1b313d3adc00258f3b3a0
python3 -m zipfile -e "$STAGE/E题数据.zip" E题/
# 只需修复截断的 aligned_50.pkl 时可单独下小包（735MB，解压到 E题/ 即覆写归位）：
#   dl 587311383 aligned_50_only.zip e34ed0182123665f562400e68ca26a800616b38b2749086402cb7397a838500d
#   python3 -m zipfile -e aligned_50_only.zip E题/

dl 586383566 bert-base-uncased.zip c64485be4735377734d6b2ec0b386f552036c16241a88d13ee4596fe6777c580
python3 -m zipfile -e "$STAGE/bert-base-uncased.zip" models/
dl 586384682 bert-tiny.zip ccb8a51416d0cef15636628ad169a7c21a7a0fdabe6bbfda75b96b576b6e839f
python3 -m zipfile -e "$STAGE/bert-tiny.zip" models/

dl 586384865 cache_misc.zip 7fb55ccd4e1c3698baf008b00606cdaddf12d261d108a08b9e5423ad8f22ae24
python3 -m zipfile -e "$STAGE/cache_misc.zip" .

# ===== Release weights-v1：冠军 3 种子（直接可用）=====
mkdir -p solution/weights/q2_base_cw/s2026 solution/weights/q2_base_cw/s7 solution/weights/q2_base_cw/s42
dl 586327300 s2026.pt 405abcdcecf39defc49f6e33ba041d2fb42ebc681381dd51d1e670d3f7199968
mv "$STAGE/s2026.pt" solution/weights/q2_base_cw/s2026/best.pt
dl 586323833 s7.pt afe0dc7d318699e02902167ef63cc9436c18eaded2d7d21fbf1b7f5da6477658
mv "$STAGE/s7.pt" solution/weights/q2_base_cw/s7/best.pt
dl 586325543 s42.pt e2cee7228e7107b61f109237f982ce7179419dd7c38a4d0f06761e4001bb07de
mv "$STAGE/s42.pt" solution/weights/q2_base_cw/s42/best.pt

# ===== Release weights-v1：其余 9 组消融/基线权重 =====
dl 586379620 q2_ablation_weights.zip e432c176c9622a64d5ccba2cf52379db5f7f363d69642d402e3d5ed7ad12328a
python3 -m zipfile -e "$STAGE/q2_ablation_weights.zip" solution/weights/

echo
echo "全部完成。抽查："
ls -lh E题/E题数据 models/bert-base-uncased models/bert-tiny \
      solution/weights/q2_base_cw/s2026/best.pt solution/weights/q2_base_cw_noaug/s2026/best.pt
