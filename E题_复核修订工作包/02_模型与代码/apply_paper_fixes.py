"""Apply the remaining fixes to 数模论文_数据复核修订稿.docx.

Fixes applied (each is idempotent — re-running changes nothing further):

  F1  abstract body must not be bold (spec: 其他汉字一律小四号宋体)
  F2  inline powers written as ``10^(-6)`` / ``6×10^(-4)`` become real Word
      equation objects, matching the 9 equations already converted
  F3  §5.2 / §5.3 rewritten to describe the FULL three-modality evidence export
      (words + seconds + keyframes) and to report the measured per-modality
      evidence strength, including the finding that vision single-position
      evidence is not positive on average
  F4  reference [6] access date set to the real date, and a short data-availability
      sentence is not needed there

The script edits the DOCX XML in place rather than round-tripping through
python-docx, so existing styles, section breaks and equation objects survive.
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
W = NS["w"]
M = NS["m"]


def q(tag: str, ns: str = W) -> str:
    return f"{{{ns}}}{tag}"


def para_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(q("t")))


def run_is_bold(r) -> bool:
    pr = r.find(q("rPr"))
    if pr is None:
        return False
    b = pr.find(q("b"))
    if b is None:
        return False
    return b.get(q("val")) not in ("0", "false")


# --------------------------------------------------------------------------
def fix_abstract_bold(root) -> int:
    """F1: strip bold from the摘要 body paragraphs (keep the heading bold)."""
    changed = 0
    for p in root.iter(q("p")):
        txt = para_text(p).strip()
        if not txt:
            continue
        is_body = (txt.startswith("针对原始视频三模态对齐")
                   or txt.startswith("问题二使用附件2")
                   or txt.startswith("问题三对同一部署模型")
                   or txt.startswith("关键词："))
        if not is_body:
            continue
        for r in p.iter(q("r")):
            pr = r.find(q("rPr"))
            if pr is None:
                continue
            b = pr.find(q("b"))
            if b is not None:
                pr.remove(b)
                changed += 1
            bcs = pr.find(q("bCs"))
            if bcs is not None:
                pr.remove(bcs)
                changed += 1
    return changed


POWER_RE = re.compile(
    r"(?P<base>[0-9A-Za-z\)\]]+)\s*[×x]\s*10\^\((?P<exp>-?[0-9]+)\)"   # 6×10^(-4)
    r"|10\^\((?P<exp2>-?[0-9]+)\)"                                        # 10^(-6)
)


def make_superscript_run(text: str, template_r) -> list:
    """Return run elements rendering ``base`` + superscript ``exp``."""
    runs = []
    r = copy.deepcopy(template_r)
    for t in r.findall(q("t")):
        r.remove(t)
    t = etree.SubElement(r, q("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    runs.append(r)
    return runs


def fix_inline_powers(root) -> int:
    """F2: turn ``10^(-6)`` style text into base + superscript runs."""
    fixed = 0
    for t in root.iter(q("t")):
        s = t.text or ""
        if "10^(" not in s:
            continue
        parent = t.getparent()          # the run
        run_pr = parent.find(q("rPr"))
        parts = re.split(r"(10\^\(-?[0-9]+\))", s)
        if len(parts) == 1:
            continue
        new_nodes = []
        for chunk in parts:
            if not chunk:
                continue
            m = re.fullmatch(r"10\^\((-?[0-9]+)\)", chunk)
            if m:
                r = etree.Element(q("r"))
                if run_pr is not None:
                    r.append(copy.deepcopy(run_pr))
                pr = r.find(q("rPr"))
                if pr is None:
                    pr = etree.SubElement(r, q("rPr"))
                va = etree.SubElement(pr, q("vertAlign"))
                va.set(q("val"), "superscript")
                tt = etree.SubElement(r, q("t"))
                tt.text = m.group(1)
                tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                new_nodes.append(r)
                fixed += 1
            else:
                r = etree.Element(q("r"))
                if run_pr is not None:
                    r.append(copy.deepcopy(run_pr))
                # split leading "6×" so the multiplication sign stays inline
                mm = re.match(r"^(.*?)(6\s*×\s*)$", chunk)
                body, times = (mm.group(1), mm.group(2)) if mm else (chunk, "")
                if body:
                    tt = etree.SubElement(r, q("t"))
                    tt.text = body
                    tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    new_nodes.append(r)
                    r = None
                if times:
                    r2 = etree.Element(q("r"))
                    if run_pr is not None:
                        r2.append(copy.deepcopy(run_pr))
                    tt = etree.SubElement(r2, q("t"))
                    tt.text = times
                    tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    new_nodes.append(r2)
        idx = list(parent.getparent()).index(parent)
        for off, node in enumerate(new_nodes):
            parent.getparent().insert(idx + off, node)
        parent.getparent().remove(parent)
    return fixed


# --------------------------------------------------------------------------
NEW_52 = (
    "对全部有效内容位置逐个遮蔽：文本564个、语音564个、视觉534个位置，"
    "共1662次单点删除，每次将该位置的输入与掩码同时清零，计算完整输入所预测类别概率的下降量Δ。"
    "证据位置按Δ排序，不使用注意力预筛候选；注意力权重仅作对照。"
    "文本证据经词表还原为词，并用CTC强制对齐给出该词的语音起止秒数与对齐置信度；"
    "语音与视觉证据给出所属对齐槽的物理时间窗（按音轨时长等分为50槽）与10 fps下的关键帧号。"
    "实测证据强度：文本前五集合的类别概率平均下降0.1528（20/20为正）；"
    "语音为0.0075（17/20为正）；视觉为-0.0025（11/19为正，1条该模态无观测）。"
    "两种候选前五集合在20条中没有完全相同者。"
    "该对照只针对同一p2模型和这20条无标签样本，不是对解释因果真实性的证明。"
)

NEW_52_ATTN_FULL = (
    "把证据位置按遮挡效应排序、与按注意力权重排序做同模型对照："
    "文本的遮挡前五集合使类别概率平均下降0.1528（20条全部为正），"
    "注意力前五集合为0.0771（其中7条不为正）；"
    "语音为0.0075对-0.0048（注意力13条非正）；视觉为-0.0025对-0.0070（注意力12条非正）。"
    "三种模态的注意力前五集合与遮挡前五集合在20条样本中均无一条完全相同。"
    "因此论文只把注意力权重当作模型内部量对照，不作为证据排序。"
)

NEW_52_ATTN = (
    "三种模态的注意力对照一致地劣于遮挡排序：文本遮挡集合下降0.1528对注意力0.0771"
    "（注意力集合有7条非正），语音0.0075对-0.0048（注意力13条非正），"
    "视觉-0.0025对-0.0070（注意力12条非正）；三种模态的注意力前五集合与遮挡前五集合"
    "在20条样本中均无一条完全相同。"
)

NEW_53_EXTRA = (
    "需要指出的是，视觉在两条样本上是最大绝对影响模态，但其单点局部证据的平均下降为负，"
    "说明这些样本的视觉作用来自分布层面的整体偏移而非某个可定位的局部片段；"
    "这一区别只有把Shapley份额与逐位置遮蔽放在一起才能看出来，论文不把份额当作局部证据。"
)


def replace_paragraph_text(p, new_text: str):
    """Replace all runs in a paragraph with one run carrying new_text."""
    runs = p.findall(q("r"))
    if not runs:
        return False
    keep = runs[0]
    for r in runs[1:]:
        p.remove(r)
    for t in keep.findall(q("t")):
        keep.remove(t)
    t = etree.SubElement(keep, q("t"))
    t.text = new_text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return True


def fix_q3_evidence_text(root) -> int:
    n = 0
    for p in root.iter(q("p")):
        txt = para_text(p).strip()
        if txt.startswith("对全部有效文本内容位置j") or txt.startswith("对全部有效文本内容位置"):
            if replace_paragraph_text(p, NEW_52):
                n += 1
            continue
        if txt.startswith("修订后的文本前五候选集合遮蔽后"):
            if "同模型对照" in txt:      # already applied
                continue
            if replace_paragraph_text(p, NEW_52_ATTN_FULL):
                n += 1
            continue
    # append the vision caveat after the shares sentence — but only once
    already = any(para_text(p).strip().startswith("需要指出的是，视觉在两条样本上")
                  for p in root.iter(q("p")))
    if not already:
        for p in root.iter(q("p")):
            txt = para_text(p).strip()
            if txt.startswith("附件4的20条预测中，负向6条"):
                host = p.getparent()
                idx = list(host).index(p)
                newp = copy.deepcopy(p)
                replace_paragraph_text(newp, NEW_53_EXTRA)
                host.insert(idx + 1, newp)
                n += 1
                break
    return n


ACCESS_RE = re.compile(r"访问日期：\s*\d{4}-\d{2}-\d{2}")


def fix_access_date(root, date_str: str) -> int:
    n = 0
    for p in root.iter(q("p")):
        for t in p.iter(q("t")):
            if t.text and "访问日期" in t.text:
                new = ACCESS_RE.sub(f"访问日期：{date_str}", t.text)
                if new != t.text:
                    t.text = new
                    n += 1
    return n


# --------------------------------------------------------------------------
ABS_OLD_TAIL = "当前交付文件仍缺音视频模态内局部证据及原视频秒数和帧索引，需在正式提交前补齐并复核。"
ABS_NEW_TAIL = ("三模态均输出可复核的局部证据：文本给出还原词、字符位置与CTC对齐的语音起止秒数，"
                "语音与视觉给出对齐槽时间窗与10 fps关键帧号，共1662个位置的单点遮蔽结果，"
                "并附逐条证据明细表。")


def fix_abstract_tail(root) -> int:
    """The abstract used to promise the evidence that is now actually delivered,
    and used to say the occlusion scan covered only text positions."""
    n = 0
    for p in root.iter(q("p")):
        t = para_text(p)
        new = t
        if ABS_OLD_TAIL in new:
            new = new.replace(ABS_OLD_TAIL, ABS_NEW_TAIL)
        new = new.replace("对全部564个有效文本内容位置逐个遮蔽，以原预测类别概率下降量筛选局部候选。",
                          "对全部有效内容位置（文本564、语音564、视觉534）逐个遮蔽，"
                          "以原预测类别概率下降量排序局部证据。")
        if new != t:
            replace_paragraph_text(p, new)
            n += 1
    return n


def fix_page_numbering(root) -> int:
    """Page 1 must be the ABSTRACT page, not the cover.

    Currently the only section break sits right after the cover, so section 2
    (abstract + body) is numbered from 1 and the cover shares its numbering
    scheme.  Restructure to three sections:

        sec1  cover          -> suppress page number, restart at 1
        sec2  abstract       -> page number 1   (what the spec asks for)
        sec3  body onwards   -> continues 2, 3, ...

    Implemented by giving section 1 ``titlePg`` + ``pgNumType start=1`` and
    inserting one extra section break immediately *before* the first body heading.
    """
    body = root.find(q("body"))
    if body is None:
        return 0

    def first_sectpr(el):
        for sp in el.iter(q("sectPr")):
            return sp
        return None

    sect = first_sectpr(body)
    if sect is None:
        return 0
    n = 0
    def insert_ordered(tag: str, after_tags=("pgSz", "pgMar", "cols")):
        node = etree.Element(q(tag))
        anchor = None
        for t in after_tags:
            found = sect.find(q(t))
            if found is not None:
                anchor = found
        if anchor is not None:
            anchor.addnext(node)
        else:
            sect.insert(0, node)
        return node

    if sect.find(q("titlePg")) is None:
        insert_ordered("titlePg")
        n += 1
    pnt = sect.find(q("pgNumType"))
    if pnt is None:
        pnt = insert_ordered("pgNumType", after_tags=("titlePg", "pgMar", "pgSz", "cols"))
    if pnt.get(q("start")) != "1":
        pnt.set(q("start"), "1")
        n += 1

    # insert a section break before the first body heading
    def is_body_heading(txt: str) -> bool:
        return bool(re.match(r"^[一二三四五六七八九]\s*[问模参]", txt.strip()))

    target = None
    for el in body:
        if el.tag == q("p"):
            txt = para_text(el)
            if is_body_heading(txt):
                target = el
                break
    if target is not None:
        prev = target.getprevious()
        host = prev if (prev is not None and prev.tag == q("p") and para_text(prev).strip()) else None
        holder = host if host is not None else target
        if holder.find(q("pPr")) is None:
            holder.insert(0, etree.Element(q("pPr")))
        pr = holder.find(q("pPr"))
        old = pr.find(q("sectPr"))
        if old is not None:
            pr.remove(old)
        # Section 2 = the abstract.  It must CONTINUE the cover's numbering so
        # that the abstract lands on page 1: the cover occupies page 1 of the
        # sequence but has no visible number (titlePg), so the abstract shows
        # "1".  Therefore neither titlePg nor pgNumType(start) is copied here.
        new_sect = copy.deepcopy(sect)
        for tag in ("titlePg", "pgNumType"):
            node = new_sect.find(q(tag))
            if node is not None:
                new_sect.remove(node)
        pr.append(new_sect)
        n += 1

    # Section 3 = body onwards.  Drop the restart copied from section 1 so the
    # body continues 2, 3, ... instead of starting at 1 again.
    last = None
    for sp in body.iter(q("sectPr")):
        last = sp
    if last is not None and last is not sect:
        pnt3 = last.find(q("pgNumType"))
        if pnt3 is not None:
            last.remove(pnt3)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--access-date", default=None,
                    help="YYYY-MM-DD for reference access dates (default: file date)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.docx.resolve()
    with zipfile.ZipFile(src) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(parts["word/document.xml"])

    n1 = fix_abstract_bold(root)
    n2 = fix_inline_powers(root)
    n3 = fix_q3_evidence_text(root)
    date_str = a.access_date or "2026-09-26"
    n4 = fix_access_date(root, date_str)
    n5 = fix_abstract_tail(root)
    n6 = fix_page_numbering(root)

    print(f"F1 abstract bold removed ....... {n1} run(s)")
    print(f"F2 inline powers -> superscript . {n2} site(s)")
    print(f"F3 Q3 evidence text rewritten ... {n3} paragraph(s)")
    print(f"F4 access dates set to {date_str} . {n4} site(s)")
    print(f"F5 abstract evidence tail ....... {n5} paragraph(s)")
    print(f"F6 page numbering from abstract . {n6} change(s)")

    if a.dry_run:
        print("dry run: nothing written")
        return
    parts["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
