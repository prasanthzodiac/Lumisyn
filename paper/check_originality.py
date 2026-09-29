"""Self-check for a DOCX manuscript: word-sequence overlap with reference texts, AI-style word list, sentence rhythm.

    python paper/check_originality.py paper/Lumisyn_DP_FlowMatching_Manuscript.docx
Reference texts: every *.txt in paper/refs_text/. This is not a substitute for iThenticate.
"""
import html
import re
import statistics
import sys
import zipfile
from pathlib import Path

BANNED = ["delve", "crucial", "pivotal", "landscape", "tapestry", "leverage", "seamless", "furthermore", "moreover",
          "additionally", "notably", "comprehensive", "intricate", "realm", "it is worth noting", "in conclusion", "robust",
          "harness", "unlock", "foster", "showcase", "novel", "utilize", "utilise", "myriad", "paramount", "underscore",
          "groundbreaking", "cutting-edge", "state-of-the-art", "meticulous", "navigate", "multifaceted", "holistic",
          "in today's", "plays a vital role", "a testament to", "shed light", "elevate", "embark", "bustling", "vibrant"]


def docx_text(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = [html.unescape(re.sub(r"<[^>]+>", "", "".join(re.findall(r"<w:t[^>]*>[^<]*</w:t>", p)))) for p in xml.split("</w:p>")]
    paras = [p for p in paras if p.strip()]
    for i, p in enumerate(paras):  # drop the reference list
        if p.strip() == "References":
            paras = paras[:i]
            break
    return paras


def words(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def grams(w, n):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    doc = Path(sys.argv[1])
    paras = docx_text(doc)
    body = " ".join(paras)
    w = words(body)
    print(f"# Self-check of {doc.name}\n\nBody words (without references): {len(w)}")
    abstract = next((p for p in paras if p.startswith("Differentially private") or p.startswith("Abstract")), "")
    if abstract:
        print(f"Abstract words: {len(abstract.split())}")
    print("\n## Shared word sequences with reference texts\n\n| Reference | 5-word | 8-word | 12-word |\n|---|---|---|---|")
    long_hits = []
    for ref in sorted((Path(__file__).parent / "refs_text").glob("*.txt")):
        rw = words(ref.read_text(encoding="utf-8", errors="ignore"))
        row = []
        for n in (5, 8, 12):
            shared = grams(w, n) & grams(rw, n)
            row.append(f"{len(shared)} ({100 * len(shared) / max(1, len(w) - n + 1):.2f}%)")
            if n == 8:
                long_hits += [(ref.stem, " ".join(g)) for g in sorted(shared)]
        print(f"| {ref.stem} | " + " | ".join(row) + " |")
    print("\n8-word matches:" + ("".join(f"\n- {r}: \"{g}\"" for r, g in long_hits) if long_hits else " none"))
    low = body.lower()
    hits = {b: len(re.findall(r"\b" + re.escape(b) + r"\b", low)) for b in BANNED}
    hits = {k: v for k, v in hits.items() if v}
    print(f"\n## Style\n\nAI-style word list hits: {hits if hits else 0}")
    print(f"Em-dashes: {body.count(chr(0x2014))}")
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body) if len(s.split()) > 2]
    lens = [len(s.split()) for s in sents]
    print(f"Sentences: {len(sents)}, mean {statistics.mean(lens):.1f} words, SD {statistics.stdev(lens):.1f}, "
          f"short (<10) {sum(l < 10 for l in lens)}, long (>35) {sum(l > 35 for l in lens)}")


if __name__ == "__main__":
    main()
