// Builds a manuscript .docx from a content module and the figures in fig/.
//   node build_docx.js [content.js] [output.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, LevelFormat, TabStopType, Footer, PageNumber, VerticalAlign,
  Math: OMath, MathRun, MathFraction, MathSubScript, MathSuperScript, MathSum, PageBreak,
} = require("docx");

const contentFile = process.argv[2] || "./content.js";
const outName = process.argv[3] || "Lumisyn_DP_FlowMatching_Manuscript.docx";
const { B, REFS } = require(path.resolve(__dirname, contentFile));

const FONT = "Times New Roman", MATHFONT = "Cambria Math", MONO = "Consolas";
const PAGE_W = 11906, MARGIN = 1440, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 1-inch margins

// ---------------------------------------------------------------- native Word equations (OMML)
const R = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [R(b)], subScript: [R(s)] });
const sup = (b, s) => new MathSuperScript({ children: [R(b)], superScript: [R(s)] });
const EQ = {
  dpsgd: () => [R("S̃(B) = "), sub("∑", "i∈B"), sub("ḡ", "i"), R(" + 𝒩(0, "), sup("σ", "2"), sup("C", "2"), R("I),    "),
    sub("ḡ", "i"), R(" = "), sub("g", "i"), R(" · min(1, "), new MathFraction({ numerator: [R("C")], denominator: [R("‖"), sub("g", "i"), sub("‖", "2")] }), R(")")],
  ot: () => [R("‖S(B) − S(B′)"), sub("‖", "2"), R(" = ‖"), sub("∑", "i∈{j}∪R"), R("("), sub("ḡ", "i"), R("(B) − "), sub("ḡ", "i"), R("(B′))"),
    sub("‖", "2"), R(" ≤ 2C(k + 1)")],
  dapc: () => [sup("ε", "∗"), R(" = "), sub("ε", "max"), R(" − ("), sub("ε", "max"), R(" − "), sub("ε", "min"), R(") · "),
    new MathFraction({ numerator: [R("1 − "), sup("e", "−βR")], denominator: [R("1 − "), sup("e", "−β")] })],
  replace: () => [R("(ε, δ) add/remove  ⟹  (2ε, (1 + "), sup("e", "ε"), R(")δ) replace-one")],
};

// ---------------------------------------------------------------- citations (numbered by first use)
const order = [];
function cite(keys) {
  const nums = keys.map((k) => {
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  }).sort((a, b) => a - b);
  const parts = []; let i = 0;
  while (i < nums.length) {
    let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    parts.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : nums.slice(i, j + 1).join(", "));
    i = j + 1;
  }
  return `[${parts.join(", ")}]`;
}

// ---------------------------------------------------------------- inline markup -> runs
const MATHCH = /([∈∉≃≤≥∑Σ∏Π‖∗∪∩≈⟹→∞√⌈⌉∼])/u;
function textRuns(text, base) {  // symbols that Times New Roman may lack go to Cambria Math
  return text.split(MATHCH).filter((s) => s).map((s) => new TextRun({ ...base, text: s, ...(MATHCH.test(s) ? { font: MATHFONT } : {}) }));
}
function scriptRuns(text, base) {
  const runs = []; let buf = "";
  const flush = () => { if (buf) { runs.push(...textRuns(buf, base)); buf = ""; } };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    // A bare "_x" is a subscript only after a one-character base or letter+digit (g_i, z0_i, δ_SGD); identifiers such as
    // flow_value or linear_sum_assignment stay literal. "_{...}" is always a subscript.
    const base = (text.slice(0, i).match(/[\p{L}\p{N}\p{M}]*$/u) || [""])[0].replace(/\p{M}/gu, "");
    const identifier = c === "_" && text[i + 1] !== "{" && base.length >= 2 && !/^\p{L}\p{N}$/u.test(base);
    if ((c === "_" || c === "^") && !identifier && i + 1 < text.length && text[i + 1] !== " ") {
      let body = "", j = i + 1;
      if (text[j] === "{") { let depth = 1; j++; while (j < text.length && depth) { if (text[j] === "{") depth++; else if (text[j] === "}") depth--; if (depth) body += text[j]; j++; } }
      else { while (j < text.length && /[\p{L}\p{N}]/u.test(text[j])) { body += text[j]; j++; } }
      if (!body) { buf += c; continue; }
      flush();
      runs.push(...textRuns(body.replace(/[_{}]/g, ""), { ...base, [c === "_" ? "subScript" : "superScript"]: true }));
      i = j - 1;
    } else buf += c;
  }
  flush();
  return runs;
}
function inline(text, base = {}) {
  text = text.replace(/([επ])\*/g, "$1∗");
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[@[^\]]+\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(...scriptRuns(text.slice(last, m.index), base));
    const t = m[0];
    if (t.startsWith("**")) out.push(...scriptRuns(t.slice(2, -2), { ...base, bold: true }));
    else if (t.startsWith("[@")) out.push(new TextRun({ ...base, text: cite(t.slice(1, -1).split(",").map((s) => s.trim().replace(/^@/, ""))) }));
    else out.push(...scriptRuns(t.slice(1, -1), { ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(...scriptRuns(text.slice(last), base));
  return out;
}

// ---------------------------------------------------------------- blocks
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function table(blk) {
  const cell = (txt, w, header) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: header ? { fill: "E7ECEF", type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { before: 0, after: 0, line: 240 }, children: inline(txt, { font: FONT, size: blk.size || 18, bold: header }) })],
  });
  const total = blk.widths.reduce((a, b) => a + b, 0);
  if (total > CONTENT_W + 5) throw new Error("table too wide: " + blk.caption.slice(0, 40));
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: inline(blk.caption, { font: FONT, size: 18 }) }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: blk.widths,
      rows: [new TableRow({ tableHeader: true, children: blk.header.map((h, i) => cell(h, blk.widths[i], true)) }),
             ...blk.rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, blk.widths[i], false)) }))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
function algorithm(blk) {
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 60 }, children: inline(blk.caption, { font: FONT, size: 18 }) }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W], rows: [new TableRow({ children: [new TableCell({
      borders, width: { size: CONTENT_W, type: WidthType.DXA }, shading: { fill: "F7F9FA", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 140, right: 140 },
      children: blk.lines.map((l) => new Paragraph({ spacing: { before: 0, after: 0, line: 240 }, children: textRuns(l || " ", { font: MONO, size: 17 }) })),
    })] })] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
function figure(blk) {
  const file = path.join(__dirname, "fig", blk.file);
  const { w, h } = pngSize(file);
  const width = Math.round(blk.width * 96), height = Math.round(width * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: blk.file, description: blk.caption.replace(/\*/g, "").slice(0, 200), name: blk.file } })] }),
    new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.JUSTIFIED, children: inline(blk.caption, { font: FONT, size: 18 }) }),
  ];
}

const body = [];
for (const blk of B) {
  switch (blk.type) {
    case "title": body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: blk.text, bold: true, size: 32, font: FONT })] })); break;
    case "authors": body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: blk.text, size: 24, font: FONT })] })); break;
    case "affil":
      blk.lines.forEach((l, i) => body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: i === blk.lines.length - 1 ? 240 : 0 },
        children: [new TextRun({ text: l, size: 18, italics: true, font: FONT })] })));
      break;
    case "kw": body.push(new Paragraph({ spacing: { after: 200 }, children: inline(blk.text, { font: FONT, size: 20 }) })); break;
    case "abstract":
      body.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "ABSTRACT", bold: true, size: 22, font: FONT })] }));
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 300, line: 276 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "808080", space: 8 } }, children: inline(blk.text, { font: FONT, size: 21 }) }));
      break;
    case "h1": body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: blk.text })] })); break;
    case "h2": body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: blk.text })] })); break;
    case "p": body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, children: inline(blk.text, { font: FONT, size: 22 }) })); break;
    case "list":
      blk.items.forEach((t) => body.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 80, line: 276 }, children: inline(t, { font: FONT, size: 22 }) })));
      break;
    case "eq":
      body.push(new Paragraph({ spacing: { before: 120, after: 120 },
        tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
        children: [new TextRun({ text: "\t" }), new OMath({ children: EQ[blk.omml]() }), new TextRun({ text: blk.n ? `\t(${blk.n})` : "", font: FONT, size: 22 })] }));
      break;
    case "algo": body.push(...algorithm(blk)); break;
    case "fig": body.push(...figure(blk)); break;
    case "pagebreak": body.push(new Paragraph({ children: [new PageBreak()] })); break;
    case "table": body.push(...table(blk)); break;
    case "refs":
      body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "References" })] }));
      order.forEach((k, i) => body.push(new Paragraph({ spacing: { after: 60, line: 252 }, indent: { left: 567, hanging: 567 },
        children: [new TextRun({ text: `[${i + 1}]\t${REFS[k]}`, font: FONT, size: 18 })], tabStops: [{ type: TabStopType.LEFT, position: 567 }] })));
      break;
    default: throw new Error("unknown block " + blk.type);
  }
}
const unused = Object.keys(REFS).filter((k) => !order.includes(k));
if (unused.length && B.some((b) => b.type === "refs") && !process.env.ALLOW_UNUSED) console.warn("unused references:", unused.join(", "));

const doc = new Document({
  creator: "", title: B[0].text, description: "Manuscript",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body,
  }],
});
const out = path.join(__dirname, "..", outName);
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length, "bytes;", order.length, "references cited"); });
