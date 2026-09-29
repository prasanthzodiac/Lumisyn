// Main manuscript followed by the Supplementary Material in one file (for single-file review).
const main = require(process.env.MAIN_CONTENT || "./content.js");
const supp = require("./content_supp.js");
const suppBody = supp.B.filter((b) => !["title", "authors", "affil"].includes(b.type));
const B = [...main.B, { type: "pagebreak" }, { type: "title", text: "Supplementary Material" }, ...suppBody];
module.exports = { B, REFS: main.REFS };
