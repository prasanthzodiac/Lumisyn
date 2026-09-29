# Originality and Writing-Style Check: Revision 4

**Date:** 2026-09-27.

**Files checked:**
- `Lumisyn_DP_FlowMatching_Manuscript.docx`: 9,535 words of body text, excluding references. The abstract is 249 words, within the IEEE limit of 250.
- `Lumisyn_Supplementary_Material.docx`: 4,060 words.

**How to reproduce:** `python paper/check_originality.py <file.docx>`.

## What this is, and what it is not
This is a self-check. It is **not** a Turnitin or iThenticate report. Those services compare against paywalled journal databases that are not accessible here. IEEE and most Wiley and Elsevier journals run iThenticate (Crossref Similarity Check) on submission, so run it yourself first if your institution has access.

## 1. Overlap with the closest published texts
The manuscript was compared word-for-word (lower-cased, punctuation stripped) against the full text of seven papers:
- the two style references;
- the five closest technical papers, whose full text was downloaded from arXiv for this check.

| Reference | 5-word sequences | 8-word | 12-word |
|---|---|---|---|
| Ortuño et al. 2026, *Adv. Sci.* (style reference) | 8 (0.08%) | **0** | **0** |
| Zhang et al. 2024, TabSyn (style reference) | 1 (0.01%) | **0** | **0** |
| Che et al. 2026, StraightDP (arXiv 2607.29100) | 5 (0.05%) | **0** | **0** |
| Wang et al. 2026, DP-SGD corrected analysis (arXiv 2605.15648) | 10 (0.10%) | **0** | **0** |
| Tong et al., OT-CFM (arXiv 2302.00482) | 1 (0.01%) | **0** | **0** |
| McKenna et al., AIM (arXiv 2201.12677) | 5 (0.05%) | **0** | **0** |
| Tao et al., DP synthesis benchmark (arXiv 2112.09238) | 0 | **0** | **0** |

The supplement also has no shared 8-word sequence with any of the seven papers.

The 5-word matches are generic phrases, such as "trained on synthetic data" or "differentially private stochastic gradient descent". The single 8-word match found in the first pass was the textbook phrase "is differentially private stochastic gradient descent (DP-SGD)", which was reworded. Each of these papers is cited wherever its ideas are used. In particular, StraightDP is credited in Sections 2.6 and 4 for the two-stage "release moments, then DP-SGD" pattern that the warm start follows.

## 2. Web exact-phrase check
We searched distinctive new sentences from revision 4 verbatim:
- "warm-started flow learns dependence that the marginals cannot carry"
- "whatever a cheap one-shot measurement can carry should not be relearned through noisy gradients"
- "runs and accounts every DP-SGD step with a learning rate of zero"

None returned a matching source. Revision 3 sentences had already been checked the same way.

The search did surface one naming clash. Warm-start flow matching (Kim 2026, arXiv 2603.19360) uses "warm start" for a different idea, so the manuscript now cites it and states that the term refers to network weights.

## 3. AI-writing-pattern scan
- **Word list:** 0 hits in either file. The list has 40 items, including delve, crucial, pivotal, leverage, seamless, furthermore, moreover, additionally, notably, comprehensive, robust, novel, utilize, "it is worth noting" and "in conclusion".
- **Em-dashes:** 0. En-dashes are used only in ranges and in names such as Runge–Kutta.
- **Sentence rhythm (main text):**
  - 333 sentences, mean 26.7 words, SD 23.8;
  - 50 short sentences (under 10 words) and 65 long ones (over 35 words).

  Tables and lists inflate the SD. The mix of short and long sentences is typical of human academic prose.
- **Structure:** the paper follows the IMRaD pattern of the Advanced Science reference:
  - 1 Introduction;
  - 2 Methods (2.1–2.10);
  - 3 Results (3.1–3.8);
  - 4 Discussion, with 4.1 Threats to Validity;
  - 5 Conclusions;
  - Data and Code Availability, Supplementary Material, and the declarations.

  The aims are stated inline as (i)–(iv), the contributions TabSyn-style as a bulleted list with numbers, and metrics as "**term:** definition".

## 4. Build and format checks
- All three DOCX files pass the Office Open XML schema validator.
- A PDF preview exported through Microsoft Word (`Lumisyn_DP_FlowMatching_Manuscript_preview.pdf`, 19 pages) was inspected visually on the pages carrying the new material: the abstract, Section 2.6, Section 3.7, Table 8, Figure 8, and Supplementary S1.3 and S4. The rest of the changed text was proofread from the extracted text.
- Every number in the new Section 3.7, Table 8 and Supplementary Tables S1b and S4e–f is read directly from `runs/flow_value/analysis.json` when the document is built. None was typed by hand.
- Fixed in this revision: code identifiers containing underscores (for example `linear_sum_assignment` and `runs/flow_value/…`) were being rendered as subscripts. Only mathematical subscripts are now subscripted.

## 5. Remaining responsibility of the authors
- Author names, affiliations, correspondence, acknowledgements, contributions, conflicts of interest and the repository URL and DOI are left as `[placeholders]`.
- Add an AI-assistance disclosure if the target journal requires one. IEEE requires disclosure in the acknowledgements.
- Run iThenticate or Turnitin before submission.
- Three references are 2026 arXiv preprints: StraightDP, Wang et al. and Kim (WS-FM). They were still preprints on 2026-09-27; update them if they are published before submission.

## 6. Budget-allocation manuscript (`Lumisyn_AdaptiveBudget_FlowMatching_Manuscript.docx`)

- **Built from:** the same study, via `docx_build/content_adaptive.js`. That script rewrites the title, abstract, introduction, Section 2.5, the new Section 3.8, the opening of the Discussion and the Conclusions. All methods, results, tables and figures are shared with the main manuscript.
- **Length:** 10,012 body words, and a 243-word abstract (IEEE limit 250).
- **Overlap:** 0 shared 8-word or 12-word sequences with any of the seven reference texts. The 5-word matches range from 0.00% to 0.09%.
- **Style:** 0 hits from the AI-style word list, and 0 em-dashes.
- **Validation:** passes the Office Open XML schema validator. The preview PDF was checked on pages 1, 6 and 16.
- **Honesty of the framing:**
  - The title names the topic, adaptive budget allocation.
  - The abstract and results state that DAPC adapts the *total* budget and is stable, not that it is optimal for utility.
  - They state that the pre-registered ε-adaptive *division* did not beat a fixed division.
