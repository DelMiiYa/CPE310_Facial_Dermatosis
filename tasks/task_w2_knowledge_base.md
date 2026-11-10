# Goldfish Task Spec: W2 — Bilingual Static Condition Knowledge Base (`EN` + `TH`)

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier (Acne, Rosacea, Dermatitis, Normal) for a university Healthcare AI System course in Thailand.
The professor requires the demo app to show a detailed educational card about the predicted condition, its typical symptoms, facial distribution, triggers, differential diagnosis, self-care, and red flags.
Static text in `webapp/knowledge_base.json` is a deliberate engineering choice: it is clinically reviewable, reproducible in the live demo, works offline, and cannot hallucinate drug names or medical claims.
Task W2 creates `webapp/knowledge_base.json` in both **English (`en`)** and **Thai (`th`)**, paraphrased from reputable dermatology sources with source URLs.

## 2. Goal
Create `webapp/knowledge_base.json` with comprehensive, paraphrased English and Thai educational entries for `"acne"`, `"rosacea"`, `"dermatitis"`, and `"normal"`, strictly matching the frozen schema in `tasks/WEBAPP_MASTER_PLAN.md`.

## 3. Files to Read First
Read these exact files before writing `webapp/knowledge_base.json`:
- `tasks/WEBAPP_MASTER_PLAN.md` (specifically Section 4.4: Frozen `webapp/knowledge_base.json` Entry Schema)
- `Project_handoff_summary.md` (Section 1 & Section 3 Step 12–13 for how our model localizes each condition via Grad-CAM and why Atopic Eczema + Contact Dermatitis are merged into `Dermatitis`)

### Copied Frozen Schema from `tasks/WEBAPP_MASTER_PLAN.md`
Top-level JSON object MUST have exactly 4 lowercase keys: `"acne"`, `"rosacea"`, `"dermatitis"`, `"normal"`.
Each of the 4 entries MUST have these exact top-level keys (in English, satisfying direct flat key lookups) PLUS `"en"` and `"th"` sub-objects with the exact same keys for the UI language toggle:
```json
{
  "acne": {
    "name": "Acne (Acne Vulgaris)",
    "model_match_phrase": "The model's best match is Acne (Acne Vulgaris).",
    "short_description": "...",
    "typical_symptoms": ["...", "..."],
    "typical_facial_distribution": "...",
    "common_triggers_or_factors": ["...", "..."],
    "commonly_confused_with": [
      {"condition": "Rosacea", "how_it_differs": "..."},
      {"condition": "Perioral or Contact Dermatitis", "how_it_differs": "..."},
      {"condition": "Folliculitis", "how_it_differs": "..."}
    ],
    "general_care_tips": ["...", "..."],
    "see_a_doctor_if": ["...", "..."],
    "caution_note": null,
    "en": {
      "name": "Acne (Acne Vulgaris)",
      "model_match_phrase": "The model's best match is Acne (Acne Vulgaris).",
      "short_description": "...",
      "typical_symptoms": ["...", "..."],
      "typical_facial_distribution": "...",
      "common_triggers_or_factors": ["...", "..."],
      "commonly_confused_with": [
        {"condition": "Rosacea", "how_it_differs": "..."},
        {"condition": "Perioral or Contact Dermatitis", "how_it_differs": "..."},
        {"condition": "Folliculitis", "how_it_differs": "..."}
      ],
      "general_care_tips": ["...", "..."],
      "see_a_doctor_if": ["...", "..."],
      "caution_note": null
    },
    "th": {
      "name": "สิว (Acne Vulgaris)",
      "model_match_phrase": "ผลลัพธ์ที่ใกล้เคียงที่สุดจากโมเดลคือ สิว (Acne Vulgaris)",
      "short_description": "...",
      "typical_symptoms": ["...", "..."],
      "typical_facial_distribution": "...",
      "common_triggers_or_factors": ["...", "..."],
      "commonly_confused_with": [
        {"condition": "โรคโรซาเซีย (Rosacea)", "how_it_differs": "..."},
        {"condition": "โรคผิวหนังอักเสบ (Dermatitis)", "how_it_differs": "..."},
        {"condition": "รูขุมขนอักเสบ (Folliculitis)", "how_it_differs": "..."}
      ],
      "general_care_tips": ["...", "..."],
      "see_a_doctor_if": ["...", "..."],
      "caution_note": null
    },
    "sources": [
      {"title": "American Academy of Dermatology (AAD) — Acne: Overview, Signs & Symptoms", "url": "https://www.aad.org/public/diseases/acne/really-acne/overview"},
      {"title": "NHS — Acne", "url": "https://www.nhs.uk/conditions/acne/"},
      {"title": "Mayo Clinic — Acne: Symptoms and Causes", "url": "https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047"},
      {"title": "DermNet NZ — Acne Vulgaris", "url": "https://dermnetnz.org/topics/acne-vulgaris"}
    ]
  }
}
```

## 4. Files to Create or Edit
Create ONLY this file (everything else is strictly off limits):
1. `webapp/knowledge_base.json`

## 5. Step-by-Step Instructions

1. **Create `webapp/knowledge_base.json` encoded strictly in UTF-8** (so Thai script renders without mojibake).
2. **Populate all 4 class entries (`"acne"`, `"rosacea"`, `"dermatitis"`, `"normal"`)** following these mandatory clinical & project content rules:
   - **Never write "You have X" or "คุณเป็นโรค X".** Always use phrasing such as `"The model's best match is ..."` / `"ผลลัพธ์ที่ใกล้เคียงที่สุดจากโมเดลคือ ..."`.
   - **No drug names or dosages** in `general_care_tips` (keep care tips conservative and non-prescriptive: gentle cleanser, lukewarm water, non-comedogenic moisturizer/sunscreen, avoiding picking/squeezing, identifying and avoiding personal triggers, patch-testing cosmetics).
   - **Paraphrase in your own words** from reputable sources (American Academy of Dermatology, National Rosacea Society, NHS, Mayo Clinic, National Eczema Association, DermNet NZ). Include 3–4 verified reference objects `{"title": "...", "url": "..."}` in `sources` for each class.
   - **Link `typical_facial_distribution` to what the model's 5-region Grad-CAM shows:**
     - **Acne (`acne`):** Oil-rich sebaceous regions — forehead, nose, chin (the T-zone), and lateral cheeks (in our test set, Grad-CAM attribution is led by chin `28.9%`, right cheek `20.6%`, left cheek `19.6%`, and forehead `16.5%`).
     - **Rosacea (`rosacea`):** Central face (centrofacial / malar distribution) — convexities of the central cheeks, nose, mid-forehead, and chin, often sparing the immediate periocular skin (in our test set, Grad-CAM attribution concentrates `68.1%` on nose, left cheek, and right cheek).
     - **Dermatitis (`dermatitis`):** Eyelids, perioral/chin area, cheeks, or hairline/flexural facial zones depending on allergen/irritant contact or atopic barrier dysfunction (in our test set, Grad-CAM attribution is highest on the chin/perioral region `42.9%` and cheeks).
     - **Normal (`normal`):** Uniform, balanced skin texture and tone across all 5 facial regions (forehead, nose, left cheek, right cheek, chin) without focal inflammatory heatmaps.
   - **Specific mandatory notes for `dermatitis` and `normal`:**
     - In `"dermatitis"`, explicitly explain in `short_description` and `caution_note` (both EN and TH) that **in this classifier, `Dermatitis` merges Atopic Eczema (Atopic Dermatitis) and Contact Dermatitis (irritant/allergic)** because they share spongiotic erythema, scaling, and pruritus that cannot be separated from a single photograph without clinical history/patch testing. Also state in `caution_note` that Dermatitis is the smallest class in the training dataset (93 training / 15 test images; in-domain F1 0.62–0.71), so predictions carry higher uncertainty.
     - In `"normal"`, explicitly explain in `short_description` and `caution_note` (both EN and TH) that **a `Normal` result means the model did not detect strong visual patterns of the 3 target conditions (Acne, Rosacea, Dermatitis) — it does NOT guarantee healthy skin or rule out other dermatological conditions** (such as pigmentary disorders, infections, or skin neoplasms) outside the 4-class scope.
3. **Ensure completeness:**
   - Every list field (`typical_symptoms`, `common_triggers_or_factors`, `general_care_tips`, `see_a_doctor_if`) must have **4 to 6 clear, informative bullet items** in both `en` and `th`.
   - `commonly_confused_with` must have **2 to 3 items**, each an object `{"condition": "...", "how_it_differs": "..."}` in both `en` and `th` (for example, explaining that Rosacea lacks blackheads/whiteheads [comedones] and features persistent centrofacial flushing/telangiectasia, whereas Acne features open/closed comedones and follicular papules/pustules; and Dermatitis features intense itching [pruritus], dry scaling/crusting, or indistinct eczematous plaques).
   - Ensure the top-level fields (`name`, `model_match_phrase`, `short_description`, `typical_symptoms`, `typical_facial_distribution`, `common_triggers_or_factors`, `commonly_confused_with`, `general_care_tips`, `see_a_doctor_if`, `caution_note`) match the `"en"` sub-object exactly.

## 6. Hard Rules
- **DO NOT** touch `data/raw/`, `checkpoints/`, `.env`, or `src/`.
- **UTF-8 encoding is mandatory** (Thai characters must be stored as real UTF-8 text, not `\u0e...` escapes if possible, and must parse cleanly with `json.loads(Path("webapp/knowledge_base.json").read_text(encoding="utf-8"))`).
- **No specific drug names or prescription dosages** anywhere in `knowledge_base.json`.
- **Never use diagnostic phrasing** ("You have...").

## 7. Acceptance Criteria
Run this command from `e:\CPE310FacialDermatorsisProject` and paste its exact output into your handoff report:
```powershell
python -c "import json; from pathlib import Path; kb = json.loads(Path('webapp/knowledge_base.json').read_text(encoding='utf-8')); req = ['name', 'model_match_phrase', 'short_description', 'typical_symptoms', 'typical_facial_distribution', 'common_triggers_or_factors', 'commonly_confused_with', 'general_care_tips', 'see_a_doctor_if', 'caution_note']; assert set(kb.keys()) == {'acne', 'rosacea', 'dermatitis', 'normal'}, f'Bad keys: {kb.keys()}'; [ (assert_top := all(k in kb[c] for k in req + ['en', 'th', 'sources']), assert_en := all(k in kb[c]['en'] for k in req), assert_th := all(k in kb[c]['th'] for k in req), print(c, '-> EN:', kb[c]['en']['name'], '| TH:', kb[c]['th']['name'], '| sources:', len(kb[c]['sources']))) for c in ['acne', 'rosacea', 'dermatitis', 'normal'] ]; print('ALL SCHEMA CHECKS PASSED')"
```
Expected: Prints the 4 classes with their English and Thai names, source counts (`>= 3` each), and `ALL SCHEMA CHECKS PASSED`.

## 8. Stop Conditions
- If `tasks/WEBAPP_MASTER_PLAN.md` is missing or its schema in Section 4.4 cannot be read, **STOP** and report that in `tasks/task_w2_handoff_report.md`.

## 9. Report Instructions
When finished, write `tasks/task_w2_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact commands run and their actual output (pasted verbatim)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check (and remind Del to review the Thai medical phrasing before recording the final demo clip)
- **Numbers to carry forward:** number of sources per condition and total unique reference URLs available for the assignment write-up.

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session 2)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w2_knowledge_base.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** The acceptance command printed `ALL SCHEMA CHECKS PASSED` and Thai names rendered cleanly without garbled characters.
