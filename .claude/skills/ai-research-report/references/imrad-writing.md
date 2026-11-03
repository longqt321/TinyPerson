# IMRaD writing guide for AI/ML

## Structure and order

Write: Method → Experiments → Related Work → Introduction → Limitations → Conclusion → Abstract. Results and discussion can share Experiments. Keep Introduction claims aligned with Method and measured Experiments.

| Section | Typical shape | Content |
|---|---|---|
| Abstract | 150–250 words, 5–7 sentences | Context, gap, method, concrete result, significance/next step; no citations or table/figure refs |
| Introduction | 4–6 paragraphs, 1–1.5 pages | Context, specific problem, gap, proposal, 3–4 evidence-backed contributions |
| Related Work | 3–5 thematic paragraphs | Organize by theme, contrast this work with prior work |
| Method | 2–4 pages | Problem formulation, architecture, training, optional theory |
| Experiments | 3–5 pages | Setup, main results, ablations, analysis/failures |
| Limitations | 1–2 paragraphs | Dataset/scope, assumptions, failure conditions, relevant social/ethical risks |
| Conclusion | 1 paragraph | Contributions and supported findings; no new results |

Venue limits override these defaults. Verify official template and instructions.

## Section guidance

### Introduction

Use a funnel: broad motivation; concrete problem and prior limitations; precise gap; proposed idea; numbered contributions. Each contribution must appear in Method and have evidence in Experiments. Avoid dictionary openings and vague “we study an important problem” claims.

### Related Work

Group by research theme, not chronology. Each paragraph synthesizes representative work, identifies distinctions, and positions current study. Avoid a sequence of one-paper summaries.

### Method

Define inputs, outputs, objective, and symbols. Explain architecture in prose before equations; place main architecture figure here. Define every symbol on first use. Describe loss, optimizer, schedule, batch size, steps, framework, and other details needed for reproduction.

### Experiments

- **Setup:** dataset source/size/split, metric definition, baseline provenance, implementation, hardware, runtime, seed count.
- **Main results:** explain key patterns without narrating every table cell.
- **Ablation:** isolate components; mark unavailable ablations explicitly.
- **Analysis:** include cost, stratification, qualitative examples, and failure cases only when evidence exists.

Use past tense for completed procedures (“We trained…”), present tense for displayed results (“Table 2 shows…”). State test and p-value when claiming statistical significance.

### Limitations and conclusion

Name actual limits. Conclusion summarizes supported contributions and results, then future work. Introduce no new result.

## Style

Prefer concise active sentences, consistent terminology, and defined acronyms. Keep most sentences under 30 words. Hedge weak evidence (“may”, “suggests”). Avoid “groundbreaking”, “unprecedented”, and “state-of-the-art” without direct, current comparison to strongest method on same benchmark. Never use “significantly” without statistical test.

For each outline section, record paragraph/sentence targets, claims linked to `03_analysis.md`, cited sources, and figure/table placement.
