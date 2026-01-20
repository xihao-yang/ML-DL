# NLP_exercise

This directory contains materials for an NLP exercise based on the **UD English GUM corpus**, focusing on
**POS tagging performance comparison and analysis**.

The exercise includes:
- data preparation
- experiment scripts
- result figures for analysis and reporting

---

## Directory Structure

```text
NLP_exercise/
├── Data/
│   └── en_gum-ud-dev.conllu
├── Fig/
│   └── (generated figures, e.g., accuracy trends, confusion matrices)
└── experiment_gum_spacy_stanza_trend.py
```
## Contents

### **experiment_gum_spacy_stanza_trend.py**

Main experiment script used to:

* load the UD GUM development set

* run POS tagging with spaCy and Stanza

* compute evaluation metrics (accuracy, F1, etc.)

* generate trend plots and confusion matrices

This script is intended to be run locally and can be adapted for further experiments.

### **Data/en_gum-ud-dev.conllu**

The development split of the Universal Dependencies English GUM corpus.

* Format: CoNLL-U

* Size: ~2.8 MB

* Usage: POS tagging evaluation and error analysis

Note:
This data is included for academic and educational purposes only.
Please refer to the official Universal Dependencies project for licensing details.

### **Fig/**

Contains figures generated from the experiment, such as:

* POS accuracy trends

* speed comparison

* token-level F1 scores

* confusion matrices

These figures are mainly used for:

* result interpretation

* reports

* course presentations

**How to Run**
Example (from repository root):

```Bash
python third_party/NLP_exercise/experiment_gum_spacy_stanza_trend.py
```
Make sure the required NLP libraries (e.g., spaCy, Stanza) are installed beforehand.

Notes
This directory is placed under ```third_party/``` to indicate that it is an exercise / external dataset–based experiment, not core library code.

Draft files and report frameworks are intentionally excluded from version control.

The focus is on reproducibility and clarity, rather than production deployment.

Reference
Universal Dependencies: https://universaldependencies.org/

GUM Corpus: https://github.com/UniversalDependencies/UD_English-GUM
