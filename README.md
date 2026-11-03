# CS60075 NLP Assignment II — Group 17
**Project 6: A Text Processing Pipeline for Code-Mixed, Real-World Text**
Dataset: SemEval-2020 Task 9 SentiMix (Hinglish), **Partition B**, with the shared Dakshina lexicon.

Members: Yashvardhan Jain (`23CS10080`), Nishant (`23EE10044`), Devubarki Pushyamithra (`24CS10092`)

## Contents
```text
code/
├── data_prep.ipynb               # Builds Partition B and the Dakshina normalization dictionary
├── task1/task1_baseline.ipynb    # Task 1: language ID + Dakshina normalization + TF-IDF Random Forest
├── task2/task2_muril.ipynb       # Task 2: MuRIL fine-tuning + test-time normalization check
├── task3/task3_lid_muril.ipynb   # Task 3: MuRIL with a per-token language-tag embedding (dim 16)
└── plots.ipynb                   # Figures used in the report
results/                          # Metrics of the reported run and the two figures
requirements.txt                  # Package versions used for the reported results
report.pdf
```

## Environment
Everything runs on **Google Colab**. Use a **T4 GPU** runtime for Tasks 2 and 3; the other notebooks run on CPU. The reported results were produced with Python 3.13 and the versions in `requirements.txt`, which Colab provides preinstalled. Elsewhere: `pip install -r requirements.txt`.

## Data and checkpoint (downloaded automatically)
* SemEval-2020 Task 9: `Train.txt`, `Validation.txt`, `Test.txt` and `Ty.txt` from https://github.com/singhnivedita/SemEval2020-Task9
* Dakshina v1.0, Hindi lexicons `hi.translit.sampled.{train,dev,test}.tsv`: https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
* MuRIL checkpoint `google/muril-base-cased`: https://huggingface.co/google/muril-base-cased

## How to run
1. Upload the notebooks in `code/` to Google Drive and open each one with Google Colab.
2. Run them **in this order**, each with *Runtime → Run all*: `data_prep` → `task1_baseline` → `task2_muril` → `task3_lid_muril` → `plots`.
3. Each notebook mounts Drive and works in `MyDrive/NLP_Group17/`. Data goes to `data/` and metrics, predictions and figures go to `results/`.

To run a single task on its own, `data_prep` and `task1_baseline` must have been run once: Tasks 2 and 3 read the language-tagged, normalized splits that Task 1 writes (`data/partition_b_*_processed.json`). Approximate run times: data prep 2 min, Task 1 under 1 min, Tasks 2 and 3 about 9 min each on a T4.

**Seed:** every random step uses `SEED = 42`: the 3-way partition, the 80/10/10 split, scikit-learn and PyTorch.

## Preprocessing decisions
* **URLs:** links are split across tokens (`https // t . co / <id>`); every piece is removed, including the short-link id and truncated links.
* **Mentions:** `@` and the username token after it are removed. Usernames containing `_` are split into several tokens, so only their first piece is removed (known limitation).
* **Normalization:** tokens the Task 1 tagger does not label `Eng` are looked up in the Dakshina dictionary (each romanization → the most frequent romanization of its Devanagari word). All other tokens are unchanged.

## Deviations from the specification
* Task 1 Random Forest: `class_weight="balanced"`; each TF-IDF vocabulary capped at 25,000 features.
* Tasks 2 and 3: AdamW weight decay 0.01, 10% linear warmup, gradient clipping at 1.0; the epoch with the best validation macro-F1 is used for testing.
* Task 3 implements the **per-token tag embedding** option. The concatenation is projected back to 768 dimensions with `[I | W_lid]` (word block fixed, `W_lid` learned from zero), because a fully learned projection collapsed to predicting one class.

## Results (Partition B test split, 616 tweets)
| Configuration | Val macro-F1 | Test macro-F1 | Test accuracy |
| :--- | :---: | :---: | :---: |
| Task 1: LID + normalization + TF-IDF Random Forest | 0.5589 | 0.5641 | 0.5698 |
| Task 2: MuRIL, raw input | 0.6691 | **0.6025** | **0.6055** |
| Task 2 check: normalization at test time only | — | 0.5915 | 0.5942 |
| Task 3: MuRIL + language-tag embedding + normalization | 0.6377 | 0.5957 | 0.6006 |

Task 1's language-ID tagger: 0.8665 token accuracy, 0.8846 macro-F1. The analysis is in `report.pdf`.
