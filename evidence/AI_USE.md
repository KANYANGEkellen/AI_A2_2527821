# AI use disclosure - AI-A2-7821

**Student:** YOUR FULL NAME  |  **Registration:** 25/27821  |  **Username:** ug2527821@ines.ac.rw

## Tool used
Claude (Anthropic) through the claude.ai chat interface.

## What the AI assistant did
- Generated the first version of the project structure and code: `src/` modules, `run_all.py`, `predict.py`, `app.py`, `data/generate_data.py`, the pytest suite, the sample JSON files and the README.
- Generated `tools/make_docs.py`, which builds the report, test-log and declaration PDFs from my generated artifacts, plus first drafts of the interpretation text (regression error, classification error, cluster).
- Helped me fix set-up problems: the PowerShell execution-policy error, a pip connection error, a wrong folder name, and a CSV line-ending issue that made the SHA-256 fingerprint differ between Windows and Linux (fixed with `lineterminator="\n"`).

## What I did myself
(Edit this list so it is true for you.)
- Created the virtual environment, installed the packages and ran `run_all.py`, `predict.py` and `pytest -q` on my own computer with my own seed (7821).
- Checked that the verification code, row count and generated artifacts matched my registration digits.
- Read the code and the generated JSON/CSV files: [say which files you studied].
- Rewrote the interpretation of one regression error, one classification error and one farm cluster in my own words: [yes / no].
- Recorded the demonstration video and wrote the declaration: [yes / no].

## Verification
- I can explain the data generator, the train/test split, the preprocessing pipeline, why the models were selected by cross-validation, and how the API validates input.
- I can make a small change live (for example change a range rule or add a response field).

## Honest limits
The code was AI-generated and I adapted it as follows: [describe any change you made].
