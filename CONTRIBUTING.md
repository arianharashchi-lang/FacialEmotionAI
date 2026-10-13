# Contributing

Contributions are welcome.

## Development setup

```powershell
git lfs install
git clone https://github.com/arianharashchi-lang/FacialEmotionAI.git
cd FacialEmotionAI
git lfs pull
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Before opening a pull request

- Keep raw datasets, captured faces, webcam adaptation media, and local databases out of Git.
- Do not commit secrets or credentials.
- Run a syntax check:

```powershell
python -m compileall -q .
```

- Update documentation when behavior or metrics change.
- Do not describe model confidence as psychological certainty.
- Clearly identify results that come from small or project-specific evaluation sets.

## Privacy

Do not contribute identifiable face media unless every depicted participant has explicitly consented to public distribution.
