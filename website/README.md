# Editing the JoJo documentation

The website lives beside the Python package in this repository. Edit the Markdown pages in `website/docs/`, then preview them locally:

```bash
python -m pip install -r website/requirements.txt
python -m mkdocs serve
```

Open the local address printed by MkDocs. Run `python -m mkdocs build --strict` to check links and page configuration. The GitHub Actions workflow builds the site from this directory and publishes it with GitHub Pages when changes are pushed to `master`. On GitHub, set **Settings → Pages → Build and deployment → Source** to **GitHub Actions** once.
