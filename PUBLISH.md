# Publishing on GitHub and archiving on Zenodo

```bash
cd materiverse
git init -b main            # already done if you downloaded the prepared repository
git add -A && git commit -m "Initial release of the MateriVerse evaluation toolkit"
git remote add origin https://github.com/<your-account>/materiverse.git
git push -u origin main
git tag v0.1.0 && git push origin v0.1.0
```

Zenodo DOI: sign in to Zenodo with GitHub, enable the repository under *GitHub*, then create a GitHub release for the
tag. Zenodo archives it and issues a DOI. Add the model weights, candidate CIF files and run logs to the Zenodo
record, or as a second record, because the toolkit itself does not contain them.

Update the article's Data availability statement only when the repository and archive exist, for example:

> The code, the run logs and the generated candidate structures are available in the GitHub repository
> https://github.com/<your-account>/materiverse and archived at Zenodo (https://doi.org/<zenodo-doi>).
> The training data are available from the Materials Project, OQMD and NOMAD.
