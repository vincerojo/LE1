# DLSU SOLST01 AY2025–2026 Term 3

Course repository for **SOLST01: Solid State Physics** at De La Salle University, AY2025–2026 Term 3.

This repository contains course templates, computational notebooks, examples, and supporting resources.

## Repository contents

```text
dlsu-solst01-ay20252026t3/
├── README.md
├── pyproject.toml
├── latex-submission-template/
│   ├── README.md
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   │   └── example_figure.png
│   └── main.pdf
├── notebooks/
├── resources/
└── tests/
```

## LaTeX submission template

Students should use the LaTeX submission template in:

```text
latex-submission-template/
```

The main file is:

```text
latex-submission-template/main.tex
```

Students should edit `main.tex`, compile it to PDF, and submit the required files.

## Required submission files

For LaTeX-based submissions, students must submit:

1. the compiled PDF file
2. the `.tex` source file
3. all figure files used in the document
4. the `.bib` file, if BibTeX was used

The submitted PDF must compile from the submitted source files.

## Figures

Place figures inside:

```text
figures/
```

In LaTeX, include figures using a relative path:

```latex
\includegraphics[width=0.7\textwidth]{figures/example_figure.png}
```

Every figure must have a caption and must be discussed in the main text.

## Equations

All important equations must be explained in words.

Students should define all symbols before using them repeatedly. For example,

```latex
\begin{equation}
\hat{H}\psi(x) = E\psi(x),
\end{equation}
```

where $\hat{H}$ is the Hamiltonian operator, $\psi(x)$ is the wavefunction, and $E$ is the energy eigenvalue.

## Computational notebooks

Course notebooks are stored in:

```text
notebooks/
```

These notebooks may include examples for:

* free-electron models
* finite-difference Hamiltonians
* periodic boundary conditions
* Bloch states
* band structures
* density of states
* tight-binding models

## Python environment

This repository includes a `pyproject.toml` file for setting up the Python environment.

Create a virtual environment:

```bash
python -m venv venv_solst01
```

Activate it on macOS or Linux:

```bash
source venv_solst01/bin/activate
```

Activate it on Windows PowerShell:

```powershell
venv_solst01\Scripts\Activate.ps1
```

Install the package and development tools:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Running notebooks

After installing the environment, start Jupyter:

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

Then open notebooks from the `notebooks/` folder.

## Compilation of LaTeX files

To compile the LaTeX template:

```bash
cd latex-submission-template
pdflatex main.tex
```

If using BibTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission rule

Students are responsible for checking that their PDF compiles correctly.

A submission is incomplete if:

* the PDF is missing
* the `.tex` source file is missing
* required figures are missing
* the `.bib` file is missing when BibTeX is used
* the submitted source files do not reproduce the submitted PDF

## License
Course notes, templates, and written materials are released under the Creative Commons Attribution 4.0 International License unless otherwise stated.

Code examples are released under the MIT License unless otherwise stated.
