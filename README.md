# Warehouse Robots and Task Allocation

A Python implementation of Worksheet 1 for Social and Collective Artificial Intelligence. The project explores robot movement, limited energy and collisions in a warehouse, then compares a genetic algorithm (GA) with particle swarm optimization (PSO) for assigning delivery jobs.

Open [the executed notebook](Worksheet_1_Solution.ipynb) to read the code, saved tables, four plots and conclusions directly on GitHub. No course files or external datasets are needed to run it.

## What is included

- A 6 × 6 warehouse with four robots, eight packages and 40 time steps.
- Experiments with two/four robots and four/twelve packages, with 100 runs per condition.
- GA and PSO implementations using 12 candidates and 30 updates.
- Convergence plots, 50-seed optimizer comparisons and an exhaustive optimum benchmark.
- Checks for collisions, energy depletion, package collection and valid task assignments.

## Setup

Use **Python 3.12**, the version used for validation. In the project directory, create a virtual environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies and open the notebook:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab Worksheet_1_Solution.ipynb
```

Choose **Restart Kernel and Run All Cells**. Fixed random seeds make runs reproducible in the same environment; results can vary across library versions.

## Files and editing

| File | Purpose |
| --- | --- |
| `Worksheet_1_Solution.ipynb` | Complete notebook with saved results and explanations. |
| `worksheet_1.py` | Readable source with `# %%` cell markers for reviewing code changes on GitHub. |
| `build_notebook.py` | Builds the notebook from the Python cell source. |
| `check_notebook.py` | Checks source synchronization, valid Python and single-line statements. |
| `execute_notebook.py` | Executes every cell and saves outputs using the active Python environment. |
| `plotly_example.py` | Optional interactive chart of the original seed-42 results. |

For repository changes, edit `worksheet_1.py`, then run:

```sh
python build_notebook.py
python check_notebook.py
python execute_notebook.py
```

The builder preserves unchanged cells and clears the outputs of edited cells. Always execute the whole notebook afterwards, because an earlier code change can affect later results. If you edit the notebook directly, copy those changes into the matching cells in `worksheet_1.py` before rebuilding. The synchronization check catches mismatches.

Descriptive names such as `occupied_positions`, `best_makespan_history` and `cognitive_random_factors` identify each value's purpose. Code statements stay on one source line, while Python blocks retain their indentation. Some long statements may require horizontal scrolling.

## Modeling choices

Robots choose random adjacent cells that are unoccupied at the start of the step. Proposals are resolved simultaneously. A collision counts as one contested target cell, and robot involvements are counted separately. Each involved robot loses two energy units: one for the attempt and one collision penalty. Packages give three energy units, do not respawn, and are collected before checking for energy depletion. Fully blocked robots wait without spending energy.

For task allocation, assignments contain six robot labels in the range 0–2. Makespan is the largest assigned workload, and fitness is its negative. PSO rounds and clips particle assignments because the job allocation is discrete. The independently verified optimum is **11**. Full assumptions and limitations are documented in the notebook.

## Optional Plotly example

```sh
python -m pip install -r requirements-optional.txt
python plotly_example.py
```

Open the generated `Plotly_Example.html` in a browser. It is a separate illustration using recorded seed-42 values, not a live link to subsequent notebook edits. GitHub does not execute interactive HTML; the main notebook includes static plots for GitHub viewing.

## GitHub validation and tracked files

The GitHub Actions workflow checks the source and executes the notebook on pushes and pull requests using Python 3.12. Local validation uses the same two commands: `python check_notebook.py` and `python execute_notebook.py`.

Commit the notebook, Python files, requirements files, README, `.gitignore`, `.gitattributes` and `.github/`. Keep notebook outputs so readers can see the results without running the code. Local dependencies, virtual environments, temporary previews, generated HTML, course resources and downloaded ZIP archives are excluded by `.gitignore`. The course resources are referenced for attribution in the notebook but are not required at runtime.
