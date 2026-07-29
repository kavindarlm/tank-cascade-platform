# Cascade GNN

Cascade GNN is a Python project for running a graph neural network workflow over the project notebooks. The workflow is executed by the runner script in the project root, which runs Jupyter notebooks sequentially and saves outputs in place.

## Project structure

- `notebooks/` - Jupyter notebooks for each workflow phase
- `data/` - Input data files used by the notebooks
- `models/` - Pickle files and trained model artifacts
- `outputs/` - Generated CSV files, plots, and other workflow outputs
- `run.py` - Runner script that executes the notebooks
- `requirements.txt` - Python dependencies for the project

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the workflow

Run the full notebook workflow from the project root:

```bash
python run.py
```

This runs the default notebook sequence:

- `01_phase1_graph.ipynb`
- `02_phase2_gnn.ipynb`
- `03_phase3_outputs.ipynb`
- `04_phase4_runoff_coefficient.ipynb`

## Runner script usage

The runner script accepts the following options:

### Run a specific notebook

```bash
python run.py --notebooks 01_phase1_graph.ipynb
```

### Run a subset of notebooks

```bash
python run.py --notebooks 01_phase1_graph.ipynb 02_phase2_gnn.ipynb
```

### Run all notebooks explicitly

```bash
python run.py --notebooks all
```

### Increase the per-notebook timeout

```bash
python run.py --timeout 7200
```

## Notes

- The runner executes each notebook with `jupyter nbconvert`, so the notebooks are run in place and any generated outputs are written back to the notebook and the project directories.
- The notebooks expect the project root to be the current working directory.
- Output files are written to `outputs/`, while trained artifacts and pickles are stored in `models/`.
