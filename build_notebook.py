"""Build the notebook from the readable worksheet_1.py cell source."""
from pathlib import Path
import nbformat

PROJECT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIRECTORY / 'worksheet_1.py'
NOTEBOOK_PATH = PROJECT_DIRECTORY / 'Worksheet_1_Solution.ipynb'


def read_source_cells():
    """Read # %% code cells and # %% [markdown] explanation cells."""
    source_cells = []
    cell_type = None
    cell_lines = []
    for source_line in SOURCE_PATH.read_text(encoding='utf-8').splitlines():
        if source_line in ('# %%', '# %% [markdown]'):
            if cell_type is not None:
                source_cells.append((cell_type, '\n'.join(cell_lines).strip()))
            cell_type = 'markdown' if '[markdown]' in source_line else 'code'
            cell_lines = []
        elif cell_type == 'markdown':
            cell_lines.append(source_line[2:] if source_line.startswith('# ') else source_line.removeprefix('#'))
        elif cell_type == 'code':
            cell_lines.append(source_line)
    if cell_type is not None:
        source_cells.append((cell_type, '\n'.join(cell_lines).strip()))
    return source_cells


def build_notebook():
    """Preserve outputs for unchanged cells; clear outputs for edited cells."""
    previous_notebook = nbformat.read(NOTEBOOK_PATH, as_version=4) if NOTEBOOK_PATH.exists() else None
    previous_cells = previous_notebook.cells if previous_notebook else []
    notebook_cells = []
    for cell_index, (cell_type, source_text) in enumerate(read_source_cells()):
        previous_cell = previous_cells[cell_index] if cell_index < len(previous_cells) else None
        if previous_cell and previous_cell.cell_type == cell_type and previous_cell.source.strip() == source_text:
            notebook_cells.append(previous_cell)
        elif cell_type == 'markdown':
            notebook_cells.append(nbformat.v4.new_markdown_cell(source_text))
        else:
            notebook_cells.append(nbformat.v4.new_code_cell(source_text))
    metadata = previous_notebook.metadata if previous_notebook else {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
    notebook = nbformat.v4.new_notebook(cells=notebook_cells, metadata=metadata)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f'Built {NOTEBOOK_PATH.name} with {len(notebook_cells)} cells.')


if __name__ == '__main__':
    build_notebook()
