"""Validate notebook structure, source synchronization and single-line statements."""
import ast
import io
import tokenize
import nbformat
from build_notebook import NOTEBOOK_PATH, read_source_cells


def check_notebook():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    source_cells = read_source_cells()
    assert len(notebook.cells) == len(source_cells), 'Notebook and source have different cell counts.'
    for cell_index, (notebook_cell, (source_type, source_text)) in enumerate(zip(notebook.cells, source_cells), start=1):
        assert notebook_cell.cell_type == source_type, f'Cell {cell_index}: cell types differ.'
        assert notebook_cell.source.strip() == source_text, f'Cell {cell_index}: source differs; synchronize the source and notebook.'
        if source_type != 'code':
            continue
        ast.parse(source_text)
        bracket_depth = 0
        for source_token in tokenize.generate_tokens(io.StringIO(source_text).readline):
            if source_token.type == tokenize.OP:
                if source_token.string in '([{':
                    bracket_depth += 1
                elif source_token.string in ')]}':
                    bracket_depth -= 1
            assert not (source_token.type == tokenize.NL and bracket_depth), f'Cell {cell_index}: a statement is split across lines.'
    print(f'Validated {len(notebook.cells)} cells: source matches and code statements stay on single lines.')


if __name__ == '__main__':
    check_notebook()
