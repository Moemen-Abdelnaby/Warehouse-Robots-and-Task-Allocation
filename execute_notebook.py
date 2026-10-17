"""Run every notebook cell with the active Python interpreter and validate outputs."""
import base64
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

PROJECT_DIRECTORY = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIRECTORY / 'Worksheet_1_Solution.ipynb'
PREVIEW_DIRECTORY = PROJECT_DIRECTORY / 'tmp' / 'plot_previews'


class CurrentPythonKernelManager(KernelManager):
    """Use this interpreter instead of a possibly unrelated installed kernel."""
    @property
    def kernel_spec(self):
        return KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='Python 3', language='python')


def execute_notebook():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    kernel_manager = CurrentPythonKernelManager()
    notebook_client = NotebookClient(notebook, km=kernel_manager, timeout=240, resources={'metadata': {'path': str(PROJECT_DIRECTORY)}})
    notebook_client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plot_count = 0
    for notebook_cell in notebook.cells:
        for cell_output in notebook_cell.get('outputs', []):
            assert cell_output.output_type != 'error'
            if cell_output.output_type == 'stream':
                print(cell_output.text)
            output_data = cell_output.get('data', {})
            if 'text/markdown' in output_data:
                print(output_data['text/markdown'])
            if 'image/png' in output_data:
                plot_count += 1
                (PREVIEW_DIRECTORY / f'plot_{plot_count}.png').write_bytes(base64.b64decode(output_data['image/png']))
    print(f'Executed and validated {len(notebook.cells)} cells; {plot_count} plots.')


if __name__ == '__main__':
    execute_notebook()
