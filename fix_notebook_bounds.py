import nbformat
from pathlib import Path

path = Path('implementation.ipynb')
nb = nbformat.read(path, as_version=4)

replacements = {
    "2 * tournament_8.n_teams - 2": "tournament_8.n_teams - 2",
    "lower_bound = 2 * n_teams - 2": "lower_bound = n_teams - 2",
    "2 * n_teams - 2": "n_teams - 2",
    "2*n_teams - 2": "n_teams - 2",
    "2 * 4 - 2": "4 - 2",
    "2 * 6 - 2": "6 - 2",
    "2 * 8 - 2": "8 - 2",
    "2 * 10 - 2": "10 - 2",
    "2 * 12 - 2": "12 - 2",
    "2 * 14 - 2": "14 - 2",
    "2 * 16 - 2": "16 - 2",
    "2 * 18 - 2": "18 - 2",
    "2 * 20 - 2": "20 - 2",
    "2n - 2": "n - 2",
    "2n-2": "n-2",
    "$2n - 2$": "$n - 2$",
    "$2n-2$": "$n-2$",
}

changed = 0
for cell in nb.cells:
    src = cell.get('source', '')
    if isinstance(src, str):
        new_src = src
        for old, new in replacements.items():
            new_src = new_src.replace(old, new)
        if new_src != src:
            cell['source'] = new_src
            changed += 1
    elif isinstance(src, list):
        new_lines = []
        modified = False
        for line in src:
            new_line = line
            for old, new in replacements.items():
                if old in new_line:
                    new_line = new_line.replace(old, new)
            if new_line != line:
                modified = True
            new_lines.append(new_line)
        if modified:
            cell['source'] = new_lines
            changed += 1

nbformat.write(nb, path)
print(f"Celdas modificadas: {changed}")
