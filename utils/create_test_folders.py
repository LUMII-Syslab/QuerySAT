from pathlib import Path
import shutil

if __name__ == '__main__':
    root_path = Path("/host-dir/data/SAT_3_TEST/")
    src_path = root_path / "test"
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))

    if not src_path.is_dir():
        raise IsADirectoryError(str(src_path))

    files = [(int(f.name.split("_")[1]), f) for f in src_path.iterdir()]
    group = {v: [] for v in range(10, 500, 10)}

    for v_count, f in files:
        group[v_count].append(f)

    for v, files in group.items():
        dir = root_path / f"test_{v}_{v}"
        if not dir.exists():
            dir.mkdir()

        for f in files:
            f = f  # type: Path
            shutil.copy(f, dir / f.name)
