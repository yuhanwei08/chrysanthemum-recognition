"""Report invalid images without modifying or deleting the dataset."""
import os
from pathlib import Path
from PIL import Image


def main():
    root = Path(os.environ.get('DATA_DIR', str(Path(__file__).resolve().parent / 'data')))
    if not root.is_dir():
        raise SystemExit(f'Dataset folder not found: {root}')
    checked = 0
    invalid = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            checked += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                invalid.append(path)
                print(f'Invalid image: {path}: {error}')
    print(f'Checked {checked} images; invalid: {len(invalid)}. No files were deleted.')
    return bool(invalid)


if __name__ == '__main__':
    raise SystemExit(main())
