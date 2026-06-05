from pathlib import Path
import re
import shutil


INPUT_FILE = Path("Classes.yaml")
OUTPUT_FILE = Path("Classes2.yaml")
BACKUP_FILE = Path("Classes_backup.yaml")


def renumber_yaml_ids(input_file: Path, output_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    text = input_file.read_text(encoding="utf-8")

    # Backup original file
    shutil.copy2(input_file, BACKUP_FILE)

    counter = 0

    def replace_id(match: re.Match) -> str:
        nonlocal counter
        prefix = match.group(1)
        new_line = f"{prefix}{counter}"
        counter += 1
        return new_line

    # Меняет только строки, где поле называется именно id
    new_text = re.sub(
        pattern=r"^(\s*-?\s*id:\s*)\d+\s*$",
        repl=replace_id,
        string=text,
        flags=re.MULTILINE,
    )

    output_file.write_text(new_text, encoding="utf-8")

    print(f"Done. Renumbered IDs: 0–{counter - 1}")
    print(f"Original backup: {BACKUP_FILE}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    renumber_yaml_ids(INPUT_FILE, OUTPUT_FILE)