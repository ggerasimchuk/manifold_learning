#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path

SOURCE_DIR = Path("/raw/records")

# --- Утилиты ---------------------------------------------------------------

def trim_trailing_empty(cells):
    """Убирает пустую «хвостовую» ячейку, которая появляется из-за завершающей запятой."""
    out = list(cells)
    while out and (out[-1] is None or str(out[-1]).strip() == ""):
        out.pop()
    return out

def slugify(value: str, allow_unicode=False) -> str:
    """Безопасное имя файла: пробелы -> '_', лишнее убираем, подчёркивания схлопываем."""
    value = str(value)
    if not allow_unicode:
        # Оставляем как есть (поддержка кириллицы macOS/Windows ok),
        # но всё равно чистим запрещённые символы.
        pass
    # Заменяем запрещённые символы
    value = re.sub(r'[\\/:"*?<>|]+', "_", value)
    # Пробелы и запятые -> _
    value = re.sub(r"[\s,]+", "_", value)
    # Схлопываем повторные _
    value = re.sub(r"_+", "_", value)
    return value.strip("_")

def unique_path(base_path: Path) -> Path:
    """Если файл существует, добавляем _1, _2, ..."""
    if not base_path.exists():
        return base_path
    stem, suffix = base_path.stem, base_path.suffix
    i = 1
    while True:
        candidate = base_path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1

def read_csv_rows(path: Path):
    """Считывает CSV в список строк (list[list[str]]), игнорируя полностью пустые строки."""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = trim_trailing_empty(row)
            if any(str(c).strip() for c in row):
                rows.append(row)
    return rows

# --- Основная логика -------------------------------------------------------

def process_file(src: Path, out_dir: Path):
    rows = read_csv_rows(src)
    if len(rows) < 4:
        print(f"[WARN] {src} — мало строк, пропускаю.")
        return

    # Первая строка — имена мета-столбцов, вторая — значения мета-столбцов
    meta_cols = [c.strip() for c in rows[0]]
    meta_vals = [c.strip() for c in rows[1]]
    # Делаем маппинг, игнорируя возможные лишние столбцы из-за завершающей запятой
    m = {}
    for k, v in zip(meta_cols, meta_vals):
        if k:
            m[k] = v

    # Собираем имя файла из нужных полей
    wanted_keys = ["Well_Nm", "Well_Type", "Wl_Status", "County", "Field"]
    name_parts = []
    for k in wanted_keys:
        val = m.get(k, "unknown")
        name_parts.append(slugify(val))
    filename = "_".join(name_parts) or "unknown"

    # Ищем строку «боевого» хедера (после первых двух строк)
    data_start_idx = None
    for idx in range(2, len(rows)):
        low = [c.strip().lower() for c in rows[idx]]
        if "api_wellno" in low and "rptdate" in low:
            data_start_idx = idx
            break

    if data_start_idx is None:
        print(f"[WARN] {src} — не нашёл заголовок данных, пропускаю.")
        return

    data_rows = rows[data_start_idx:]  # заголовок + данные

    # Пишем результат
    # Папка вывода — где лежит сам скрипт
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # Если запускаем из интерактива — используем текущую рабочую директорию
        script_dir = Path.cwd()

    out_dir = script_dir if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = unique_path(out_dir / f"{filename}.csv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in data_rows:
            writer.writerow(row)

    print(f"[OK] {src.name} -> {out_path.name}")

def main():
    # Папка, где создадим новые CSV (по умолчанию — папка скрипта)
    out_dir = None  # можно заменить на Path("out") при желании отдельной папки

    if not SOURCE_DIR.exists():
        print(f"[ERROR] Не найдена папка: {SOURCE_DIR}")
        return

    csv_files = list(SOURCE_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"[INFO] В {SOURCE_DIR} не найдено CSV.")
        return

    print(f"Найдено файлов: {len(csv_files)}")
    for src in csv_files:
        try:
            process_file(src, out_dir)
        except Exception as e:
            print(f"[ERROR] Ошибка в {src}: {e}")

if __name__ == "__main__":
    main()
