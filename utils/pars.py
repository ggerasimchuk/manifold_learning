#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
import itertools
from pathlib import Path
from typing import Set, Dict, Any

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout, Page

URL = "https://bogapps.dnrc.mt.gov/dataminer/Production/ProductionByWell.aspx"

# ---------------------------
# УТИЛИТЫ
# ---------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "file"

def load_state(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text("utf-8"))
        return {k: set(v) for k, v in raw.items()}
    except Exception:
        return {}

def save_state(path: Path, state: Dict[str, Set[str]]) -> None:
    serializable: Dict[str, Any] = {k: sorted(list(v)) for k, v in state.items()}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), "utf-8")

def unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for i in itertools.count(1):
        cand = base.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand

REC_RANGE_RE = re.compile(r"Records?\s+(\d+)\s+to\s+(\d+)", re.I)

def expected_count_from_group_name(name: str) -> int:
    m = REC_RANGE_RE.search(name or "")
    if not m:
        return 300
    a, b = int(m.group(1)), int(m.group(2))
    return max(1, b - a + 1)

def normalize_group_dir_name(group_label: str) -> str:
    m = REC_RANGE_RE.search(group_label or "")
    if m:
        return f"records {m.group(1)} - {m.group(2)}"
    return sanitize_filename(group_label).lower()

def get_group_index_from_href(href: str) -> int:
    # javascript:__doPostBack('...dvTreeView','s0\\1')
    m = re.search(r"s0\\\\(\d+)", href or "")
    if not m:
        raise RuntimeError(f"Не удалось извлечь индекс группы из href={href!r}")
    return int(m.group(1))

def well_selector_for_group(group_index: int) -> str:
    # Только листовые узлы (скважины) внутри конкретной группы:
    # <a class="leafnode treenode" href="...,'s0\\{group_index}\\{WELL_ID}')">
    return rf"a.leafnode.treenode[href*='s0\\\\{group_index}\\\\']"

def load_all_wells_for_group(page: Page, group_index: int, expected: int, timeout_ms: int = 60000):
    """
    На странице используется ленивая отрисовка списка скважин.
    Крутим вниз, пока счётчик растёт или не достигли expected/таймаут.
    """
    sel = well_selector_for_group(group_index)
    last = -1
    stable_rounds = 0

    # простой таймер без доступа к приватным атрибутам
    from time import monotonic
    deadline = monotonic() + (timeout_ms / 1000)

    while True:
        cnt = page.locator(sel).count()
        if cnt >= expected:
            break
        if cnt == last:
            stable_rounds += 1
        else:
            stable_rounds = 0
            last = cnt
        if stable_rounds >= 3:
            break
        if monotonic() > deadline:
            break
        page.mouse.wheel(0, 2400)
        page.wait_for_timeout(200)

    # Возвращаемся наверх, чтобы клики были стабильнее
    page.evaluate("window.scrollTo(0,0)")
    return page.locator(sel)

# ---------------------------
# СЕЛЕКТОРЫ/ДЕЙСТВИЯ
# ---------------------------

def find_search_input(page: Page):
    # Поле перед кнопкой Search
    loc = page.get_by_role("textbox")
    if loc.count() > 0:
        return loc.first
    # запасные варианты
    return page.locator("input[type='text']").first

def get_record_group_links(page: Page):
    # Ссылки вида "Records 1 to 300"
    # Нельзя брать все .treenode, иначе попадут и другие узлы.
    return page.locator("a.treenode", has_text=re.compile(r"^\s*Records?\s+\d+\s+to\s+\d+\s*$", re.I))

def click_data_list(page: Page):
    link = page.get_by_role("link", name=re.compile(r"^\s*Data\s+List\s*$", re.I))
    if link.count() == 0:
        link = page.locator("a.treenode", has_text=re.compile(r"^\s*Data\s+List\s*$", re.I))
    link.first.wait_for(state="visible")
    link.first.scroll_into_view_if_needed()
    link.first.click()

def click_csv_and_download(page: Page, out_path: Path, well_label: str, timeout_ms: int = 60000) -> Path:
    csv_link = page.get_by_role("link", name=re.compile(r"csv\s*\(excel\)", re.I))
    if csv_link.count() == 0:
        csv_link = page.locator("a", has_text=re.compile(r"csv\s*\(excel\)", re.I))
    if csv_link.count() == 0:
        csv_link = page.get_by_role("button", name=re.compile(r"csv", re.I))

    base_name = sanitize_filename(well_label) + ".csv"
    file_path = unique_path(out_path / base_name)

    # Попытка 1 — клик по UI
    try:
        if csv_link.count() > 0:
            with page.expect_download(timeout=timeout_ms) as dl_info:
                csv_link.first.scroll_into_view_if_needed()
                csv_link.first.click()
            download = dl_info.value
            download.save_as(str(file_path))
            return file_path
    except Exception:
        pass

    # Фолбэк — прямой вызов экспортного URL в рамках текущей сессии
    with page.expect_download(timeout=timeout_ms) as dl_info:
        page.evaluate("window.location.href = '/dataminer/Export.aspx?type=csv';")
    download = dl_info.value
    download.save_as(str(file_path))
    return file_path

# ---------------------------
# ОСНОВНОЙ СЦЕНАРИЙ
# ---------------------------

def run(out_root: Path, headful: bool, slowmo: int, timeout_ms: int,
        start_group: int, max_groups: int):

    out_root.mkdir(parents=True, exist_ok=True)
    state_file = out_root / "state.json"
    state: Dict[str, Set[str]] = load_state(state_file)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful, slow_mo=slowmo or 0)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        log(f"Открываю {URL}")
        page.goto(URL, wait_until="domcontentloaded")

        # 1) Ввод '%' и Search
        log("Ввожу '%' и нажимаю 'Search'...")
        search_input = find_search_input(page)
        search_input.fill("%")

        btn = page.get_by_role("button", name=re.compile(r"^\s*Search\s*$", re.I))
        if btn.count() == 0:
            btn = page.locator("input[type='submit'][value='Search']")
        btn.first.click()

        # Ждём появления 'Data List'
        page.get_by_role("link", name=re.compile(r"^\s*Data\s+List\s*$", re.I)).first.wait_for(state="visible", timeout=timeout_ms)

        # 2) Открываем Data List
        log("Открываю 'Data List'...")
        click_data_list(page)
        page.wait_for_timeout(300)

        # 3) Получаем группы
        groups = get_record_group_links(page)
        total_groups = groups.count()
        if total_groups == 0:
            raise RuntimeError("Группы 'Records X to Y' не найдены. Проверьте селекторы.")

        log(f"Найдено групп: {total_groups}")
        start_idx = max(start_group, 0)
        end_idx = total_groups if max_groups in (0, None) else min(total_groups, start_idx + max_groups)

        for gi in range(start_idx, end_idx):
            group_link = groups.nth(gi)
            group_name = group_link.inner_text().strip()
            group_href = group_link.get_attribute("href") or ""
            try:
                group_index = get_group_index_from_href(group_href)
            except RuntimeError as e:
                log(f"⚠️ Пропускаю группу: {e}")
                continue
            expected = expected_count_from_group_name(group_name)

            group_dir = out_root / normalize_group_dir_name(group_name)
            group_dir.mkdir(parents=True, exist_ok=True)

            done_for_group = state.get(group_name, set())

            log(f"[Группа {gi+1}/{total_groups}] Открываю '{group_name}' (ожидаю ~{expected})")
            group_link.scroll_into_view_if_needed()
            group_link.click()

            # Ждём появления хотя бы одной скважины этой группы
            try:
                page.locator(well_selector_for_group(group_index)).first.wait_for(state="visible", timeout=timeout_ms)
            except PlaywrightTimeout:
                log(f"  ⚠️ Скважины не отрисовались для '{group_name}'. Пропускаю.")
                continue

            # Догружаем список (ленивая подгрузка)
            wells = load_all_wells_for_group(page, group_index, expected, timeout_ms=timeout_ms)
            count_wells = wells.count()
            log(f"  Скважин в группе: {count_wells}")

            for wi in range(count_wells):
                well = wells.nth(wi)
                well_label = well.inner_text().strip()

                # Защита от мусорных узлов
                if not re.match(r"^\s*\d{6,}\s*-\s*.+", well_label):
                    continue

                if well_label in done_for_group:
                    log(f"  [{wi+1}/{count_wells}] Уже скачано: {well_label}")
                    continue

                log(f"  [{wi+1}/{count_wells}] Открываю скважину '{well_label}'")
                well.scroll_into_view_if_needed()
                well.click()

                # Ждём появления ссылки Csv (Excel)
                try:
                    page.get_by_role("link", name=re.compile(r"csv\s*\(excel\)", re.I)).first.wait_for(
                        state="visible", timeout=timeout_ms
                    )
                except PlaywrightTimeout:
                    log(f"      ⚠️ Csv (Excel) не появилась для '{well_label}', пробую fallback.")

                # Скачиваем
                try:
                    file_path = click_csv_and_download(page, group_dir, well_label, timeout_ms=timeout_ms)
                    log(f"      ✓ Сохранено: {file_path.name}")
                    done_for_group.add(well_label)
                    state[group_name] = done_for_group
                    save_state(state_file, state)
                except Exception as e:
                    log(f"      ✖ Ошибка скачивания CSV для '{well_label}': {e}")

            log(f"Группа '{group_name}' завершена.")

        context.close()
        browser.close()

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Скачивание CSV по всем скважинам (DNRC DataMiner).")
    ap.add_argument("--out", default="./records", help="Корневая папка для выгрузки.")
    ap.add_argument("--headful", action="store_true", help="Открывать браузер в видимом режиме.")
    ap.add_argument("--slowmo", type=int, default=0, help="Замедление действий (мс).")
    ap.add_argument("--timeout", type=int, default=12000, help="Таймаут ожиданий (мс).")
    ap.add_argument("--start-group", type=int, default=0, help="С какой группы начинать (0 — первая).")
    ap.add_argument("--max-groups", type=int, default=0, help="Сколько групп обрабатывать (0 — все).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        out_root=Path(args.out),
        headful=args.headful,
        slowmo=args.slowmo,
        timeout_ms=args.timeout,
        start_group=args.start_group,
        max_groups=args.max_groups,
    )
