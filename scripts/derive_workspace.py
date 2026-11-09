"""Deriva um workspace novo a partir de outro, aplicando edições só na cópia.

Exemplo (v10 a partir da v5):
    python scripts/derive_workspace.py \
        --src workspaces/iph_solo_superior_ds15_v5 \
        --dst workspaces/iph_solo_superior_ds15_v10 \
        --model 8=VPI-4448-02-inverted --model 46=VPI-4448-02-inverted \
        --remove-ids 49,50,51,52,53,55,56,58 \
        --disable-inverted-flag

Garantias: escreve apenas em --dst; recusa se --dst já existir ou se
--src/--dst forem o mesmo diretório ou um estiver dentro do outro.
Objetos removidos mantêm o id (label "None"), então connections.json segue válido.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

# JSONs por objeto dos quais um id removido sai por completo.
DROP_FILES = ("models.json", "sizes.json", "references.json", "clicks.json")


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def _dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _is_within(child: str, parent: str) -> bool:
    return os.path.commonpath([child, parent]) == parent


def _parse_models(items):
    models = {}
    for item in items:
        obj_id, sep, name = item.partition("=")
        if not sep or not obj_id.strip().isdigit() or not name.strip():
            raise SystemExit(f"--model inválido: {item!r} (use ID=NOME)")
        models[str(int(obj_id))] = name.strip()
    return models


def _parse_ids(text):
    if not text:
        return []
    return sorted({int(x) for x in text.split(",") if x.strip()})


def rename_models(dst, models):
    if not models:
        return
    path = os.path.join(dst, "models.json")
    data = _load(path) if os.path.exists(path) else {}
    data.update(models)
    _dump(path, data)


def remove_objects(dst, ids):
    keys = {str(i) for i in ids}
    for name in DROP_FILES:
        path = os.path.join(dst, name)
        if os.path.exists(path):
            data = _load(path)
            _dump(path, {k: v for k, v in data.items() if k not in keys})

    labels_path = os.path.join(dst, "object_labels.json")
    if os.path.exists(labels_path):
        labels = _load(labels_path)
        for k in keys:
            labels[k] = "None"
        _dump(labels_path, labels)

    for k in keys:
        shutil.rmtree(os.path.join(dst, "soft_masks", k), ignore_errors=True)

    changed = 0
    for mask_path in sorted(glob.glob(os.path.join(dst, "masks", "*.png"))):
        with Image.open(mask_path) as img:
            mode, palette = img.mode, img.getpalette()
            arr = np.array(img)
        hit = np.isin(arr, ids)
        if not hit.any():
            continue
        arr[hit] = 0
        out = Image.fromarray(arr, mode=mode)
        if mode == "P" and palette is not None:
            out.putpalette(palette)
        out.save(mask_path)
        changed += 1
    return changed


def disable_inverted_flag(dst):
    inv = os.path.join(dst, "inverted.json")
    if os.path.exists(inv):
        os.remove(inv)
    cfg_path = os.path.join(dst, "workspace_config.json")
    cfg = _load(cfg_path) if os.path.exists(cfg_path) else {}
    cfg["use_inverted_flag"] = False
    _dump(cfg_path, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--model", action="append", default=[], metavar="ID=NOME")
    ap.add_argument("--remove-ids", default="", metavar="1,2,3")
    ap.add_argument("--disable-inverted-flag", action="store_true")
    args = ap.parse_args(argv)

    src = os.path.realpath(args.src)
    dst = os.path.realpath(args.dst)
    if not os.path.isdir(src):
        sys.exit(f"Origem não encontrada: {src}")
    if _is_within(dst, src) or _is_within(src, dst):
        sys.exit("Origem e destino não podem ser o mesmo diretório nem estar um dentro do outro.")
    if os.path.exists(dst):
        sys.exit(f"Destino já existe, nada foi alterado: {dst}")

    models = _parse_models(args.model)
    ids = _parse_ids(args.remove_ids)

    shutil.copytree(src, dst)
    rename_models(dst, models)
    changed = remove_objects(dst, ids) if ids else 0
    if args.disable_inverted_flag:
        disable_inverted_flag(dst)

    print(f"Workspace criado: {dst}")
    print(f"  modelos renomeados: {models or '-'}")
    print(f"  ids removidos: {ids or '-'} ({changed} máscaras alteradas)")
    print(f"  use_inverted_flag: {'false' if args.disable_inverted_flag else 'inalterado'}")


if __name__ == "__main__":
    main()
