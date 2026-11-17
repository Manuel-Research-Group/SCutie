"""Funde objetos de um workspace do SCutie: os pixels do objeto origem passam a ser do objeto destino em todas as
máscaras, e o id de origem vira "None" em object_labels.json (mantém a sequência de ids; o Colmap ignora o id),
como já feito no superior (ids 5 e 6). As demais entradas da origem saem dos JSONs e das conexões do destino.

  python scripts/merge_objects.py --workspace workspaces/iph_solo_inferior_ds15_v12 --merge 21:5 --merge 16:17
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
from PIL import Image


def load(ws, name):
    p = os.path.join(ws, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(ws, name, data):
    with open(os.path.join(ws, name), 'w') as f:
        json.dump(data, f, indent=4)


ap = argparse.ArgumentParser()
ap.add_argument('--workspace', required=True)
ap.add_argument('--merge', action='append', required=True, help='origem:destino, ex. 21:5')
args = ap.parse_args()
ws = args.workspace
pairs = [tuple(int(v) for v in m.split(':')) for m in args.merge]

# 1. máscaras (PNG com paleta: troca o índice e preserva a paleta)
files = sorted(glob.glob(os.path.join(ws, 'masks', '*.png')))
px = {s: 0 for s, _ in pairs}; frames = {s: 0 for s, _ in pairs}
for f in files:
    im = Image.open(f)
    a = np.array(im); changed = False
    for src, dst in pairs:
        m = a == src
        if m.any():
            a[m] = dst; px[src] += int(m.sum()); frames[src] += 1; changed = True
    if changed:
        out = Image.fromarray(a, mode='P')
        out.putpalette(im.getpalette())
        out.save(f)
left = {s: sum(int((np.array(Image.open(f)) == s).any()) for f in files) for s, _ in pairs}
print(f'máscaras: {len(files)} arquivos')
for src, dst in pairs:
    print(f'  {src} -> {dst}: {px[src]} pixels em {frames[src]} quadros; quadros com {src} restantes: {left[src]}')

# 2. JSONs
before = {n: load(ws, n) for n in ('object_labels.json', 'models.json', 'sizes.json', 'references.json',
                                    'connections.json', 'possible_connections.json', 'clicks.json')}
labels = dict(before['object_labels.json'])
for src, _ in pairs:
    labels[str(src)] = 'None'
save(ws, 'object_labels.json', labels)
for n in ('models.json', 'sizes.json', 'references.json', 'possible_connections.json', 'clicks.json'):
    d = before[n]
    if d is None:
        continue
    d = {k: v for k, v in d.items() if k not in {str(s) for s, _ in pairs}}
    save(ws, n, d)
conn = {k: v for k, v in before['connections.json'].items() if k not in {str(s) for s, _ in pairs}}
pc = load(ws, 'possible_connections.json')
for src, dst in pairs:
    conn[str(dst)] = [c for c in conn[str(dst)] if c != src]
    if str(dst) in pc:
        pc[str(dst)] = len(conn[str(dst)])
save(ws, 'connections.json', conn)
save(ws, 'possible_connections.json', pc)

# 3. soft_masks vazias da origem
for src, _ in pairs:
    d = os.path.join(ws, 'soft_masks', str(src))
    if os.path.isdir(d):
        if os.listdir(d):
            print(f'  aviso: soft_masks/{src} não está vazia, mantida')
        else:
            shutil.rmtree(d)

# 4. relatório antes/depois
for n, old in before.items():
    if old is None:
        continue
    new = load(ws, n)
    for k in sorted(set(old) | set(new), key=lambda x: int(x) if x.isdigit() else x):
        if old.get(k) != new.get(k):
            o = old.get(k, '<ausente>'); w = new.get(k, '<removido>')
            if n == 'clicks.json':
                o, w = ('<cliques>' if k in old else '<ausente>'), ('<cliques>' if k in new else '<removido>')
            print(f'  {n} [{k}]: {o} -> {w}')
