"""Comprueba el contenido que se compartirá: fuentes, derivados y exclusión de crudos."""
from pathlib import Path
import ast
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
errors=[]
count=0
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if any(part in {'.git','build','.venv','__pycache__'} for part in rel.parts) or not p.is_file():
        continue
    count+=1
    if p.stat().st_size>25*1024*1024:
        errors.append(f'Archivo mayor que el límite de revisión: {rel}')
    if re.fullmatch(r'T_\d+(?:\.\d+)?\.npz',p.name):
        errors.append(f'Archivo crudo incluido: {rel}')
    if p.suffix=='.py':
        try:ast.parse(p.read_text(),filename=str(rel))
        except SyntaxError as e:errors.append(f'Error sintáctico en {rel}: {e.msg}')
    if p.suffix=='.ipynb':
        d=json.loads(p.read_text())
        for cell in d['cells']:
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'Cuaderno con salidas de sesión: {rel}')
    if p.suffix in {'.py','.sh','.md','.json','.ipynb','.html','.yml','.txt','.tex'}:
        s=p.read_text(errors='replace')
        if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',s):
            errors.append(f'Posible clave privada en {rel}')
        if re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}',s):
            errors.append(f'Posible credencial GitHub en {rel}')
        if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/', s) and p.suffix in {'.py','.sh'}:
            errors.append(f'Ruta local no portable en {rel}')
if errors:
    print('\n'.join(errors));sys.exit(1)
print(f'PASS: {count} archivos revisados; fuentes Python válidas, cuadernos sin salidas, sin crudos ni credenciales detectadas.')
