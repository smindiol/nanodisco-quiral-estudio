"""Verifica integridad de datos y recursos del HTML sin usar la base cruda."""
from pathlib import Path
from html.parser import HTMLParser
import base64, io, json, re, xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build'
BUILD.mkdir(exist_ok=True)
class Check(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.images=[];self.external=[];self.text=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and a.get('href','').startswith('#'):self.links.append(a['href'][1:])
        if tag=='img' and a.get('src'):self.images.append(a['src']);assert a.get('alt')
        for k in ['src','href']:
            if a.get(k,'').startswith(('http:','https:')):self.external.append(a[k])
        if tag in ['script','style']:self.hidden+=1
    def handle_endtag(self,tag):
        if tag in ['script','style']:self.hidden-=1
    def handle_data(self,s):
        if not self.hidden:self.text.append(s)

page=(ROOT/'docs/explicacion_completa.html').read_text();c=Check();c.feed(page)
assert len(c.ids)==len(set(c.ids)), 'IDs duplicados'
assert set(c.links)<=set(c.ids), 'Índice roto'
assert not c.external, 'Recursos externos'
for src in c.images:
    mime,data=src.split(',',1);blob=base64.b64decode(data)
    if 'svg' in mime:ET.fromstring(blob)
    else:Image.open(io.BytesIO(blob)).verify()
d=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',page,re.S)[1])
assert len(d['auditoria']['momentos'])==56
assert len(d['magnetizacion'])==1128
assert len(d['auditoria']['texturas'])==128
for s in d['auditoria']['curvas']:
    assert len(s['T'])==141 and abs(s['T'][0]-20)<1e-4 and abs(s['T'][-1]-6)<1e-4
    for key in ['c_recalculado','energia_meV_por_espin','rhat_M','rhat_E','tau_M','tau_E']:assert len(s[key])==141
for r in d['magnetizacion']:assert 0<=r['m_media']<=1 and 0<=r['fraccion_var_entre']<=1
assert '<eq>' not in page
(BUILD/'texto_html.txt').write_text('\n'.join(c.text))
(BUILD/'datos_html.json').write_text(json.dumps(d,ensure_ascii=False))
print(f'PASS: {len(c.images)} imágenes válidas, sin recursos externos; índice íntegro; 8 estados × 141 temperaturas; 56 pruebas y 128 ternas de texturas.')

(BUILD/'explorador.js').write_text(re.search(r'</script><script>(.*?)</script>',page,re.S)[1])
