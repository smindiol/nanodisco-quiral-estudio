// Prueba funcional de los controles con un DOM mínimo; no sustituye revisión visual.
const fs=require('fs'),vm=require('vm'),path=require('path');
const base=path.resolve(__dirname,'../build'),data=JSON.parse(fs.readFileSync(path.join(base,'datos_html.json'),'utf8'));
let calls=0,downloads=0;
const inputs=[];
class El{constructor(id){this.id=id;this.value='';this.checked=true;this.dataset={};this.children=[];this.textContent='';this._html='';this.width=195;this.height=195}set innerHTML(s){this._html=s;if(this.id==='legend'){inputs.length=0;for(const m of s.matchAll(/data-e="(\d+)"/g)){let e=new El('input');e.dataset.e=m[1];inputs.push(e)}}}get innerHTML(){return this._html}addEventListener(k,f){this['on'+k]=f}appendChild(e){this.children.push(e)}querySelector(){return new El('canvas')}getContext(){return {fillRect(){calls++},set fillStyle(x){}}}click(){downloads++}showModal(){}close(){}}
const ids=['data','metric','scale','curve','metric-note','tempcompare','tempvalue','temp-table','legend','allstates','download','snaptemp','copy','snapvalue','textures','animate','audit-download','moments-download','image-dialog','modal-image','close-modal'];
const els=Object.fromEntries(ids.map(x=>[x,new El(x)]));
els.data.textContent=JSON.stringify(data);els.metric.value='c_recalculado';els.scale.value='linear';els.tempcompare.value='57';els.snaptemp.value='0';els.copy.value='0';
const ctx={document:{getElementById:k=>{if(!els[k])throw Error('ID inexistente: '+k);return els[k]},querySelectorAll:q=>q==='[data-e]'?inputs:[],createElement:k=>new El(k)},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Blob,Set,Map,Math,JSON,Number,setInterval:()=>1,clearInterval(){},setTimeout:f=>f()};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(base,'explorador.js'),'utf8'),ctx);
let graphs=0;
for(const metric of ['c_recalculado','energia_meV_por_espin','m_media','chi_protocolo','fraccion_var_entre','rhat_M','rhat_E','tau_M','tau_E'])for(const scale of ['linear','log']){els.metric.value=metric;els.scale.value=scale;els.metric.onchange();if(!els.curve.innerHTML.includes('<svg')||/NaN|Infinity/.test(els.curve.innerHTML))throw Error('Curva inválida: '+metric+'/'+scale);graphs++}
for(let t=0;t<141;t++){els.tempcompare.value=String(t);els.tempcompare.oninput();if(/No disponible|undefined|NaN/.test(els['temp-table'].innerHTML))throw Error('Dato ausente en tabla '+t)}
for(let copy=0;copy<3;copy++)for(let t=0;t<16;t++){els.copy.value=String(copy);els.snaptemp.value=String(t);els.snaptemp.oninput()}
for(const e of inputs){e.checked=false;e.onchange()}if(!els.curve.textContent.includes('Seleccione'))throw Error('Selección vacía');els.allstates.onclick();
els.download.onclick();els['audit-download'].onclick();els['moments-download'].onclick();if(downloads!==3)throw Error('Descargas');
console.log(`PASS: ${graphs} combinaciones de curva/escala, 141 tablas de temperatura, 48 vistas de texturas de los 8 estados y 3 descargas. ${calls} celdas dibujadas. DOM simulado; no es validación visual de navegador.`);
