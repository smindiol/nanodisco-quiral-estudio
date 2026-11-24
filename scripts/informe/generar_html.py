"""Construye una guía autónoma a partir del main final y la auditoría local.
Las fórmulas se convierten en SVG: el lector no requiere internet ni MathJax.
"""
from pathlib import Path
import base64, io, json, re, html, subprocess
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuracion import ROOT, RESULTADOS, AUDITORIA, BUILD, archivo_crudo

BASE=ROOT/'informe/latex'
BUILD.mkdir(parents=True, exist_ok=True)
A=json.loads((AUDITORIA/'auditoria.json').read_text())
M=json.loads((AUDITORIA/'magnetizacion.json').read_text())
SECTIONS=[]

def eq(s):
    buf=io.BytesIO()
    math_to_image('$'+s+'$',buf,format='svg',dpi=130,color='#132d45')
    return '<div class="equation"><img alt="'+html.escape(s,quote=True)+'" src="data:image/svg+xml;base64,'+base64.b64encode(buf.getvalue()).decode()+'"></div>'

def table(head,rows,caption=''):
    return '<div class="table-wrap"><table><caption>'+caption+'</caption><thead><tr>'+''.join('<th scope="col">'+str(c)+'</th>' for c in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure(name,caption):
    p=BASE/'fig'/name
    if p.suffix=='.pdf':
        out=BUILD/'figuras_web'/p.stem
        out.parent.mkdir(exist_ok=True)
        target=out.with_suffix('.png')
        if not target.exists(): subprocess.run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(p),str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        p=target
    data=base64.b64encode(p.read_bytes()).decode()
    return '<figure><img loading="lazy" class="original" tabindex="0" alt="'+html.escape(caption,quote=True)+'" src="data:image/png;base64,'+data+'"><figcaption>'+caption+' <span class="source">Fuente: informe final, fig/'+name+'.</span></figcaption></figure>'

def sec(id,title,text):
    text=re.sub(r'<eq>(.*?)</eq>',lambda m:eq(m.group(1)),text,flags=re.S)
    SECTIONS.append((id,title,text))

sec('problema','1. Problema y alcance del estudio',r'''
<p>Un nanodisco magnético contiene espines que interactúan con sus vecinos y fluctúan con la temperatura. Cada espín es un vector de norma uno. El intercambio favorece la alineación; la interacción de Dzyaloshinskii y Moriya favorece giros entre vecinos. La anisotropía selecciona direcciones preferidas y el campo externo favorece una orientación perpendicular al disco. Al enfriar, estas contribuciones compiten y producen configuraciones con distinto orden espacial.</p>
<p>El problema consiste en determinar cómo cambia ese orden durante el enfriamiento y cuánto depende de la interacción quiral y del campo. Para responderlo hay que resolver, a la vez, una dificultad estadística: las configuraciones proceden de cadenas de Metropolis que pueden permanecer en regiones diferentes del espacio de estados. Una curva suave o una serie estable no bastan para demostrar equilibrio.</p>
<p>El objeto simulado es un disco de radio 18.5 sitios, cinco capas y 5425 espines, recortado de una red de 39 × 39 × 5 posiciones, con bordes libres. El estudio contiene ocho combinaciones de parámetros. Cada combinación tiene 100 caminos de enfriamiento independientes que recorren 141 temperaturas entre 20 y 6 K, en pasos de 0.1 K. En cada temperatura se descartan 25000 barridos y se registran 10000. Un barrido ofrece una actualización a cada espín.</p>
<p>La configuración final de una temperatura inicia la siguiente. Por eso las temperaturas de un mismo camino están relacionadas, aunque los 100 caminos usen claves aleatorias distintas. El tiempo del algoritmo se mide en barridos; no representa una duración física calibrada en segundos.</p>
<div class="note"><strong>Dos objetos de inferencia.</strong> El equilibrio corresponde a la distribución de Boltzmann a temperatura fija. El protocolo corresponde a la distribución de resultados producida por este enfriamiento finito. Si falla la mezcla global, los datos siguen siendo útiles para estudiar memoria, diversidad de texturas, atrapamiento y costo del algoritmo. Esa utilidad exige nombrar correctamente qué distribución describen.</div>
<p>Esta guía desarrolla el <code>main.tex</code> de <code>informe_final</code>, sus secciones y sus figuras. Conserva las estimaciones originales de la transición y añade comprobaciones sobre los datos existentes. No representa una nueva simulación ni una repetición del bootstrap original de 50000 réplicas.</p>
'''+table(['Estado','k<sub>DM</sub>','H','Lote','Caminos','Temperaturas'],[[i,s['kDM'],s['H'],(i+1)//2,100,141] for i,s in enumerate(A['curvas'],1)],'Diseño simulado. J = 1 y K = 0.05; H actúa como coeficiente del término de Zeeman. Energías y campo están normalizados y son adimensionales.')+r'''
<p>Los ocho estados no forman un diseño factorial completo: los tres casos con campo tienen valores distintos de k<sub>DM</sub>. Solo los estados 4 y 7 comparten k<sub>DM</sub> = 0.6 con distinto campo. Este par permite una comparación puntual; los demás contrastes describen combinaciones de parámetros y no aíslan por sí solos el efecto causal de H.</p>
''')

sec('preguntas','2. Preguntas, hipótesis y criterios de respuesta',r'''
<p>Las hipótesis se formulan sobre cantidades identificables en los datos. Las pruebas diagnósticas sirven para detectar fallas concretas del muestreo; un resultado favorable no demuestra que se hayan visitado todas las configuraciones relevantes.</p>
'''+table(['Pregunta','Hipótesis o expectativa','Evidencia que la evalúa','Alcance de la respuesta'],[
['¿Dónde cambia el régimen térmico?','El calor específico tiene un máximo interior reproducible.','Curvas c(T), repesado, bootstrap y diagnósticos de E y de su segundo momento.','Temperatura característica del disco y del protocolo; equilibrio condicionado a mezcla suficiente.'],
['¿Varía T* con kDM?','H₀: asociación de rangos nula en los ocho estados.','Spearman y permutación exacta de 40320 asignaciones.','No rechazar H₀ no prueba equivalencia ni ausencia de efecto.'],
['¿El campo modifica la transición?','Las combinaciones con campo pueden desplazar el máximo.','Diferencias bootstrap y corrección de Holm; diseño de parámetros.','Distingue estados, pero el campo queda parcialmente confundido con kDM.'],
['¿Muestrean los caminos una misma distribución?','Los caminos deberían coincidir dentro del error Monte Carlo.','R-hat, homogeneidad, Geweke y truncamiento MSER-5.','Puede refutar convergencia; no certificar equilibrio global.'],
['¿Cómo cambia el costo al enfriar?','La memoria temporal aumenta y la mezcla puede estancarse.','Autocorrelación, tamaño efectivo y R-hat frente a barridos.','Costo observado; extrapolaciones largas solo orientan.'],
['¿Hay poblaciones finales diferentes?','Una sola normal puede ser insuficiente para describir las medias finales.','EM, BIC, carga topológica y factor de estructura.','Poblaciones del protocolo; sus frecuencias no son pesos de equilibrio demostrados.']
],'Relación entre problema, hipótesis, método y conclusión.')+r'''
<p>La independencia necesaria para la prueba exacta de Spearman se refiere a la intercambiabilidad de los estados bajo la hipótesis nula. Dado que los parámetros fueron elegidos y el diseño es pequeño, su interpretación es descriptiva y condicional al diseño. Los valores p no convierten ocho combinaciones en una muestra representativa de todos los nanodiscos posibles.</p>
<p>Para estudiar utilidad científica conviene conservar también las hipótesis físicas: una espiral de período más corto debería caber mejor en el disco; un campo podría estabilizar texturas de carga distinta. Las texturas y los diagnósticos pueden apoyar estas explicaciones, pero no sustituyen un experimento de mezcla ni una comparación factorial.</p>
''')

sec('modelo','3. Modelo matemático: de los espines a la distribución objetivo',r'''
<h3>3.1. Configuración y grados de libertad</h3>
<p>El espín i tiene tres componentes, pero su norma impone una restricción. Cada espín aporta dos grados de libertad. La configuración completa pertenece al producto de 5425 esferas y tiene 10850 grados de libertad continuos.</p>
<eq>\mathbf{s}=(\mathbf{S}_1,\ldots,\mathbf{S}_N),\quad |\mathbf{S}_i|=1,\quad \mathbf{s}\in(\mathbb{S}^{2})^N</eq>
<p>En las siguientes ecuaciones, E designa la energía física total de una configuración y H el parámetro del campo. Esta distinción evita confundir el campo con el hamiltoniano. El archivo crudo almacena una energía normalizada E/E₀; el análisis debe multiplicarla por E₀ antes de calcular momentos.</p>
<h3>3.2. Las cuatro contribuciones de energía</h3>
<eq>E(\mathbf{s})=E_{\mathrm{exc}}+E_{\mathrm{DM}}+E_{\mathrm{an}}+E_{\mathrm{campo}}</eq>
<eq>E_{\mathrm{exc}}=-J\sum_{\langle ij\rangle}\mathbf{S}_i\cdot\mathbf{S}_j</eq>
<p>El producto escalar vale 1 para espines paralelos y −1 para antiparalelos. Con J positivo, la energía disminuye al alinearse. Cada enlace debe contarse una sola vez. El código obtiene la suma sobre vecinos de cada sitio y aplica un factor 1/2 a la energía de intercambio para evitar duplicarla.</p>
<eq>E_{\mathrm{DM}}=-k_{DM}\sum_i[\hat{x}\cdot(\mathbf{S}_i\times\mathbf{S}_{i+\hat{x}})+\hat{y}\cdot(\mathbf{S}_i\times\mathbf{S}_{i+\hat{y}})]</eq>
<p>El producto vectorial cambia de signo al invertir el orden de los espines. Por ello esta interacción distingue sentidos de giro. El modelo incluye los enlaces quirales del plano; no añade una contribución DM entre capas. Los enlaces que salen de la máscara del disco no contribuyen.</p>
<eq>E_{\mathrm{an}}=K\sum_i[(S_i^x S_i^y)^2+(S_i^x S_i^z)^2+(S_i^y S_i^z)^2]</eq>
<p>Con K positivo, este término vale cero sobre los ejes y penaliza direcciones con varias componentes no nulas. No es una anisotropía uniaxial del tipo −K(Sᶻ)²: cambiar esa expresión cambiaría el problema.</p>
<eq>E_{\mathrm{campo}}=-H\sum_i S_i^z</eq>
<p>Un espín con componente z positiva reduce la energía cuando H es positivo. En este código H es un parámetro normalizado y adimensional, como las energías.</p>
<h3>3.3. Por qué aparece una espiral</h3>
<p>En una cadena ideal y para un giro compatible con el vector DM, la energía por enlace es −J cos δ − k<sub>DM</sub> sin δ. Al derivar respecto al ángulo y anular la derivada se obtiene:</p>
<eq>J\sin\delta-k_{DM}\cos\delta=0,\quad \delta=\arctan(k_{DM}/J),\quad \lambda=\frac{2\pi}{\delta}</eq>
<p>El período λ cuenta los sitios necesarios para completar una vuelta. Esta relación ofrece una escala física de referencia. Un disco finito con varias capas, anisotropía y campo no tiene por qué seguirla exactamente; el período medido depende además de los bordes y de la resolución del análisis de Fourier.</p>
<h3>3.4. Distribución de Boltzmann e integración</h3>
<eq>\pi_\beta(\mathbf{s})=\frac{\exp[-\beta E(\mathbf{s})]}{Z(\beta)},\quad \beta=\frac{1}{k_B T}</eq>
<eq>Z(\beta)=\int_{(\mathbb{S}^2)^N}\exp[-\beta E(\mathbf{s})]\,d\Omega,\quad k_B=0.086173404</eq>
<p>La medida dΩ es el producto de elementos de área sobre las esferas. Z normaliza la distribución. A temperatura alta, las diferencias de energía pesan menos; al enfriar, las configuraciones de menor energía reciben más probabilidad. El mínimo de energía no representa por sí solo el equilibrio: también importa cuántas configuraciones tienen energías similares.</p>
<eq>\langle A\rangle_\beta=\int A(\mathbf{s})\pi_\beta(\mathbf{s})\,d\Omega\ \simeq\ \frac{1}{n}\sum_{t=1}^n A(\mathbf{s}_t)</eq>
<p>La integral tiene demasiadas dimensiones para una cuadratura directa. Monte Carlo la aproxima con estados de una cadena. La aproximación exige explorar adecuadamente la distribución objetivo; si la cadena queda atrapada, el promedio puede describir solo un sector.</p>
''')

sec('simulacion','4. Simulación: propuesta, aceptación y enfriamiento',r'''
<h3>4.1. Transformada inversa sobre la esfera</h3>
<p>La propuesta debe cubrir uniformemente el área de la esfera. En coordenadas esféricas, el área contiene el jacobiano sin θ. Por ello un ángulo polar uniforme acumularía propuestas cerca de los polos y sería incorrecto para este algoritmo.</p>
<eq>d\Omega=\sin\theta\,d\theta\,d\varphi,\quad f(\theta,\varphi)=\frac{\sin\theta}{4\pi}</eq>
<p>El cambio z = cos θ tiene derivada de magnitud sin θ. Al transformar la densidad, ese factor se cancela y z resulta uniforme en [−1,1]. La coordenada azimutal es uniforme en [0,2π).</p>
<eq>f_z(z)=\frac{1}{2},\quad F_z(z)=\frac{z+1}{2},\quad z=2U_1-1,\quad \varphi=2\pi U_2</eq>
<eq>\mathbf{S}'=(\sqrt{1-z^2}\cos\varphi,\sqrt{1-z^2}\sin\varphi,z)</eq>
<p>U₁ y U₂ son uniformes independientes. La propuesta tiene norma uno porque la suma de sus componentes al cuadrado vale (1−z²)(cos²φ+sin²φ)+z²=1. La densidad de propuesta es constante, 1/(4π), tanto al ir del espín actual al propuesto como al regresar.</p>
<h3>4.2. Aceptación de Metropolis y balance</h3>
<eq>\alpha(\mathbf{s},\mathbf{s}')=\min\left(1,\frac{\pi_\beta(\mathbf{s}')q(\mathbf{s}\mid\mathbf{s}')}{\pi_\beta(\mathbf{s})q(\mathbf{s}'\mid\mathbf{s})}\right)=\min(1,e^{-\beta\Delta E})</eq>
<p>La constante Z y las densidades simétricas de propuesta se cancelan. Si ΔE es negativa se acepta siempre. Si es positiva se acepta cuando una nueva uniforme es menor que exp(−βΔE). Por ejemplo, para ΔE = 0.1, la probabilidad es aproximadamente 0.944 a 20 K y 0.824 a 6 K. El enfriamiento dificulta salir de una región mediante pasos que aumenten la energía.</p>
<eq>\pi(\mathbf{s})q(\mathbf{s}'\mid\mathbf{s})\alpha(\mathbf{s},\mathbf{s}')=\min[\pi(\mathbf{s})q(\mathbf{s}'\mid\mathbf{s}),\pi(\mathbf{s}')q(\mathbf{s}\mid\mathbf{s}')]</eq>
<p>La expresión es simétrica al intercambiar los estados; demuestra balance detallado de la actualización elemental. El código divide la red en dos subredes sin vecinos dentro de una misma subred. Sus espines se pueden actualizar en paralelo condicionados a la otra. La composición de ambas actualizaciones preserva la distribución objetivo, aunque la composición no necesite ser reversible.</p>
<h3>4.3. Qué significa un camino independiente</h3>
<p>Los 100 caminos usan claves aleatorias distintas. Cada uno continúa su propia configuración al pasar a la temperatura siguiente. Dos temperaturas del mismo camino no constituyen réplicas independientes. La independencia entre caminos permite remuestrear caminos completos y comparar su dispersión.</p>
<p>El protocolo tiene 141 × 35000 = 4935000 barridos por camino, de los cuales 1410000 se registran. Los 25000 barridos descartados en cada temperatura son una decisión de diseño, no una prueba de que desapareció todo transitorio. Esa etapa no quedó guardada como serie temporal completa; las pruebas disponibles examinan la ventana posterior de medición.</p>
<h3>4.4. Validación original del generador</h3>
<p>El informe generó cuatro millones de direcciones por cada una de 100 semillas. La región de uniformes se dividió en 64 celdas equiprobables. Bajo la hipótesis de uniformidad, cada celda espera n/64 observaciones y el contraste usa 63 grados de libertad.</p>
<eq>X^2=\sum_{c=1}^{64}\frac{(O_c-n/64)^2}{n/64}\ \sim\ \chi^2_{63}</eq>
<p>Hubo 2 rechazos conjuntos, 5 para z y 7 para φ al nivel 0.05. La probabilidad binomial de observar al menos esos números de rechazos fue 0.96, 0.56 y 0.23. Los segundos momentos de las componentes fueron 0.33333, 0.33334 y 0.33332, próximos a 1/3. Estos resultados no muestran una desviación detectable en las pruebas realizadas; no validan por sí solos las interacciones, el equilibrio o todas las propiedades del generador. Los resultados proceden del análisis original, no de una nueva ejecución de sus 400 millones de propuestas.</p>
''')

sec('estimadores','5. Observables, calor específico e incertidumbre',r'''
<h3>5.1. Magnetización y energía</h3>
<eq>\mathbf{M}=\frac{1}{N}\sum_i\mathbf{S}_i,\quad m=|\mathbf{M}|,\quad e=\frac{E}{N}</eq>
<p>m está entre cero y uno. Una magnetización pequeña puede corresponder a espines desordenados o a una textura ordenada cuyos dominios se cancelan. Por eso m no clasifica por sí sola espirales y skyrmiones. El factor de estructura y la carga topológica aportan información espacial complementaria.</p>
<div class="note"><strong>Precisión física.</strong> Una rotación rígida de todos los espines conserva |ΣSᵢ| y, por tanto, m. Diferencias en la norma de magnetización entre caminos requieren cambios en las alineaciones relativas, en la fracción de dominios o en la textura. No pueden atribuirse solo a una orientación global diferente.</div>
<h3>5.2. Derivación del calor específico</h3>
<p>La energía media se obtiene al derivar el logaritmo de la función de partición. Una segunda derivada produce la varianza de la energía.</p>
<eq>\frac{\partial\log Z}{\partial\beta}=-\langle E\rangle,\quad \frac{\partial\langle E\rangle}{\partial\beta}=-(\langle E^2\rangle-\langle E\rangle^2)</eq>
<p>Como dβ/dT = −1/(kBT²), la regla de la cadena convierte esa derivada en una respuesta térmica positiva. Dividir por N da el calor específico por espín:</p>
<eq>c(T)=\frac{1}{N}\frac{d\langle E\rangle}{dT}=\frac{\langle E^2\rangle-\langle E\rangle^2}{Nk_BT^2}</eq>
<p>Si la energía normalizada del archivo se usara sin recuperar E₀, la varianza perdería un factor E₀². Tampoco se puede introducir la energía por espín en esta fórmula sin ajustar N: Var(E)=N² Var(e).</p>
<p>Esta identidad es termodinámica y supone equilibrio. Sobre un conjunto de caminos que no mezcla, la misma fórmula sigue definiendo una curva de fluctuaciones del protocolo. Su interpretación como derivada de equilibrio queda condicionada a la representatividad del conjunto.</p>
<h3>5.3. Fluctuaciones dentro de los caminos y entre ellos</h3>
<eq>\widehat{\mathrm{Var}}_{\mathrm{total}}(X)=\frac{1}{R}\sum_{j=1}^R\widehat{\mathrm{Var}}_t(X_j)+\frac{1}{R}\sum_{j=1}^R(\bar X_j-\bar X)^2</eq>
<p>Con cadenas de igual longitud y varianzas temporales calculadas con divisor n, la identidad es exacta. El primer término recoge fluctuaciones temporales dentro de cada copia. El segundo recoge diferencias entre medias de caminos. El programa oficial agrupa las sumas de momentos; por ello su curva no es simplemente el promedio de las varianzas internas.</p>
<p>La diferencia importa cuando los caminos terminan en sectores distintos. La varianza agrupada incorpora una dispersión entre caminos que depende de las frecuencias producidas por el enfriamiento. Si esas frecuencias no son los pesos de Boltzmann, tampoco está garantizado que la varianza agrupada sea una fluctuación de equilibrio. La revisión añade esa descomposición para la magnetización en todas las temperaturas.</p>
<h3>5.4. Qué susceptibilidad se ha calculado</h3>
<eq>\chi_m(T)=\frac{N}{k_BT}(\langle m^2\rangle-\langle m\rangle^2)</eq>
<p>Esta es la fluctuación de la norma de magnetización empleada en el informe. La respuesta al campo perpendicular tiene otra definición: involucra Mᶻ, la componente conjugada a H. En equilibrio, d⟨Mᶻ⟩/dH = N Var(Mᶻ)/(kBT). Ambas expresiones no deben identificarse sin justificar la simetría y las convenciones. En este documento χ<sub>m</sub> se interpreta como indicador de fluctuación de m.</p>
<h3>5.5. Temperatura característica y repesado</h3>
<eq>T^*=\underset{T\in[6,20]}{\mathrm{arg\,max}}\ c(T)</eq>
<p>En este disco finito, el máximo es redondeado. T* identifica una temperatura característica o pseudocrítica; una singularidad y sus exponentes exigirían estudiar tamaños crecientes. Con campo externo tampoco se puede llamar automáticamente temperatura de Curie a cualquier máximo de c.</p>
<p>El repesado utiliza configuraciones de una temperatura cercana T₀ para estimar momentos a otra T. Al dividir las densidades de Boltzmann, el factor que depende de la configuración es exp[−(β−β₀)E].</p>
<eq>\langle A\rangle_\beta\simeq\frac{\sum_t A_t w_t}{\sum_t w_t},\quad w_t=e^{-(\beta-\beta_0)E_t},\quad n_{\mathrm{Kish}}=\frac{(\sum_t w_t)^2}{\sum_t w_t^2}</eq>
<p>El código asigna cada punto de la malla fina de 0.01 K a la temperatura medida más cercana en β. No usa un ajuste de múltiples histogramas que imponga suavidad entre todas las ranuras. Un salto de ranura puede afectar la ubicación del máximo. La malla más fina permite interpolar estadísticamente, pero no crea mediciones independientes ni una precisión de 0.01 K.</p>
<p>El tamaño de Kish detecta concentración de pesos. El mínimo registrado fue alrededor de 0.65 de la muestra nominal. Ese cociente no descuenta la autocorrelación temporal: no equivale al número efectivo de configuraciones independientes.</p>
<h3>5.6. Autocorrelación y error de Monte Carlo</h3>
<eq>\mathrm{Var}(\bar X)=\frac{\sigma^2}{n}\left[1+2\sum_{k=1}^{n-1}\left(1-\frac{k}{n}\right)\rho(k)\right]\simeq\frac{\sigma^2\tau}{n}</eq>
<eq>\tau=1+2\sum_{k=1}^{W}\rho(k),\quad n_{\mathrm{ef}}=\frac{n}{\tau},\quad \tau_S=\frac{\tau}{2}</eq>
<p>La identidad finita se obtiene al sumar las covarianzas de todos los pares de observaciones de la media. Cuando las correlaciones decaen antes de n, el factor entre corchetes se aproxima por τ. Con τ = 946 y n = 10000, una cadena aporta unas 10.6 observaciones efectivas para esa variable; con τ = 104 aporta unas 96.2. Son aproximaciones locales y pueden ser optimistas si el camino no cambia de sector.</p>
<p>El análisis oficial calcula τ<sub>S</sub> por copia y usa una ventana W ≥ 10τ<sub>S</sub>. El análisis de barridos promedia autocovarianzas entre copias y usa τ con W ≥ 6τ. Difieren tanto la convención como la agregación y la ventana. Por ello los máximos de ambas tablas no se convierten siempre mediante un factor dos exacto. Las cifras 946 y 104 pertenecen al análisis de barridos con τ completo.</p>
<h3>5.7. Jackknife por bloques</h3>
<p>Para estimar el error de una cantidad no lineal, como la varianza, se divide la serie en bloques contiguos. Se recalcula el estimador omitiendo cada bloque y se mide cuánto cambia. Eliminar bloques conserva la dependencia interna y reduce la correlación entre unidades eliminadas.</p>
<eq>\widehat{\mathrm{se}}_{JK}=\sqrt{\frac{g-1}{g}\sum_{j=1}^g(\hat\vartheta_{(j)}-\bar\vartheta_{(\cdot)})^2}</eq>
<p>El programa oficial selecciona b ≥ 10τ<sub>S</sub>, equivalente a b ≥ 5τ bajo la misma estimación de autocorrelación. Exige al menos 20 bloques y admite hasta 500 barridos por bloque. Esas reglas requieren aproximadamente τ ≤ 100. El requisito es operativo; no demuestra independencia exacta entre bloques ni un límite universal del sesgo del 10 %.</p>
<h3>5.8. Bootstrap por caminos e intervalos</h3>
<p>La unidad independiente es el camino completo. Cada réplica del bootstrap selecciona 100 caminos con reemplazo y conserva las mismas multiplicidades en todas las temperaturas. Después recalcula la curva y su máximo. Remuestrear cada temperatura por separado destruiría la dependencia del enfriamiento y cambiaría el problema inferencial.</p>
<eq>I=\lfloor100U\rfloor,\quad U\sim\mathcal{U}(0,1),\quad IC_{\mathrm{percentil}}=[q_{0.025}(T^{*b}),q_{0.975}(T^{*b})]</eq>
<p>El intervalo BCa ajusta los cuantiles por sesgo y asimetría. La corrección z₀ depende de la proporción de máximos bootstrap menores que el máximo observado. La aceleración a se calcula eliminando un camino completo cada vez.</p>
<eq>z_0=\Phi^{-1}[P_b(T^{*b}<\hat T^*)],\quad a=\frac{\sum_j(\bar\theta_{JK}-\theta_{(j)})^3}{6[\sum_j(\bar\theta_{JK}-\theta_{(j)})^2]^{3/2}}</eq>
<eq>\alpha'=\Phi\left[z_0+\frac{z_0+z_\alpha}{1-a(z_0+z_\alpha)}\right]</eq>
<p>El intervalo utiliza los cuantiles bootstrap de niveles α′. Para un argmax discreto, con empates o varios máximos, el ajuste BCa puede ser inestable. El propio código lo marca como orientativo. La distribución completa y el intervalo percentil son necesarios para interpretar el estado 5. Ningún bootstrap corrige un sesgo común a todos los caminos.</p>
''')

sec('diagnosticos','6. Pruebas de convergencia y homogeneidad',r'''
<h3>6.1. Geweke y truncamiento MSER-5</h3>
<p>Geweke compara el primer 10 % de la ventana medida con el último 50 %. Las medias deberían ser compatibles si no hay deriva detectable. El denominador incorpora varianzas espectrales para no tratar los barridos como independientes.</p>
<eq>z=\frac{\bar X_A-\bar X_B}{\sqrt{s_A^2\tau_A/n_A+s_B^2\tau_B/n_B}}</eq>
<p>La implementación original utiliza el tiempo de autocorrelación estimado para la serie completa en ambos segmentos. Con |z| &gt; 1.96 se marca una discrepancia; el exceso de rechazos entre 100 caminos se contrasta mediante una binomial de probabilidad nominal 0.05. Holm corrige las pruebas a través de temperaturas. La aproximación depende de estimar bien la autocorrelación y no prueba estacionariedad ante toda alternativa.</p>
<p>MSER-5 agrupa observaciones en lotes de cinco barridos y busca, dentro de la primera mitad de la ventana, el truncamiento que minimiza una estimación del error de la media restante. El criterio oficial exige que la mediana de los truncamientos no supere 500 barridos. Detecta un transitorio residual visible; no reconstruye los 25000 barridos que ya se descartaron.</p>
<h3>6.2. R-hat: comparar la dispersión interna con la externa</h3>
<eq>W=\frac{1}{R}\sum_{j=1}^R s_j^2,\quad B=\frac{n}{R-1}\sum_{j=1}^R(\bar X_j-\bar X)^2</eq>
<eq>\widehat R=\sqrt{\frac{(n-1)W/n+B/n}{W}}</eq>
<p>W mide variación temporal dentro de caminos. B/n mide dispersión de sus medias. Cuando cada camino recorre la misma distribución y la longitud crece, las medias se acercan y R-hat tiende a uno. Si cada camino permanece en un valle diferente, la dispersión entre medias puede persistir.</p>
<p>El informe utiliza dos versiones con propósitos distintos. Sus figuras de costo usan R-hat clásico. El veredicto oficial divide cada cadena en mitades, transforma los rangos a puntuaciones normales y toma el máximo del diagnóstico ordinario y del plegado alrededor de la mediana. El plegado ayuda a detectar diferencias de escala. Las dos versiones no deben mezclarse en una misma comparación numérica.</p>
<p>El umbral es R-hat &lt; 1.01, siguiendo la práctica citada por Vehtari et al. (2021). No es un nivel de significación ni una probabilidad de convergencia. Dos cadenas pueden coincidir porque ambas quedaron atrapadas en el mismo sector.</p>
<h3>6.3. Homogeneidad entre caminos</h3>
<eq>Q_h=\sum_{j=1}^R\frac{(\bar X_j-\bar X_w)^2}{\widehat{\mathrm{se}}_j^2},\quad \bar X_w=\frac{\sum_j\bar X_j/\widehat{\mathrm{se}}_j^2}{\sum_j1/\widehat{\mathrm{se}}_j^2}</eq>
<p>Si las medias estiman un mismo valor y sus errores están bien calibrados, Q<sub>h</sub> se compara aproximadamente con una χ² de R−1 grados de libertad. El contraste puede detectar diferencias pequeñas con muchas copias. Si el error interno está subestimado o las series no son normales en la escala relevante, la referencia pierde precisión. Holm controla las comparaciones entre temperaturas.</p>
<h3>6.4. Tres criterios operativos diferentes</h3>
'''+table(['Nombre del análisis original','Regla aplicada en el código','Qué significa'],[
['Converge','R-hat de E y m &lt; 1.01; sin exceso de Geweke; mediana MSER ≤ 500.','Pasa los diagnósticos seleccionados.'],
['Representativa','Cumple el criterio anterior y no se rechaza homogeneidad.','Las medias de copias son compatibles bajo sus errores estimados.'],
['Confiable','ESS combinado &gt; 400 y jackknife por bloques admisible.','Suficiencia operativa para el cálculo de error; no exige pasar R-hat.']
],'Definiciones extraídas de informe.py; “confiable” no es una categoría subordinada a “converge”.')+r'''
<p>La última distinción explica por qué el estado 5 figura como confiable en el 100 % de las temperaturas, aunque solo el 55 % cumpla el criterio conjunto de convergencia. La precisión interna y la representatividad global responden preguntas diferentes. Los nombres históricos se conservan en las tablas para permitir la comparación, con su alcance explícito.</p>
''')

TSTAR=[14.35,14.14,14.34,14.15,14.55,14.45,14.35,13.64]
SEP=[.117,.068,.164,.127,.445,.166,.201,.127]
ICP=['[14.00, 14.37]','[14.04, 14.35]','[14.04, 14.65]','[14.04, 14.55]','[13.14, 14.55]','[14.05, 14.65]','[13.87, 14.54]','[13.25, 13.74]']
ICB=['[14.14, 14.45]','[13.95, 14.14]','[13.85, 14.55]','[13.64, 14.15]','[13.55, 14.65]','[14.05, 14.65]','[13.94, 14.55]','[13.14, 13.64]']
CONV=[30,31,38,54,55,30,38,40]
TRAMO=[15.9,15.7,14.7,13.8,13.7,15.8,15.1,14.7]
REP=[29,31,38,46,38,28,36,40]
CONF=[30,31,38,72,100,32,40,41]
TAUE=[52,47,38,30,57,32,32,30]
TAUM=[946,933,643,194,104,665,421,403]
sec('transicion','7. Resultados: transición y diferencias entre estados',r'''
<p>Las ocho curvas de calor específico presentan una región de máximo en torno a 14 K. El máximo es ancho, como corresponde a una muestra finita y al ensamble observado. La estimación fina original sitúa siete estados entre 14.14 y 14.55 K y el estado 8 en 13.64 K.</p>
'''+figure('calor_especifico.pdf','El máximo de fluctuación energética se concentra alrededor de 14 K. La comparación entre estados debe incorporar la incertidumbre del argumento del máximo y las diferencias del diseño.')+table(['Estado','T* (K)','se (K)','IC percentil 95 %','IC BCa 95 %'],[[i,TSTAR[i-1],SEP[i-1],ICP[i-1],ICB[i-1]] for i in range(1,9)],'Estimaciones originales con 50000 réplicas bootstrap por caminos. La auditoría conserva estos resultados; no volvió a ejecutar ese bootstrap.')+r'''
<h3>7.1. Asociación con la interacción quiral</h3>
<p>La revisión reprodujo la prueba exacta de asociación entre k<sub>DM</sub> y T*: ρ = 0.05422 y p = 0.89926. Estos datos no detectan una asociación monotónica en los ocho estados. La conclusión no es que T* sea independiente de k<sub>DM</sub>. Una prueba de correlación no contrasta independencia general y un resultado no significativo no demuestra equivalencia física.</p>
<p>Una afirmación de equivalencia requeriría fijar antes una diferencia térmica científicamente despreciable, aumentar el diseño y comprobar que los intervalos quedan dentro de ese margen. La incertidumbre del estado 5 y el pequeño número de combinaciones impiden una conclusión tan fuerte con el estudio actual.</p>
<h3>7.2. Comparaciones de máximos y multiplicidad</h3>
<p>El análisis original comparó los 28 pares mediante diferencias bootstrap y corrigió los valores p con Holm. Para valores p ordenados, el procedimiento contrasta el primero con 0.05/28, el siguiente con 0.05/27 y continúa hasta el primer contraste no rechazado. Así limita el error familiar asociado a buscar diferencias en muchos pares.</p>
<p>El estado 8 difiere de los estados 1, 2, 3, 4 y 6, con valores p ajustados entre 0.00224 y 0.0365. Las diferencias puntuales son de 0.50 a 0.81 K. Los otros pares no alcanzan el criterio usado. El contraste 7 frente a 8 tiene un intervalo individual que excluye cero, pero p ajustado = 0.168; no hay contradicción, pues ese intervalo individual no es un intervalo simultáneo corregido.</p>
<p>El estado 8 combina H = 0.3 y k<sub>DM</sub> = 0.75. Su menor T* caracteriza esa combinación. El diseño no permite atribuir todo el desplazamiento al campo por separado. El par 4–7 mantiene k<sub>DM</sub> = 0.6 y cambia H de cero a 0.1; su diferencia de T* no es significativa bajo las comparaciones originales.</p>
<h3>7.3. Por qué el estado 5 exige una lectura especial</h3>
'''+figure('T_estrella_bootstrap.png','La distribución bootstrap muestra cuánta estabilidad tiene la ubicación del pico. En el estado 5, máximos alternativos producen una distribución multimodal y un intervalo más ancho.')+r'''
<p>El estado 5 tiene T* = 14.55 K, pero su intervalo percentil llega hasta 13.14 K y su error estándar es 0.445 K. Las réplicas acumulan máximos cerca de 13.55, 14.05, 14.35 y 14.55 K. Este comportamiento impide resumir el resultado como 14.55 ± 0.01 K, aunque la malla fina tenga pasos de 0.01 K.</p>
<p>La meseta original alrededor del máximo seleccionado, [14.55,14.58] K, se define como un tramo contiguo que permanece a menos de un error estándar de la altura máxima. No abarca necesariamente todos los máximos alternativos. Por ello una meseta local estrecha puede coexistir con un intervalo amplio para el argmax.</p>
<h3>7.4. Referencia del cristal infinito</h3>
<p>Chen et al. (1993), citados en el informe, dan kBT<sub>C</sub> ≈ 1.443J para el Heisenberg clásico cúbico de referencia. Con J = 1 resulta T<sub>C</sub> ≈ 16.75 K. El valor cercano a 14.3 K es aproximadamente 15 % menor. Los bordes y el pequeño espesor ofrecen una explicación plausible, pero esta comparación cambia también anisotropía, interacción quiral y campo. No aísla un efecto de tamaño. Para atribuirlo al confinamiento haría falta simular varios radios y espesores con el mismo hamiltoniano.</p>
''')

sec('explorador','8. Curvas comparativas de los ocho estados',r'''
<p>Las curvas de esta sección utilizan las 141 temperaturas originales. La energía y el calor específico de la malla medida se reconstruyen a partir de las sumas de momentos guardadas en el análisis de barridos. La magnetización se recalculó leyendo la matriz completa de cada una de las 1128 combinaciones de estado y temperatura. Los selectores permiten concentrar la comparación sin desplegar una figura por temperatura.</p>
<div class="controls"><label>Variable <select id="metric"><option value="c_recalculado">Calor específico, malla medida</option><option value="energia_meV_por_espin">Energía por espín</option><option value="m_media">Magnetización media</option><option value="chi_protocolo">Fluctuación χm del protocolo</option><option value="fraccion_var_entre">Fracción de varianza de m entre caminos</option><option value="rhat_M">R-hat clásico de m</option><option value="rhat_E">R-hat clásico de energía</option><option value="tau_M">Autocorrelación de m</option><option value="tau_E">Autocorrelación de energía</option></select></label><label>Escala <select id="scale"><option value="linear">Lineal</option><option value="log">Logarítmica</option></select></label><button id="allstates" type="button">Mostrar todos</button><button id="download" type="button">Descargar datos de curvas</button></div>
<div id="legend" class="legend"></div><div id="curve" class="chart" role="img" aria-label="Curvas comparativas de los ocho estados"></div><p id="metric-note" class="note"></p>
<div class="controls"><label>Temperatura para comparar <input id="tempcompare" type="range" min="0" max="140" value="57"><output id="tempvalue"></output></label></div><div id="temp-table"></div>
<noscript><p>Los exploradores requieren JavaScript local. Las figuras, tablas, fórmulas y conclusiones de las demás secciones permanecen disponibles sin JavaScript.</p></noscript>
<h3>8.1. Cómo interpretar las curvas</h3>
<p>La energía media desciende al enfriar y revela el cambio de pendiente asociado a la región de mayor fluctuación. El calor específico enfatiza esa región. El gráfico interactivo usa la malla medida de 0.1 K; las estimaciones T* de la sección anterior provienen de la malla repesada de 0.01 K. Los máximos de ambos gráficos no deben exigirse idénticos.</p>
<p>La magnetización media debe leerse junto a la textura. Una espiral ordenada puede tener m pequeña por cancelación de orientaciones. Con campo, una magnetización mayor puede coexistir con remolinos de carga distinta. La curva no mide por sí sola cuántos skyrmiones hay.</p>
<p>La fracción de varianza entre caminos es la segunda contribución de la ley de varianza total dividida por la suma de ambas. Cuando aumenta, una mayor parte de la dispersión agrupada procede de diferencias persistentes entre historias. Es una descripción del ensamble simulado y ayuda a explicar por qué χ<sub>m</sub> agrupada puede crecer aunque cada camino fluctúe poco.</p>
<p>El error entre caminos de la media de m se calculó como desviación estándar de las 100 medias dividida por √100. La tabla interactiva también muestra los percentiles 10 y 90 de esas medias. El intervalo entre percentiles describe diversidad de caminos; no es un intervalo de confianza del 80 % para la media. El error de la media describe el protocolo bajo independencia de caminos y no elimina un sesgo común.</p>
''')

sec('texturas','9. Enfriamiento y evolución espacial',r'''
<p>El selector muestra la capa central de las primeras tres copias guardadas de cada estado. Se mantienen los mismos índices de copia en todas las temperaturas para seguir tres caminos completos. Hay 16 instantáneas por estado: las temperaturas enteras de 20 a 6 K y 14.3 K cerca del máximo térmico. No se interpolan configuraciones entre temperaturas.</p>
<div class="controls"><label>Temperatura <input id="snaptemp" type="range" min="0" max="15" value="0"><output id="snapvalue"></output></label><label>Camino <select id="copy"><option value="0">Copia 1</option><option value="1">Copia 2</option><option value="2">Copia 3</option></select></label><button id="animate" type="button">Recorrer enfriamiento</button></div>
<p class="small">Componente Sᶻ: azul para −1, blanco para 0 y rojo para +1. Misma escala en todos los estados y temperaturas. Las cifras m son instantáneas de la configuración final, no promedios de los 10000 barridos.</p>
<div id="textures" class="textures"></div>
<h3>9.1. Qué permite afirmar esta visualización</h3>
<p>A temperatura alta se esperan fluctuaciones espaciales intensas; al enfriar pueden aparecer dominios y modulaciones. Aumentar k<sub>DM</sub> acorta la escala de giro de referencia. Los estados con campo pueden presentar patrones compatibles con remolinos. Estas son observaciones de configuraciones almacenadas al final de cada temperatura, no películas de toda la dinámica interna.</p>
<p>Tres copias permiten observar ejemplos de diversidad, pero no estimar sus frecuencias. Para eso se usan las 100 copias de las tablas de carga y las mezclas. La clasificación de un skyrmión tampoco debe hacerse solo por el color Sᶻ: necesita información vectorial y una medida topológica.</p>
'''+figure('factor_estructura.png','El factor de estructura compara las escalas espaciales de las configuraciones finales. Un máximo fuera del origen indica modulación y permite estimar un período, con resolución limitada por el disco y la discretización.')+r'''
<eq>S(\mathbf{k})=\frac{1}{N}\sum_\ell\left|\sum_{i\in\ell}\mathbf{S}_i e^{-i\mathbf{k}\cdot\mathbf{r}_i}\right|^2,\quad \lambda\simeq\frac{2\pi}{k_{\mathrm{pico}}}</eq>
<p>El código calcula Fourier por capa, suma las intensidades de capas y componentes y promedia radialmente. El relleno con ceros facilita evaluar la transformada en más puntos, pero no añade resolución física al tamaño del disco. Los períodos medidos para los estados 3, 4 y 5 fueron 18.1, 10.1 y 7.0 sitios, frente a 19.7, 11.6 y 8.0 de la fórmula ideal. Con k<sub>DM</sub> pequeño, períodos mayores que el diámetro no pueden resolverse bien como vueltas completas dentro del disco.</p>
<eq>Q=\frac{1}{4\pi}\int\mathbf{n}\cdot(\partial_x\mathbf{n}\times\partial_y\mathbf{n})\,dx\,dy</eq>
<p>Q cuantifica el recubrimiento de la esfera por una textura vectorial suave. El código utiliza ángulos sólidos en triángulos de la red, según la formulación citada de Berg y Lüscher (1981), y luego promedia entre capas. Con bordes libres, Q no tiene que ser estrictamente entero. Las cargas modales de la tabla son categorías obtenidas al redondear ese promedio; conviene mantener esta distinción al hablar de sectores.</p>
'''+figure('estados_finales_Q_q.png','Las cargas topológicas y los solapamientos describen heterogeneidad entre configuraciones finales. Un valor modal de carga resume la categoría más frecuente del protocolo, sin establecer su peso de equilibrio.')+table(['Estado','Q modal a 6 K','Copias en la moda','Período medido (sitios)','Lectura útil'],[
[1,0,'100 %',30.2,'Escala de giro mayor que el disco; magnetización lenta.'],[2,0,'100 %',30.2,'Textura extensa y dispersión de magnetización.'],[3,0,'90 %',18.1,'Modulación espacial resoluble.'],[4,0,'92 %',10.1,'Espiral más corta; comparación controlada con estado 7.'],[5,0,'95 %',7.0,'Giro corto; argmax térmico poco estable.'],[6,-1,'78 %',18.1,'Dos poblaciones finales claramente separadas en m.'],[7,-4,'85 %',12.9,'Carga negativa frecuente; mezcla de m más difícil que en estado 4.'],[8,-6,'50 %',10.1,'Diversidad de cargas y tres componentes de m seleccionadas por BIC.']
],'Resultados topológicos y espaciales originales sobre 100 caminos. Los períodos no son predicciones exactas del modelo unidimensional.')+r'''
<p>Las cargas y las poblaciones finales apoyan una interpretación de metastabilidad. Sin embargo, no hay una serie temporal completa de Q dentro de los 10000 barridos para medir tasas de cambio de sector. El número de barridos insuficiente y las diferencias finales son evidencia de mezcla limitada, no una medida directa de la barrera energética ni de la tasa de creación de skyrmiones.</p>
''')

sec('convergencia','10. Resultados de convergencia: todos los estados',
figure('rhat_T.pdf','La energía y la magnetización presentan dificultades de mezcla distintas. El diagnóstico de una variable no sustituye al de la otra; las curvas usan la versión clásica de R-hat.')+
figure('trazas_m.pdf','Las trayectorias frías pueden permanecer estables en niveles diferentes. La estabilidad temporal dentro de una copia puede coexistir con heterogeneidad entre caminos.')+
table(['Estado','Converge (%)','Tramo desde 20 K hasta','Representativa (%)','Error confiable (%)','τE máx.','τm máx.'],[[i,CONV[i-1],f'{TRAMO[i-1]} K',REP[i-1],CONF[i-1],TAUE[i-1],TAUM[i-1]] for i in range(1,9)],'Veredictos originales. Los tiempos máximos τE y τm pertenecen al análisis de barridos y no son la mediana de τS de la tabla oficial de errores.')+r'''
<p>Ningún estado pasa el criterio conjunto en las 141 temperaturas. El porcentaje va del 30 al 55 %. Los estados 4 y 5 extienden el tramo favorable por debajo de sus temperaturas T*. En los demás, el criterio conjunto se pierde antes de alcanzar el máximo energético. Esto limita una afirmación global de equilibrio, aunque la energía pase sus diagnósticos cerca del pico.</p>
<p>La nueva auditoría añade el diagnóstico de la fluctuación energética cuadrática. En cada una de siete temperaturas por estado se calculó R-hat dividido, normalizado por rangos y plegado para E, (E−Ē)² y m. Las siete temperaturas incluyen 6, 10 y 20 K y cuatro puntos de la malla alrededor de T*. Son 56 puntos comprobados, no una auditoría adicional de los momentos en toda la escalera.</p>
'''+table(['Estado','Máx. R-hat de (E−Ē)² cerca de T*','Máx. R-hat de m cerca de T*','R-hat de (E−Ē)² a 6 K'],[[e,f"{max(r['rhat_E2_centrado'] for r in A['momentos'] if r['estado']==e and 13<r['T']<15):.6f}",f"{max(r['rhat_m'] for r in A['momentos'] if r['estado']==e and 13<r['T']<15):.6f}",f"{next(r['rhat_E2_centrado'] for r in A['momentos'] if r['estado']==e and r['T']==6):.6f}"] for e in range(1,9)],'Comprobación nueva sobre los datos crudos. El máximo cercano al pico se toma sobre cuatro temperaturas medidas, distintas para cada estado.')+r'''
<p>En los 32 puntos cercanos al pico, el diagnóstico del segundo momento centrado es menor que 1.002. La magnetización supera 1.01 en varios estados. Esta evidencia refuerza la estabilidad entre caminos de las cantidades energéticas que entran en c(T), y permite fundamentar mejor su uso que con la energía media sola.</p>
<p>El diagnóstico se centra en un mismo Ē global por temperatura. Así conserva las diferencias entre caminos; centrar cada copia en su propia media podría ocultar parte de ellas. Todavía puede existir un sector no visitado por ninguno de los 100 caminos. La conclusión es evidencia favorable para los momentos energéticos observados, sin garantía matemática de representatividad global.</p>
'''+figure('medias_copias.pdf','La dispersión de las medias entre caminos ayuda a distinguir un error de Monte Carlo interno pequeño de una diferencia persistente entre historias de enfriamiento.'))

sec('costo','11. Costo en barridos y relaciones estadísticas',r'''
<h3>11.1. Qué se puede extrapolar desde R-hat(n)</h3>
<p>El análisis original recalculó R-hat con 25 prefijos de entre 100 y 10000 barridos. Si el exceso sobre uno sigue aproximadamente una potencia a·nᵇ, una pendiente b cercana a −1 es compatible con disminución de la varianza de las medias al alargar la observación.</p>
<eq>\log(\widehat R(n)-1)=\log a+b\log n,\quad n_{\mathrm{objetivo}}=\left(\frac{0.01}{a}\right)^{1/b}</eq>
<p>La fórmula supone que el régimen ajustado continúa hasta el objetivo. Cerca de T*, las pendientes originales fueron de −0.88 a −1.08 y los cálculos orientan hacia 10000 a 34000 barridos. A 6 K, algunos ajustes se aplanan. Extrapolar desde 10000 hasta millones o 10¹³ barridos es demasiado lejano para interpretarlo como una predicción validada.</p>
'''+figure('rhat_n.pdf','El descenso de R-hat con la longitud observada separa una mezcla que mejora de un diagnóstico casi estancado. La extrapolación pierde fiabilidad cuando la pendiente se aproxima a cero.')+figure('barridos_necesarios.pdf','El presupuesto de medición alcanza el umbral solo en parte de la escalera. Las regiones donde no lo alcanza requieren distinguir mezcla lenta de atrapamiento antes de invertir en cadenas más largas.')+r'''
<p>El resultado útil no es un número astronómico de barridos. Es identificar en qué estados y temperaturas el movimiento local deja de mejorar al ritmo esperado. Una prueba futura debe ejecutar longitudes mayores previamente fijadas o comparar un algoritmo con intercambio de réplicas. La propuesta de intercambio no se ejecutó en esta revisión.</p>
<h3>11.2. Correlaciones entre parámetros y dificultad de muestreo</h3>
'''+figure('correlacion_kdm.pdf','En estas combinaciones, una interacción quiral mayor se asocia con menor autocorrelación de m y con más temperaturas que pasan R-hat. El diseño permite describir esa asociación, con limitaciones para aislar el campo.')+r'''
<p>El análisis original obtuvo ρ = −0.96 y p = 0.001 entre k<sub>DM</sub> y el máximo de τ de m. La asociación con la fracción de temperaturas que pasan el diagnóstico clásico fue ρ = 0.92 y p = 0.003. Los intervalos bootstrap fueron amplios, porque la unidad de comparación son ocho estados.</p>
<p>La relación tiene una explicación física plausible: las modulaciones cortas se compensan espacialmente y la magnetización total puede cambiar con reorganizaciones locales. Las texturas extensas pueden exigir reorganizaciones colectivas. Esta explicación no supone que una rotación rígida cambie m. Tampoco establece que el tiempo de autocorrelación observado mida toda la dificultad de cambiar de sector.</p>
<h3>11.3. Regresión reproducida y dependencia entre temperaturas</h3>
<eq>\log\tau_m=3.029-2.365k_{DM}+2.749H+30.610/T</eq>
<p>La regresión resume 1128 pares estado–temperatura y reproduce los coeficientes originales. El ajuste usa el logaritmo para describir efectos multiplicativos. Un incremento de 0.1 en k<sub>DM</sub> multiplica el tiempo ajustado por exp(−0.2365) ≈ 0.789. Un incremento de 0.1 en H lo multiplica por exp(0.2749) ≈ 1.316. Estas son asociaciones del modelo ajustado, no efectos causales identificados.</p>
<p>Los errores HC0 toleran heterocedasticidad, pero no corrigen la relación entre las 141 temperaturas de un estado. La revisión añade una covarianza agrupada por estado, con corrección CR1. Cada grupo aporta el producto externo de su suma de puntuaciones.</p>
<eq>\widehat V_{CR1}=\frac{G}{G-1}\frac{n-1}{n-p}(X^TX)^{-1}\left[\sum_{g=1}^G X_g^Tu_g u_g^T X_g\right](X^TX)^{-1}</eq>
'''+table(['Coeficiente','Estimación','se HC0','se agrupado CR1','Rango al retirar un estado'],[[label,f"{A['regresion']['coef'][i]:.4f}",f"{A['regresion']['se_HC0'][i]:.4f}",f"{A['regresion']['se_cluster_CR1'][i]:.4f}",f"[{min(row[i] for row in A['regresion']['leave_one_state_out']):.4f}, {max(row[i] for row in A['regresion']['leave_one_state_out']):.4f}]"] for i,label in enumerate(['Intercepto','kDM','H','1/T'])],'Nueva evaluación de sensibilidad. Hay ocho grupos; CR1 no convierte este diseño pequeño en una inferencia de gran muestra.')+r'''
<p>El error del coeficiente de H sube de 0.217 a 0.537 y el de 1/T de 0.792 a 2.086. Al retirar el estado 8, el coeficiente del campo pasa a 6.154. La dirección del efecto ajustado se mantiene, pero su magnitud depende de una combinación influyente. La asociación con k<sub>DM</sub> conserva el signo al retirar cualquiera de los ocho estados.</p>
<p>Con solo ocho grupos, los errores agrupados también son aproximados. La revisión no añade valores p asintóticos para presentar certeza artificial. La evidencia útil es la sensibilidad de coeficientes y errores, junto al par controlado 4–7: con el mismo k<sub>DM</sub> = 0.6, el máximo observado de τm sube de 194 a 421 barridos al pasar de H = 0 a H = 0.1.</p>
''')

sec('em','12. Poblaciones finales: mezclas gaussianas y EM',r'''
<p>Para estudiar la diversidad final, cada camino aporta su media de magnetización a 6 K. Una mezcla gaussiana describe la distribución de esas 100 medias como combinación de poblaciones. Sus componentes son aproximaciones estadísticas; no son fases termodinámicas demostradas.</p>
<eq>f(x_j)=\sum_{c=1}^k w_c\phi(x_j;\mu_c,\sigma_c^2),\quad \sum_cw_c=1</eq>
<p>EM alterna dos operaciones. El paso E calcula la probabilidad de pertenencia de cada media a cada componente con los parámetros actuales. El paso M usa esas responsabilidades como pesos para actualizar los parámetros.</p>
<eq>r_{jc}=\frac{w_c\phi(x_j;\mu_c,\sigma_c^2)}{\sum_\ell w_\ell\phi(x_j;\mu_\ell,\sigma_\ell^2)},\quad N_c=\sum_j r_{jc}</eq>
<eq>w_c=\frac{N_c}{R},\quad \mu_c=\frac{\sum_jr_{jc}x_j}{N_c},\quad \sigma_c^2=\frac{\sum_jr_{jc}(x_j-\mu_c)^2}{N_c}</eq>
<p>El algoritmo aumenta o mantiene la verosimilitud, pero puede llegar a máximos locales. El estudio probó 20 inicializaciones para cada k entre 1 y 3 y conservó la mejor. El BIC penaliza el número de parámetros: k medias, k varianzas y k−1 pesos suman 3k−1.</p>
<eq>\mathrm{BIC}=-2\ell+(3k-1)\log100</eq>
'''+figure('mezcla_em.pdf','En el estado 6 las medias finales se separan en dos poblaciones. El ascenso de la log-verosimilitud verifica el comportamiento numérico de EM, sin demostrar unicidad del ajuste.')+table(['Estado','k elegido','Medias y pesos aproximados','Interpretación'],[
[1,2,'0.775 (72 %); 0.714 (28 %)','Heterogeneidad de caminos en una textura extensa.'],[2,1,'0.300 (100 %)','Una normal basta entre las opciones probadas.'],[3,1,'0.065 (100 %)','Una componente estadística no implica mezcla temporal completa.'],[4,1,'0.027 (100 %)','Magnetización pequeña compatible con cancelación espacial.'],[5,2,'0.015 (60 %); 0.017 (40 %)','Separación pequeña: no atribuir automáticamente dos valles físicos.'],[6,2,'0.580 (78 %); 0.702 (22 %)','Dos poblaciones bien separadas en el protocolo.'],[7,2,'0.389 (82 %); 0.372 (18 %)','Diferencias de media y dispersión; falta clasificación conjunta.'],[8,3,'0.577 (50 %); 0.563 (39 %); 0.595 (11 %)','Diversidad final con apoyo de la estructura topológica.']
],'Selección original por BIC sobre medias de 100 caminos a 6 K.')+r'''
<p>El caso 6 es especialmente útil: las poblaciones están separadas por aproximadamente 0.12 en m, con desviaciones internas cercanas a 0.007. El grupo dominante del análisis espacial contiene también el 78 % de las copias. La coincidencia es compatible con una misma partición, pero verificarlo exige cruzar las etiquetas camino por camino. La igualdad de proporciones por sí sola no demuestra que sean los mismos individuos.</p>
<p>Los pesos 78/22 o 50/39/11 describen la probabilidad de llegada bajo el protocolo simulado. No son pesos de Boltzmann estimados mediante transiciones entre sectores. Además, con 100 caminos y distribuciones acotadas, la selección entre una, dos o tres gaussianas depende de la familia elegida y de los errores de las medias. El BIC selecciona una aproximación útil entre las candidatas, no el número absoluto de estados físicos posibles.</p>
''')

sec('sintesis','13. Conclusiones para el estudio de los estados',r'''
<p>El enfriamiento presenta una escala térmica energética reproducible alrededor de 14 K. Los máximos originales y sus intervalos deben conservarse como estimaciones de una temperatura característica finita. La auditoría de energía y fluctuación cuadrática cerca del pico aporta evidencia favorable para su estabilidad entre caminos. No permite afirmar una transición singular ni equilibrio global demostrado.</p>
<p>La interacción quiral muestra una asociación clara con el costo observado de decorrelacionar la magnetización en las ocho combinaciones. La asociación conserva el signo al retirar un estado. Su vínculo con la escala espacial de las texturas ofrece una interpretación física que merece una prueba factorial. En cambio, la ausencia de asociación significativa entre k<sub>DM</sub> y T* no establece independencia.</p>
<p>El estado 8 tiene un máximo térmico inferior al de varias combinaciones. El resultado distingue ese estado y orienta una nueva exploración del campo. La magnitud del coeficiente de H cambia al excluirlo, de modo que el diseño actual no separa con suficiente estabilidad el campo y la interacción quiral.</p>
<p>La falta de mezcla de m es un resultado científico del protocolo. Señala memoria del camino y reorganizaciones difíciles con actualizaciones locales. La descomposición de varianza permite distinguir fluctuaciones temporales de diferencias persistentes entre historias. Las texturas, las cargas y las mezclas describen qué diversidad se conserva al final, aun cuando no se pueda asignar a todas las poblaciones una frecuencia de equilibrio.</p>
'''+table(['Estado','Resultado aprovechable','Límite principal','Siguiente contraste útil'],[
[1,'Magnetización grande y memoria larga.','Textura extensa y mezcla lenta.','Variar duración e inicialización; comparar movimientos colectivos.'],[2,'Distribución final amplia descrita por una normal.','Una componente no prueba equilibrio.','Comparar orientación relativa y cancelación entre dominios.'],[3,'Período espacial resoluble y costo intermedio.','Criterio conjunto falla cerca de parte del pico.','Aumentar longitud alrededor del máximo y medir Q en el tiempo.'],[4,'Buena referencia sin campo para kDM = 0.6.','Región fría aún puede fallar.','Comparación emparejada con estado 7.'],[5,'Decorrelación de m relativamente rápida.','Máximo térmico multimodal y problemas de representatividad.','Repetir rejilla alrededor de máximos alternativos y comparar repesados.'],[6,'Dos poblaciones finales nítidas.','Pesos del protocolo sin tasas de transición.','Cruzar EM con Q por copia y medir cambios de sector.'],[7,'Carga negativa frecuente y costo mayor que estado 4.','Efecto del campo solo parcialmente aislado.','Diseño con varios H a kDM = 0.6.'],[8,'Menor T* y diversidad de poblaciones finales.','Estado influyente y mezcla limitada.','Añadir H cercanos a 0.3 con el mismo kDM.']
],'Qué aportan los datos de cada estado y qué estudio resuelve su principal incertidumbre.')+r'''
<h3>13.1. Qué corregir en la interpretación</h3>
<p>“No se detectó asociación” sustituye a “no depende”. “Temperatura característica del disco” sustituye al uso indiscriminado de “temperatura de Curie”. Una energía estable no garantiza por sí sola su varianza, y un error interno pequeño no garantiza representatividad. La comparación con el cristal infinito no atribuye de forma exclusiva la diferencia al tamaño.</p>
<p>La coexistencia de poblaciones finales tampoco debe calificarse como “legítima” en el sentido de equilibrio. Los caminos son realizaciones válidas del algoritmo y del protocolo. Sus estados finales pueden conservar historia; eso es precisamente lo que se estudia. El equilibrio exige una evidencia adicional sobre exploración y pesos relativos.</p>
<h3>13.2. Pruebas futuras que sí responderían preguntas nuevas</h3>
<p>Un diseño factorial de k<sub>DM</sub> y H permitiría separar sus asociaciones. Variar radio y espesor con el resto del hamiltoniano fijo permitiría evaluar confinamiento. Repetir enfriamientos a distintas duraciones e incluir calentamiento permitiría medir dependencia del protocolo e histéresis, sin llamar tiempo físico a los barridos.</p>
<p>Para evaluar mezcla, conviene guardar Q y las componentes de magnetización durante la medición, registrar la termalización y comparar arranques aleatorios con configuraciones ordenadas. Un ensayo de intercambio de réplicas debe comprobar viajes de ida y vuelta en temperatura y cambios de sector, además de aceptación de intercambios. Ninguna de estas nuevas simulaciones se presenta aquí como ejecutada.</p>
''')

# Tablas originales completas, sin incluir la base de 25 GB.
rawtables=RESULTADOS/'tablas'
def md_table(p):
    lines=[l.strip() for l in p.read_text().splitlines() if l.strip().startswith('|')]
    cells=[[v.strip().replace(r'\|','|') for v in re.split(r'(?<!\\)\|',l.strip('|'))] for l in lines]
    if len(cells)<2:return ''
    rows=[c for c in cells[1:] if not all(re.fullmatch(r'[-: ]+',x or '-') for x in c)]
    return table([html.escape(c) for c in cells[0]],[[html.escape(v) for v in row] for row in rows],p.name)
sec('pruebas','14. Registro completo de pruebas y datos derivados',r'''
<p>La base cruda queda fuera de esta guía y de la publicación prevista. Se conservan todos los puntos derivados que sustentan los exploradores y la nueva auditoría, junto a las tablas originales. La extensión de la explicación no depende de incorporar los 25 GB de matrices temporales. Los archivos descargables contienen resultados derivados, no sustituyen la base para recalcular cualquier estadístico.</p>
<div class="controls"><button id="audit-download" type="button">Descargar auditoría completa</button><button id="moments-download" type="button">Descargar prueba de momentos en CSV</button></div>
'''+table(['Prueba','Procedencia','Unidad y alcance','Resultado o archivo'],[
['Uniformidad esférica','Original','100 semillas; 4 millones de propuestas por semilla.','Sin desviación detectada en los contrastes documentados.'],
['Jackknife por bloques','Original','Ventanas medidas, energía y magnetización.','Validez operativa limitada por autocorrelación y número de bloques.'],
['Bootstrap y BCa','Original','50000 réplicas de 100 caminos; misma multiplicidad en toda la escalera.','T*, intervalos y 28 comparaciones con Holm.'],
['Geweke, MSER, homogeneidad, R-hat por rangos','Original','8 × 141 temperaturas, 100 caminos.','Veredictos y tablas completas más abajo.'],
['R-hat frente a longitud','Original','25 prefijos de medición por temperatura.','Datos conservados en el descargable de auditoría.'],
['EM y BIC','Original','100 medias a 6 K; 3 tamaños, 20 arranques por estado.','Parámetros y BIC del informe final.'],
['R-hat de E y (E−Ē)²','Nueva ejecución','56 puntos; versiones dividida, por rangos y plegada.','Tabla por punto y JSON descargable.'],
['Sensibilidad de regresión','Nueva ejecución','1128 observaciones en 8 grupos.','HC0, CR1 y 8 ajustes retirando un estado.'],
['Spearman exacto kDM–T*','Nueva ejecución','40320 permutaciones de los ocho estados.','ρ = 0.0542169; p = 0.899256.'],
['Curvas y descomposición de m','Nueva ejecución','1128 archivos; 10000 × 100 mediciones cada uno.','Media, error por caminos, percentiles y varianza entre caminos.'],
['Texturas','Nueva extracción','16 temperaturas × 8 estados × 3 copias.','Capa central e índices de copia fijos.']
],'Procedencia de cada resultado. “Original” significa documentado en el main y sus programas, sin nueva ejecución de esa prueba en esta revisión.')+
'<details><summary>Los 56 puntos de la nueva prueba de momentos</summary>'+table(['Estado','T (K)','R-hat E','R-hat fluctuación cuadrática','R-hat m','c malla medida'],[[r['estado'],r['T'],f"{r['rhat_E']:.6f}",f"{r['rhat_E2_centrado']:.6f}",f"{r['rhat_m']:.6f}",f"{r['c']:.6f}"] for r in A['momentos']])+'</details>'+
''.join('<details><summary>Tabla original completa: '+html.escape(p.stem)+'</summary>'+md_table(p)+'</details>' for p in sorted(rawtables.glob('*.md')))+r'''
<p>Los archivos de la revisión son <code>revisar_datos.py</code>, <code>curvas_magnetizacion.py</code>, <code>auditoria.json</code> y <code>magnetizacion.json</code>. Sus registros de ejecución están junto a ellos. Los datos originales se leen desde la ubicación externa configurada con NANODISCO_DATOS; no se suben aquí. La conexión futura a Kaggle debe preservar parámetros, semillas, dimensiones y correspondencia entre estado, temperatura y camino.</p>
<p>El informe original utiliza energía E y magnetización m como series temporales y conserva configuraciones finales. No se dispone en esta guía de una trayectoria completa de espines ni de la serie de Q a cada barrido. El resultado disponible no permite medir directamente tasas de transición topológica o barreras de energía libre.</p>
''')

sec('referencias','15. Referencias y trazabilidad',r'''
<p>La bibliografía que fundamenta esta explicación procede de <code>refs.bib</code> del informe final. Las demostraciones algebraicas se desarrollan aquí; los resultados numéricos proceden de las tablas originales o de los programas de revisión identificados en la sección anterior.</p>
<ul>
<li>Metropolis et al. (1953). <em>Equation of state calculations by fast computing machines.</em> Journal of Chemical Physics, 21, 1087–1092. Fundamento del muestreo de configuraciones.</li>
<li>Hastings (1970). <em>Monte Carlo sampling methods using Markov chains and their applications.</em> Biometrika, 57, 97–109. Razón de aceptación para propuestas generales.</li>
<li>Dzyaloshinskii (1958), Journal of Physics and Chemistry of Solids, 4, 241–255; Moriya (1960), Physical Review, 120, 91–98. Interacción antisimétrica entre espines.</li>
<li>Nagaosa y Tokura (2013). <em>Topological properties and dynamics of magnetic skyrmions.</em> Nature Nanotechnology, 8, 899–911. Interpretación de texturas topológicas.</li>
<li>Berg y Lüscher (1981). Nuclear Physics B, 190, 412–424. Número topológico en una red mediante ángulos sólidos.</li>
<li>Chen, Ferrenberg y Landau (1993). Physical Review B, 48, 3249–3256. Modelo Heisenberg cúbico de referencia.</li>
<li>Ferrenberg y Swendsen (1988). <em>New Monte Carlo technique for studying phase transitions.</em> Physical Review Letters, 61, 2635–2638. Repesado de histogramas.</li>
<li>Sokal (1997). <em>Monte Carlo methods in statistical mechanics: foundations and new algorithms.</em> Functional Integration: Basics and Applications, 131–192; Wolff (2004), Computer Physics Communications, 156, 143–153. Autocorrelación y error Monte Carlo.</li>
<li>Efron y Tibshirani (1993). <em>An Introduction to the Bootstrap.</em> Chapman &amp; Hall. Remuestreo e intervalos.</li>
<li>Gelman y Rubin (1992). Statistical Science, 7, 457–472; Vehtari et al. (2021), Bayesian Analysis, 16, 667–718. Diagnósticos entre cadenas y normalización por rangos.</li>
<li>Geweke (1992). Bayesian Statistics 4, 169–193; White (1997), Simulation, 69, 323–334. Comparación de segmentos y truncamiento inicial.</li>
<li>Dempster, Laird y Rubin (1977). Journal of the Royal Statistical Society B, 39, 1–38; Schwarz (1978), Annals of Statistics, 6, 461–464. EM y selección mediante BIC.</li>
<li>Holm (1979). Scandinavian Journal of Statistics, 6, 65–70. Corrección de pruebas múltiples.</li>
</ul>
<p>El estudio de nanopartículas de cobre de Rincón Gutiérrez et al. (2026), citado en el main, ofrece un antecedente sobre enfriamientos y diversidad de caminos. Su potencial y su sistema físico son distintos; no valida directamente el hamiltoniano ni la temperatura del nanodisco.</p>
<p>La revisión matemática distingue definiciones, resultados observados e interpretaciones. La fuente principal sigue siendo el <code>main.tex</code> de la entrega final.</p>
''')


# La guía de ejecución se mantiene en Markdown y se incorpora íntegra a la página.
def inline_md(value):
    value=html.escape(value)
    value=re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    value=re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<a href="'+m.group(2).replace('../datos/README.md','datos/README.md').replace('REPRODUCIBILIDAD.md','docs/REPRODUCIBILIDAD.md')+'">'+m.group(1)+'</a>', value)
    return value

def markdown_guide(path):
    lines=path.read_text().splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
        if line.startswith('# '):i+=1;continue
        if line.startswith('## '):out.append('<h3>'+inline_md(line[3:])+'</h3>');i+=1;continue
        if line.startswith('|'):
            group=[]
            while i<len(lines) and lines[i].startswith('|'):group.append(lines[i]);i+=1
            cells=[[inline_md(v.strip()) for v in line.strip('|').split('|')] for line in group]
            out.append(table(cells[0],cells[2:]));continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):para.append(lines[i]);i+=1
        out.append('<p>'+inline_md(' '.join(para))+'</p>')
    return '\n'.join(out)

sec('kaggle','Ejecución en Kaggle con dos GPU',markdown_guide(ROOT/'docs/GUIA_KAGGLE.md'))
report_view='''
<p>El informe final puede leerse completo en este visor. La explicación matemática y los gráficos interactivos continúan debajo, dentro de la misma página.</p>
<p class="mainlinks"><a href="informe_final.pdf" target="_blank" rel="noopener">Abrir el PDF a pantalla completa</a> <a href="informe_final.pdf" download>Descargar informe</a> <a href="presentacion/presentacion.pdf">Diapositivas de resultados</a> <a href="#problema">Leer la explicación matemática</a> <a href="#kaggle">Simular en Kaggle</a></p>
<iframe src="informe_final.pdf#view=FitH" title="Informe final completo del estudio del nanodisco" loading="lazy"></iframe>
<p class="small">Si el navegador no muestra el visor integrado, el enlace de apertura o descarga permite acceder al mismo informe completo.</p>
'''
SECTIONS.insert(0,('informe-completo','Informe final completo',report_view))

# Una vista estática complementa el explorador para impresión y lectores sin JS.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8))
colors=['#0066b3','#cb4b16','#198754','#8b47a8','#bc2f50','#097c88','#856300','#303d52']
for s,c in zip(A['curvas'],colors):
    label=f"E{s['estado']} (DM={s['kDM']}, H={s['H']})"
    rows=[r for r in M if r['estado']==s['estado']]
    axes[0,0].plot(s['T'],s['c_recalculado'],color=c,label=label)
    axes[0,1].plot([r['T'] for r in rows],[r['m_media'] for r in rows],color=c)
    axes[1,0].plot(s['T'],s['rhat_M'],color=c)
    axes[1,1].plot([r['T'] for r in rows],[r['fraccion_var_entre'] for r in rows],color=c)
for ax,title,ylabel in zip(axes.flat,['Fluctuación energética','Magnetización del protocolo','Mezcla entre caminos','Heterogeneidad de caminos'],['c por espín','m media','R-hat clásico de m','Fracción de varianza entre caminos']):
    ax.set(title=title,xlabel='Temperatura (K)',ylabel=ylabel,xlim=(20,6));ax.grid(alpha=.2)
axes[1,0].axhline(1.01,color='black',linestyle=':')
axes[0,0].legend(fontsize=8,ncol=2)
fig.tight_layout()
static=BASE/'fig/comparacion_ocho_estados.png';fig.savefig(static,dpi=170);plt.close(fig)
static_tag='<figure><img class="original" alt="Comparación de calor específico, magnetización, R-hat y heterogeneidad en los ocho estados" src="data:image/png;base64,'+base64.b64encode(static.read_bytes()).decode()+'"><figcaption>Comparación de los ocho estados a lo largo de las 141 temperaturas. Nuevas curvas de magnetización y fracción de varianza entre caminos, junto al calor específico reconstruido de la malla y el diagnóstico clásico original.</figcaption></figure>'
SECTIONS=[(i,t,static_tag+b if i=='explorador' else b) for i,t,b in SECTIONS]

# Únicamente datos derivados; no se incrusta la base de series temporales.
DATA=json.dumps(dict(auditoria=A,magnetizacion=M,Tstar=TSTAR),ensure_ascii=False,allow_nan=False,separators=(',',':'))
CSS='''
:root{--ink:#162d43;--blue:#0066b3;--line:#cfdae4;--bg:#f1f5f9;--muted:#526779}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:17px/1.75 Georgia,'Times New Roman',serif}header{background:#063559;color:white;padding:3rem max(5vw,1.5rem);border-bottom:6px solid #1595d2}header p{max-width:900px}h1{font:600 clamp(1.7rem,3.4vw,2.7rem)/1.25 system-ui;max-width:1100px}h2{font:650 1.5rem/1.4 system-ui;color:var(--blue);margin:0 0 1.5rem}h3{font:600 1.14rem/1.5 system-ui;margin:2rem 0 .75rem}p{margin:.8rem 0 1.1rem}main{display:grid;grid-template-columns:260px minmax(0,1060px);gap:2rem;max-width:1400px;margin:2rem auto;padding:0 1.5rem}nav{position:sticky;top:1rem;align-self:start;max-height:95vh;overflow:auto;font:14px/1.5 system-ui}nav a{display:block;color:#274966;text-decoration:none;padding:.55rem .7rem;border-left:3px solid transparent}nav a:hover{background:white;border-color:var(--blue)}section{background:white;padding:2.7rem clamp(1.2rem,4vw,3.5rem);margin-bottom:1.7rem;border:1px solid var(--line);border-radius:4px;scroll-margin-top:1rem;min-width:0}article{min-width:0}.note{background:#edf5fc;border-left:4px solid var(--blue);padding:1rem 1.2rem;margin:1.5rem 0}.equation{overflow-x:auto;background:#fafcfe;padding:1.1rem;margin:1.3rem 0;text-align:center}.equation img{max-width:none;height:auto;vertical-align:middle}.table-wrap{overflow:auto;margin:1.6rem 0}table{border-collapse:collapse;min-width:100%;font:14px/1.5 system-ui}caption{text-align:left;color:var(--muted);padding:0 0 .7rem;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:.65rem .7rem;text-align:left;vertical-align:top}th{background:#e8f0f8;color:#17436a;font-weight:600}tr:nth-child(even){background:#f8fafc}figure{margin:2rem 0}figure img{display:block;width:100%;height:auto;border:1px solid #dde5eb;cursor:zoom-in}figcaption{font:14px/1.6 system-ui;margin:.6rem 0;color:#38566e}.source{display:block;color:#667d8f;font-size:13px}details{margin:1rem 0;border:1px solid var(--line);padding:.9rem}summary{font:600 15px/1.6 system-ui;cursor:pointer}code{font-size:.85em;background:#eff3f6;padding:.15em .3em;overflow-wrap:anywhere}button,select,input{font:15px system-ui}button,select{padding:.6rem .8rem;border:1px solid #afc1d1;border-radius:4px;background:white;color:#123955}button{cursor:pointer}button:hover{background:#e4f0fa}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #df9a23;outline-offset:2px}.controls{display:flex;flex-wrap:wrap;gap:.8rem;align-items:end;margin:1rem 0;font:15px/1.5 system-ui}.controls label{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}.legend{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0;font:14px/1.4 system-ui}.legend label{border:1px solid var(--line);padding:.4rem;cursor:pointer}.chart svg{width:100%;height:auto;background:#fafcfe}.textures{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem}.texture{padding:.7rem;background:#f4f7fa;border:1px solid var(--line);font:14px/1.5 system-ui}.texture canvas{width:100%;height:auto;image-rendering:pixelated}.small{font:14px/1.6 system-ui;color:var(--muted)}footer{padding:2rem;text-align:center;font:14px system-ui}dialog{max-width:96vw;max-height:96vh;border:1px solid #9ab1c4;padding:1rem;background:white}dialog img{max-width:90vw;max-height:83vh;object-fit:contain}dialog::backdrop{background:#11263bcc}#close-modal{display:block;margin:0 0 1rem auto}.print-only{display:none}@media(max-width:1000px){main{display:block}nav{position:static;max-height:none;display:grid;grid-template-columns:repeat(2,1fr);margin-bottom:1.5rem}.textures{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){body{font-size:16px}main{padding:0 .6rem}nav{display:block}section{padding:1.4rem 1rem}.controls{display:block}.controls label{margin:.7rem 0}.textures{grid-template-columns:repeat(2,minmax(0,1fr))}.equation{text-align:left}}@media print{body{background:white;font-size:11pt}header{background:white;color:black;padding:1rem;border-bottom:1px solid black}main{display:block;margin:0;padding:0}nav,.controls,.legend,#curve,#textures,#temp-table,dialog,.chart,.small{display:none}section{border:0;padding:1rem 0;margin:0;break-before:auto}h2,h3{break-after:avoid}figure,table,.equation{break-inside:avoid}details{display:block}details>*{display:block}figure img{max-height:70vh;object-fit:contain}.equation img{max-width:100%}h2{font-size:15pt}h3{font-size:12pt}}
'''
JS=r'''
const D=JSON.parse(document.getElementById('data').textContent), A=D.auditoria, colors=['#0066b3','#cb4b16','#198754','#8b47a8','#bc2f50','#097c88','#856300','#303d52'];
let active=new Set([1,2,3,4,5,6,7,8]);
const extra=new Map(D.magnetizacion.map(r=>[r.estado+':'+r.T.toFixed(1),r]));
const metrics={c_recalculado:['c por espín','Curva reconstruida en las 141 temperaturas medidas a partir de la varianza agrupada de energía. Las estimaciones T* oficiales usan repesado: no se recalculan con este selector.'],energia_meV_por_espin:['E/N (por espín)','Energía recuperada mediante E₀ y dividida por N. Curvas del ensamble observado.'],m_media:['m media','Media del protocolo sobre 100 caminos. La tabla añade error entre caminos y percentiles 10 y 90 de sus medias.'],chi_protocolo:['χm del protocolo','N Var(m)/(kBT) agrupada. Puede contener diferencias persistentes entre caminos. No equivale por definición a la respuesta d⟨Mz⟩/dH.'],fraccion_var_entre:['Fracción de varianza entre caminos','Varianza de las medias de caminos dividida por la varianza total. Resume heterogeneidad del protocolo, no una fracción de fases.'],rhat_M:['R-hat clásico de m','Diagnóstico clásico original; línea de referencia 1.01. Los veredictos oficiales usan una versión por rangos, dividida y plegada.'],rhat_E:['R-hat clásico de E','Diagnóstico clásico de energía. Un valor próximo a 1 no prueba ausencia de sectores no visitados.'],tau_M:['τm (barridos)','Convención completa 1+2Σρ. Autocovarianzas agrupadas entre caminos y ventana W≥6τ. No mezclar con τS por copia.'],tau_E:['τE (barridos)','Autocorrelación completa de energía en el análisis de barridos. Describe información temporal de la variable medida.']};
function val(s,key,i){return s[key] ? s[key][i] : extra.get(s.estado+':'+s.T[i].toFixed(1))[key]}
const fmt=x=>Number.isFinite(x)?x.toLocaleString('es-CO',{maximumSignificantDigits:5}):'No disponible';
function draw(){
 const key=document.getElementById('metric').value,log=document.getElementById('scale').value==='log';
 const series=A.curvas.filter(s=>active.has(s.estado));
 const v=series.flatMap(s=>s.T.map((t,i)=>val(s,key,i))).filter(x=>Number.isFinite(x)&&(!log||x>0));
 const trans=x=>log?Math.log10(x):x;
 let lo=Math.min(...v),hi=Math.max(...v); if(!v.length){document.getElementById('curve').textContent='Seleccione al menos un estado.';return}
 if(lo===hi){lo-=.01;hi+=.01}let l=trans(lo),h=trans(hi),pad=(h-l)*.07;l-=pad;h+=pad;
 const W=1000,H=460,L=85,R=25,T=30,B=70,x=t=>L+(20-t)/14*(W-L-R),y=v=>H-B-(trans(v)-l)/(h-l)*(H-B-T);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${metrics[key][0]} por temperatura"><title>${metrics[key][0]} por temperatura, ocho estados disponibles</title>`;
 for(let i=0;i<6;i++){const z=l+(h-l)*i/5,yy=H-B-i*(H-B-T)/5;svg+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#dce5ec"/><text x="${L-9}" y="${yy+5}" text-anchor="end" font-size="13">${fmt(log?10**z:z)}</text>`}
 for(let t=20;t>=6;t-=2)svg+=`<text x="${x(t)}" y="${H-B+25}" text-anchor="middle" font-size="14">${t}</text>`;
 if(key.startsWith('rhat')&&1.01>=lo&&1.01<=hi)svg+=`<line x1="${L}" x2="${W-R}" y1="${y(1.01)}" y2="${y(1.01)}" stroke="#111" stroke-dasharray="6 5"/>`;
 for(const s of series){let path='',started=false;for(let i=0;i<s.T.length;i++){let vv=val(s,key,i);if(!Number.isFinite(vv)||(log&&vv<=0)){started=false;continue}path+=(started?' L':' M')+x(s.T[i]).toFixed(2)+','+y(vv).toFixed(2);started=true}svg+=`<path d="${path}" fill="none" stroke="${colors[s.estado-1]}" stroke-width="2.4"><title>Estado ${s.estado}: kDM=${s.kDM}, H=${s.H}</title></path>`}
 svg+=`<text x="${W/2}" y="${H-15}" text-anchor="middle" font-size="16">Temperatura (K), sentido del enfriamiento</text><text x="${L}" y="20" font-size="15">${metrics[key][0]}</text></svg>`;
 document.getElementById('curve').innerHTML=svg;document.getElementById('metric-note').textContent=metrics[key][1];drawTable();
}
function drawTable(){const i=+document.getElementById('tempcompare').value,key=document.getElementById('metric').value,t=A.curvas[0].T[i];document.getElementById('tempvalue').textContent=t.toFixed(1)+' K';
 let head=['Estado','kDM','H',metrics[key][0],'m media','se entre caminos','P10 / P90 de medias de m'];let rows=A.curvas.filter(s=>active.has(s.estado)).map(s=>{const r=extra.get(s.estado+':'+t.toFixed(1));return [s.estado,s.kDM,s.H,fmt(val(s,key,i)),fmt(r.m_media),fmt(r.m_se_caminos),fmt(r.m_p10)+' / '+fmt(r.m_p90)]});
 document.getElementById('temp-table').innerHTML='<div class="table-wrap"><table><thead><tr>'+head.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
}
document.getElementById('legend').innerHTML=A.curvas.map(s=>`<label style="border-left:5px solid ${colors[s.estado-1]}"><input type="checkbox" checked data-e="${s.estado}"> Estado ${s.estado}, kDM=${s.kDM}, H=${s.H}</label>`).join('');
document.querySelectorAll('[data-e]').forEach(e=>e.addEventListener('change',()=>{e.checked?active.add(+e.dataset.e):active.delete(+e.dataset.e);draw()}));
document.getElementById('allstates').onclick=()=>{active=new Set([1,2,3,4,5,6,7,8]);document.querySelectorAll('[data-e]').forEach(e=>e.checked=true);draw()};
['metric','scale'].forEach(id=>document.getElementById(id).onchange=draw);document.getElementById('tempcompare').oninput=drawTable;
const snapMap=new Map(A.texturas.map(s=>[s.estado+':'+s.T.toFixed(1),s]));
function drawTextures(){const ti=+document.getElementById('snaptemp').value,t=A.temperaturas_texturas[ti],copy=+document.getElementById('copy').value;document.getElementById('snapvalue').textContent=t.toFixed(1)+' K';
 const box=document.getElementById('textures');box.innerHTML='';
 for(const s of A.curvas){const q=snapMap.get(s.estado+':'+t.toFixed(1)),card=document.createElement('div');card.className='texture';card.innerHTML=`<strong>Estado ${s.estado}</strong><br>kDM=${s.kDM}; H=${s.H}<br><canvas width="195" height="195" aria-label="Componente Sz, estado ${s.estado}, ${t} K, copia ${copy+1}"></canvas><br>m instantánea=${q.m_instantanea[copy].toFixed(4)}`;box.appendChild(card);const c=card.querySelector('canvas').getContext('2d');for(let j=0;j<39;j++)for(let i=0;i<39;i++){const z=q.sz[copy][j][i],a=Math.min(1,Math.abs(z));c.fillStyle=!q.mascara[j][i]?'#e8eef3':z>=0?`rgb(${Math.round(255-50*a)},${Math.round(255-210*a)},${Math.round(255-210*a)})`:`rgb(${Math.round(255-225*a)},${Math.round(255-155*a)},${Math.round(255-50*a)})`;c.fillRect(i*5,j*5,5,5)}}
}
let timer=null;document.getElementById('snaptemp').oninput=drawTextures;document.getElementById('copy').onchange=drawTextures;
document.getElementById('animate').onclick=()=>{const b=document.getElementById('animate');if(timer){clearInterval(timer);timer=null;b.textContent='Recorrer enfriamiento'}else{b.textContent='Pausar';timer=setInterval(()=>{const e=document.getElementById('snaptemp');e.value=(+e.value+1)%16;drawTextures()},1000)}};
function download(name,data,type='application/json'){const u=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
document.getElementById('download').onclick=()=>download('curvas_ocho_estados.json',JSON.stringify({curvas:A.curvas,magnetizacion:D.magnetizacion},null,2));
document.getElementById('audit-download').onclick=()=>download('auditoria_completa.json',JSON.stringify(A,null,2));
document.getElementById('moments-download').onclick=()=>{const k=Object.keys(A.momentos[0]);download('momentos_56_puntos.csv',k.join(',')+'\n'+A.momentos.map(r=>k.map(x=>r[x]).join(',')).join('\n'),'text/csv;charset=utf-8')};
const dialog=document.getElementById('image-dialog');document.querySelectorAll('img.original').forEach(img=>{const open=()=>{document.getElementById('modal-image').src=img.src;document.getElementById('modal-image').alt=img.alt;dialog.showModal()};img.onclick=open;img.onkeydown=e=>{if(e.key==='Enter')open()}});document.getElementById('close-modal').onclick=()=>dialog.close();
draw();drawTextures();
'''
CSS+='''
.mainlinks{display:flex;flex-wrap:wrap;gap:.7rem;font:15px/1.5 system-ui}.mainlinks a{display:inline-block;padding:.65rem .85rem;background:#0066b3;color:white;text-decoration:none;border-radius:4px}#informe-completo iframe{width:100%;height:78vh;min-height:500px;border:1px solid #cfdae4}pre{overflow:auto;padding:1rem;background:#f1f5f9;font:14px/1.6 monospace}pre code{padding:0;background:none}@media print{#informe-completo iframe,.mainlinks{display:none}}
'''
nav=''.join('<a href="#'+i+'">'+html.escape(t)+'</a>' for i,t,b in SECTIONS)
body=''.join('<section id="'+i+'"><h2>'+t+'</h2>'+b+'</section>' for i,t,b in SECTIONS)
page='<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Guía matemática completa del nanodisco quiral, ocho estados, enfriamiento, pruebas y revisión del informe final"><title>Nanodisco quiral: matemática, estados y pruebas</title><style>'+CSS+'</style></head><body><header><p>Universidad Nacional de Colombia · Computación Estadística · 2026-II</p><h1>Nanodisco quiral: matemática, enfriamiento y evidencia estadística</h1><p>Guía ampliada del informe final de Sebastián Mindiola Garizado. Problema, hipótesis, derivaciones, ocho estados y pruebas de convergencia, con comprobaciones de los resultados.</p><p>Informe y explicación completos en una misma página. Las curvas y las texturas proceden de las simulaciones del estudio.</p><p class="mainlinks"><a href="#informe-completo">Informe completo</a><a href="#problema">Explicación paso a paso</a><a href="#kaggle">Guía de Kaggle</a><a href="presentacion/presentacion.pdf">Diapositivas</a></p></header><main><nav aria-label="Contenido">'+nav+'</nav><article>'+body+'</article></main><footer>Fuente principal: informe/latex/main.tex. Revisión local: 23 de noviembre de 2026. Base cruda excluida; incorporación futura desde Kaggle.</footer><dialog id="image-dialog"><button id="close-modal" type="button">Cerrar figura</button><img id="modal-image" alt=""></dialog><script id="data" type="application/json">'+DATA.replace('</',r'<\/')+'</script><script>'+JS+'</script></body></html>'
out=ROOT/'index.html';out.write_text(page)
(BUILD/'explorador.js').write_text(JS)
print(f'HTML: {out}\nTamaño: {out.stat().st_size/1024/1024:.2f} MiB\nSecciones: {len(SECTIONS)}\nPalabras aproximadas de prosa y tablas: {len(re.sub("<[^>]+>"," ",body).split())}')
