PYTHON ?= python
NODE ?= node

.PHONY: verificar html informe presentacion auditoria magnetizacion publicar

verificar:
	$(PYTHON) scripts/verificar_repositorio.py
	$(PYTHON) pruebas/validar_html.py
	$(NODE) --check build/explorador.js
	$(NODE) pruebas/validar_controles.cjs
	$(PYTHON) -m unittest discover -s pruebas -p 'test_*.py' -v

html:
	MPLCONFIGDIR="$(CURDIR)/build/matplotlib" $(PYTHON) scripts/informe/generar_html.py

informe:
	bash scripts/informe/compilar_informe.sh

presentacion:
	MPLCONFIGDIR="$(CURDIR)/build/matplotlib" $(PYTHON) presentacion/generar_figuras.py
	bash presentacion/compilar.sh

auditoria:
	$(PYTHON) scripts/auditoria/revisar_datos.py

magnetizacion:
	$(PYTHON) scripts/auditoria/curvas_magnetizacion.py

publicar:
	bash scripts/publicar_github.sh
