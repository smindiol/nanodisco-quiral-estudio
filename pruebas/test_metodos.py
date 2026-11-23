"""Propiedades estadísticas verificables sin GPU ni matrices crudas."""
from pathlib import Path
import importlib.util
import json
import unittest
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('auditoria', ROOT/'scripts/auditoria/revisar_datos.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Metodos(unittest.TestCase):
    def test_dominio_general_reproduce_el_sistema_tridimensional(self):
        with np.load(ROOT/'presentacion/datos/estado6_copia63_3d.npz') as data:
            mask = data['mascara'].astype(bool)
            y, x = np.mgrid[-19:20, -19:20]
            disk = x*x+y*y <= float(data['Rd'])**2
            expected = np.broadcast_to(disk, (5, 39, 39))
            np.testing.assert_array_equal(mask, expected)
            self.assertEqual(int(mask.sum()), 5425)
            self.assertEqual(data['spins'].shape, (5425, 3))
            np.testing.assert_allclose(np.linalg.norm(data['spins'], axis=1), 1, atol=1e-3)

    def test_descubrimiento_admite_los_tres_formatos_de_descarga(self):
        spec = importlib.util.spec_from_file_location('analisis', ROOT/'scripts/analisis/dmi_analisis.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for layout in ['lote1/lote_01/estado_1', 'lote_01/estado_1', 'estado_1']:
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)/layout
                folder.mkdir(parents=True)
                for name in ['T_19.9.npz','T_20.0.npz']:
                    (folder/name).touch()
                found = module.descubrir(tmp)
                self.assertEqual(list(found), [1])
                self.assertEqual([Path(p).name for p in found[1]], ['T_20.0.npz','T_19.9.npz'])

    def test_rhat_es_invariante_a_un_cambio_de_unidades(self):
        rng = np.random.default_rng(20261123)
        x = rng.normal(size=(2000, 12))
        self.assertAlmostEqual(audit.rh(x), audit.rh(37*x+200), places=12)
        self.assertAlmostEqual(audit.rank_rh(x), audit.rank_rh(37*x+200), places=12)

    def test_rhat_detecta_sectores_desplazados(self):
        rng = np.random.default_rng(20261123)
        x = rng.normal(size=(2000, 12))
        self.assertLess(audit.rank_rh(x), 1.01)
        x[:, 6:] += 4
        self.assertGreater(audit.rank_rh(x), 1.1)

    def test_descomposicion_de_varianza_con_divisores_poblacionales(self):
        rng = np.random.default_rng(20261123)
        x = rng.normal(size=(1000, 8)) + np.arange(8)
        within = x.var(axis=0).mean()
        between = x.mean(axis=0).var()
        self.assertAlmostEqual(x.var(), within+between, places=12)
        self.assertGreater(between, within)

    def test_reconstruccion_energetica_coincide_con_auditoria(self):
        data = json.loads((ROOT/'resultados/auditoria/auditoria.json').read_text())
        for state in data['curvas']:
            with np.load(ROOT/f"resultados/pasos_mc/estado_{state['estado']}.npz") as d:
                v = ((9999/10000)*d['W_E']+(99/100)*d['Bn_E'])*d['Eo']**2
                c = v/(5425*0.086173404*d['T']**2)
                np.testing.assert_allclose(c, state['c_recalculado'], rtol=1e-12)
                np.testing.assert_allclose(d['E_med']*d['Eo']/5425,
                                           state['energia_meV_por_espin'], rtol=1e-12)

    def test_diseno_y_auditoria_tienen_las_mismas_temperaturas(self):
        data = json.loads((ROOT/'resultados/auditoria/auditoria.json').read_text())
        rows = json.loads((ROOT/'resultados/auditoria/magnetizacion.json').read_text())
        self.assertEqual(len(rows), 1128)
        self.assertEqual(len({(r['estado'], r['T']) for r in rows}), 1128)
        for state in data['curvas']:
            actual = sorted(r['T'] for r in rows if r['estado']==state['estado'])
            np.testing.assert_allclose(actual, sorted(state['T']), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
