# =============================================================================
# TESTS AUTOMATISES — API Consanguinite (Wright seul)
# Fichier: test_api_consanguinite.py
# =============================================================================
# Usage :
#   python test_api_consanguinite.py
#
# Ces tests utilisent un faux client Supabase (aucune connexion reseau
# requise), donc reproductibles a l'identique partout, y compris sur
# une machine sans acces a votre base reelle.
#
# Chaque test correspond a un cas reellement rencontre et corrige durant
# le developpement :
#   - test_demi_frere_soeur       -> bug de dilution (#6)
#   - test_pere_fille_directe     -> bug de detection lineaire (#5)
#   - test_pere_fille_grands_parents_inconnus -> bug d'ecrasement (#7)
#   - test_aucun_lien             -> non-regression (pas de faux positif)
#   - test_pedigree_totalement_absent -> non-regression (F moyen race)
# =============================================================================

import sys
import json
from unittest.mock import patch

sys.path.insert(0, '.')

import os
os.environ.setdefault('SUPABASE_URL', 'https://fake.supabase.co')
os.environ.setdefault('SUPABASE_KEY', 'fake_key')

import api


# ==================================================================
# INFRASTRUCTURE DE TEST — faux client Supabase en memoire
# ==================================================================
class FausseReponse:
    def __init__(self, data):
        self.data = data


class FauxQuery:
    def __init__(self, table, base_animaux):
        self.table = table
        self.base_animaux = base_animaux
        self._id = None

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._id = val
        return self

    def limit(self, n):
        return self

    def execute(self):
        animal = self.base_animaux.get((self.table, str(self._id)))
        return FausseReponse([animal] if animal else [])


class FauxSupabase:
    def __init__(self, base_animaux):
        self.base_animaux = base_animaux

    def from_(self, table):
        return FauxQuery(table, self.base_animaux)


def appeler_api(base_animaux, brebis_id, source_brebis, belier_id, source_belier):
    """Appelle /analyser-pedigree avec une base Supabase simulee."""
    with patch('supabase.create_client', return_value=FauxSupabase(base_animaux)):
        with api.app.test_client() as client:
            resp = client.post('/analyser-pedigree', json={
                'brebis_id': brebis_id, 'source_brebis': source_brebis,
                'belier_id': belier_id, 'source_belier': source_belier,
            })
            return resp.get_json()


# ==================================================================
# JEUX DE DONNEES REUTILISABLES
# ==================================================================
FAMILLE_FATOU_CHAMPION = {
    ('nouveaux_nee', 'fatou'): {
        'id': 'fatou', 'nom': 'Fatou', 'pere_id': 'baba', 'mere_id': None,
        'source_pere': 'achete', 'source_mere': None, 'score_sante': 0.9,
        'statut_fondateur': 0,
    },
    ('animal_acheter', 'champion'): {
        'id': 'champion', 'nom': 'Champion', 'pere_id': 'baba', 'mere_id': None,
        'source_pere': 'achete', 'source_mere': None, 'score_sante': 0.85,
        'statut_fondateur': 0,
    },
    ('animal_acheter', 'baba'): {
        'id': 'baba', 'nom': 'Baba', 'pere_id': None, 'mere_id': None,
        'statut_fondateur': 1,
    },
}

FAMILLE_FIFI_JR = {
    ('nouveaux_nee', 'fifi_jr'): {
        'id': 'fifi_jr', 'nom': 'Fifi Jr', 'pere_id': 'champion2', 'mere_id': 'yassine',
        'source_pere': 'achete', 'source_mere': 'achete', 'score_sante': 0.88,
        'statut_fondateur': 0,
    },
    ('animal_acheter', 'champion2'): {
        'id': 'champion2', 'nom': 'Champion', 'pere_id': None, 'mere_id': None,
        'score_sante': 0.9, 'statut_fondateur': 1,
    },
    ('animal_acheter', 'yassine'): {
        'id': 'yassine', 'nom': 'Yassine', 'pere_id': None, 'mere_id': None,
        'score_sante': 0.85, 'statut_fondateur': 1,
    },
}

FAMILLE_SANS_LIEN = {
    ('animal_acheter', 'x'): {
        'id': 'x', 'nom': 'X', 'pere_id': 'px', 'mere_id': 'mx',
        'source_pere': 'achete', 'source_mere': 'achete', 'statut_fondateur': 0,
    },
    ('animal_acheter', 'y'): {
        'id': 'y', 'nom': 'Y', 'pere_id': 'py', 'mere_id': 'my',
        'source_pere': 'achete', 'source_mere': 'achete', 'statut_fondateur': 0,
    },
    ('animal_acheter', 'px'): {'id': 'px', 'nom': 'PX', 'pere_id': None, 'mere_id': None, 'statut_fondateur': 1},
    ('animal_acheter', 'mx'): {'id': 'mx', 'nom': 'MX', 'pere_id': None, 'mere_id': None, 'statut_fondateur': 1},
    ('animal_acheter', 'py'): {'id': 'py', 'nom': 'PY', 'pere_id': None, 'mere_id': None, 'statut_fondateur': 1},
    ('animal_acheter', 'my'): {'id': 'my', 'nom': 'MY', 'pere_id': None, 'mere_id': None, 'statut_fondateur': 1},
}


# ==================================================================
# TESTS
# ==================================================================
def test_demi_frere_soeur():
    """Fatou et Champion partagent un pere confirme (Baba).
    F Wright attendu = 12.5% (valeur standard demi-frere/soeur).
    Ne doit PAS etre dilue en dessous de cette valeur confirmee."""
    r = appeler_api(FAMILLE_FATOU_CHAMPION, 'fatou', 'nee', 'champion', 'achete')
    assert r['f_pourcent'] == 12.5, f"F attendu 12.5%, obtenu {r['f_pourcent']}%"
    assert r['niveau'] == 'ÉLEVÉ', f"Niveau attendu ÉLEVÉ, obtenu {r['niveau']}"
    assert r['action'] == 'BLOQUER'
    assert 'Baba' in r['ancetres_communs']
    assert r['relation'] == 'Demi-frère/sœur'
    print("✅ test_demi_frere_soeur")


def test_pere_fille_directe():
    """Fifi Jr est la fille directe de Champion. F attendu = 25%
    (relation lineaire directe, pas juste un ancetre partage en amont)."""
    r = appeler_api(FAMILLE_FIFI_JR, 'fifi_jr', 'nee', 'champion2', 'achete')
    assert r['f_pourcent'] == 25.0, f"F attendu 25%, obtenu {r['f_pourcent']}%"
    assert r['niveau'] == 'ÉLEVÉ', f"Niveau attendu ÉLEVÉ, obtenu {r['niveau']}"
    assert r['action'] == 'BLOQUER'
    assert r['couleur'] == 'rouge'
    assert 'Champion' in r['ancetres_communs']
    print("✅ test_pere_fille_directe")


def test_pere_fille_grands_parents_inconnus():
    """Meme cas que ci-dessus : verifie explicitement que l'incompletude
    des grands-parents (Champion et Yassine sont fondateurs, donc
    incompletude=75%) ne fait PAS retomber le niveau a MODERE/ACCEPTABLE."""
    r = appeler_api(FAMILLE_FIFI_JR, 'fifi_jr', 'nee', 'champion2', 'achete')
    assert r['incompletude_moyenne'] == 0.75
    assert r['niveau'] == 'ÉLEVÉ', (
        "REGRESSION: le lien confirme pere-fille a ete efface par "
        "l'heuristique 'pedigree incomplet' -- verifier le garde-fou "
        "'f_wright == 0.0' dans api.py"
    )
    print("✅ test_pere_fille_grands_parents_inconnus (garde-fou anti-regression)")


def test_aucun_lien():
    """Deux animaux sans aucun ancetre commun. F Wright doit rester a 0%,
    et le resultat ne doit jamais indiquer un risque eleve par erreur."""
    r = appeler_api(FAMILLE_SANS_LIEN, 'x', 'achete', 'y', 'achete')
    assert r['f_wright'] == 0.0
    assert r['niveau'] in ('ACCEPTABLE', 'MODÉRÉ'), (
        f"Aucun lien reel mais niveau={r['niveau']} (ne devrait jamais etre ÉLEVÉ)"
    )
    assert r['niveau'] != 'ÉLEVÉ'
    print("✅ test_aucun_lien (pas de faux positif)")


def test_pedigree_totalement_absent():
    """Aucun des deux animaux n'existe dans la base simulee.
    Doit retomber sur F=8% (moyenne race), classe honnetement
    (pas force a ACCEPTABLE)."""
    r = appeler_api({}, 'inconnu1', 'achete', 'inconnu2', 'achete')
    assert r['f_pourcent'] == 8.0
    assert r['succes'] is True
    print("✅ test_pedigree_totalement_absent")


# ==================================================================
# EXECUTION
# ==================================================================
if __name__ == '__main__':
    tests = [
        test_demi_frere_soeur,
        test_pere_fille_directe,
        test_pere_fille_grands_parents_inconnus,
        test_aucun_lien,
        test_pedigree_totalement_absent,
    ]

    print("=" * 70)
    print("TESTS AUTOMATISES — API CONSANGUINITE")
    print("=" * 70)

    echecs = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            echecs.append((test.__name__, str(e)))
            print(f"❌ {test.__name__} : {e}")
        except Exception as e:
            echecs.append((test.__name__, f"Erreur inattendue: {e}"))
            print(f"💥 {test.__name__} : erreur inattendue: {e}")

    print("=" * 70)
    if echecs:
        print(f"❌ {len(echecs)}/{len(tests)} TEST(S) EN ECHEC")
        sys.exit(1)
    else:
        print(f"✅ TOUS LES TESTS PASSENT ({len(tests)}/{len(tests)})")
        sys.exit(0)