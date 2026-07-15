# =============================================================================
# SERVICE PEDIGREE — Construit l'arbre généalogique depuis Supabase
# Fichier: pedigree_service.py
# =============================================================================
# Remonte jusqu'à MAX_GENERATIONS (6) dans les tables :
#   - animal_acheter  (source='achete')
#   - nouveaux_nee    (source='nee')
# =============================================================================

import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

MAX_GENERATIONS = 6  # 6 générations pour couvrir les pedigrees profonds


def _table(source: str) -> str:
    return 'animal_acheter' if source == 'achete' else 'nouveaux_nee'


def _cle(animal_id, source: str) -> str:
    return f"{source}_{animal_id}"


def charger_animal(supabase: Client, animal_id, source: str) -> dict | None:
    """
    Charge les infos généalogiques d'un animal depuis Supabase.
    Compatible supabase-py v2 (.limit(1) au lieu de .maybeSingle()).
    """
    try:
        res = (
            supabase
            .from_(_table(source))
            .select('id, nom, pere_id, mere_id, source_pere, source_mere')
            .eq('id', str(animal_id))
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        print(f"⚠️  Erreur chargement {source}_{animal_id}: {e}")
        return None


def construire_pedigree(
    supabase: Client,
    animal_id,
    source: str,
    pedigree: dict | None = None,
    profondeur: int = 0,
) -> dict:
    """
    Remonte l'arbre généalogique récursivement jusqu'à MAX_GENERATIONS.

    Retourne pedigree_dict compatible wright_calculator.py :
    {
      'achete_12': {'pere_id': 'achete_5', 'mere_id': 'nee_8', 'nom': 'Fatou'},
      'achete_5' : {'pere_id': None,        'mere_id': None,    'nom': 'Baba'},
      ...
    }
    """
    if pedigree is None:
        pedigree = {}
    if profondeur >= MAX_GENERATIONS:
        return pedigree

    cle = _cle(animal_id, source)
    if cle in pedigree:
        return pedigree

    animal = charger_animal(supabase, animal_id, source)
    if animal is None:
        pedigree[cle] = {'pere_id': None, 'mere_id': None,
                         'nom': f'Inconnu ({cle})'}
        return pedigree

    source_pere = animal.get('source_pere') or 'achete'
    source_mere = animal.get('source_mere') or 'achete'
    pere_id     = animal.get('pere_id')
    mere_id     = animal.get('mere_id')

    cle_pere = _cle(pere_id, source_pere) if pere_id is not None else None
    cle_mere = _cle(mere_id, source_mere) if mere_id is not None else None

    pedigree[cle] = {
        'pere_id': cle_pere,
        'mere_id': cle_mere,
        'nom'    : animal.get('nom', cle),
    }

    if pere_id is not None:
        construire_pedigree(supabase, pere_id, source_pere, pedigree, profondeur + 1)
    if mere_id is not None:
        construire_pedigree(supabase, mere_id, source_mere, pedigree, profondeur + 1)

    return pedigree


def fusionner_pedigrees(
    supabase: Client,
    id_a, source_a: str,
    id_b, source_b: str,
) -> tuple[dict, str, str]:
    """
    Construit le pedigree fusionné des deux animaux.
    Retourne (pedigree_dict, cle_a, cle_b).
    """
    pedigree = {}
    construire_pedigree(supabase, id_a, source_a, pedigree)
    construire_pedigree(supabase, id_b, source_b, pedigree)
    return pedigree, _cle(id_a, source_a), _cle(id_b, source_b)


def extraire_noms_ancetres_communs(
    pedigree: dict,
    ancetres_communs: list[str]
) -> list[str]:
    """Retourne les noms lisibles des ancêtres communs."""
    return [pedigree.get(cle, {}).get('nom', cle) for cle in ancetres_communs]


# ==================================================================
# TEST LOCAL
# ==================================================================
if __name__ == '__main__':
    print("Test pedigree_service — simulation sans Supabase")
    pedigree_sim = {
        'achete_1': {'pere_id': 'achete_3', 'mere_id': 'nee_4',    'nom': 'Fatou'},
        'achete_2': {'pere_id': 'achete_3', 'mere_id': 'achete_5', 'nom': 'Champion'},
        'achete_3': {'pere_id': None,        'mere_id': None,        'nom': 'Baba'},
        'nee_4'   : {'pere_id': None,        'mere_id': None,        'nom': 'Mere_Fatou'},
        'achete_5': {'pere_id': None,        'mere_id': None,        'nom': 'Mere_Champion'},
    }
    from wright_calculator import analyser_couple_complet
    res = analyser_couple_complet('achete_1', 'achete_2', pedigree_sim)
    print(f"F Wright : {res['f_wright']*100:.2f}%  (attendu: 12.50%)")
    print(f"F ajusté : {res['f_pourcent']}%")
    print(f"Relation : {res['relation']}")
    print(f"Ancêtres : {res['ancetres_communs']}")
    print("✅ OK" if abs(res['f_wright'] - 0.125) < 0.001 else "❌ ERREUR")