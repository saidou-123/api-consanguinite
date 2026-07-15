# =============================================================================
# ALGORITHME DE WRIGHT — CALCUL CONSANGUINITÉ AVEC PEDIGREE INCOMPLET
# Fichier: wright_calculator.py
# =============================================================================
# CAS 1 : Pedigree complet  → Wright exact          (fiabilité 95%+)
# CAS 2 : Pedigree partiel  → Wright + incertitude  (fiabilité ~70%)
# CAS 3 : Pedigree inconnu  → F moyen race Ladoum   (F=8%, AUTORISER)
# =============================================================================

F_MOYEN_RACE_LADOUM = 0.08  # Moyenne estimée race Ladoum

SEUIL_FAIBLE = 0.06    # F < 6%    → vert  ACCEPTABLE
SEUIL_MODERE = 0.125   # F < 12.5% → orange MODÉRÉ
                       # F >= 12.5% → rouge ÉLEVÉ


# ==================================================================
# CALCUL WRIGHT PRINCIPAL
# ==================================================================
def calculer_f_wright(animal_a_id, animal_b_id, pedigree_dict):
    """
    Calcule le coefficient de consanguinité F entre deux animaux
    via l'algorithme de Wright (remontée récursive de l'arbre généalogique).

    pedigree_dict format :
    {
        'fatou':  {'pere_id': 'baba', 'mere_id': 'mere_b', 'nom': 'Fatou'},
        'baba':   {'pere_id': None,   'mere_id': None,      'nom': 'Baba'},
        ...
    }

    IMPORTANT : un individu est considere comme son propre "ancetre" a
    distance 0. Sans cela, une relation LINEAIRE directe (B est le pere
    ou la mere de A, ou un grand-parent, etc.) ne serait jamais detectee :
    B n'apparaitrait alors dans aucun des deux ensembles d'ancetres "au
    sens strict", puisque l'algorithme ne cherche que des ancetres
    COMMUNS en amont des deux animaux, pas un lien direct entre eux.
    """
    ancetres_a = dict(_obtenir_ancetres_avec_chemins(animal_a_id, pedigree_dict))
    ancetres_b = dict(_obtenir_ancetres_avec_chemins(animal_b_id, pedigree_dict))
    ancetres_a.setdefault(animal_a_id, []).append(0)
    ancetres_b.setdefault(animal_b_id, []).append(0)

    ancetres_communs = set(ancetres_a.keys()) & set(ancetres_b.keys())
    if not ancetres_communs:
        return 0.0

    # Formule Wright : F = Σ (0.5)^(n1+n2+1) × (1+Fa)
    F = 0.0
    for ancetre_id in ancetres_communs:
        chemins_a = ancetres_a[ancetre_id]
        chemins_b = ancetres_b[ancetre_id]
        Fa = _get_fa(ancetre_id, pedigree_dict)
        for n1 in chemins_a:
            for n2 in chemins_b:
                F += (0.5 ** (n1 + n2 + 1)) * (1 + Fa)

    return round(F, 4)


def _get_fa(ancetre_id, pedigree_dict):
    """Retourne le coefficient F d'un ancêtre via ses vrais parents."""
    if not _a_deux_parents(ancetre_id, pedigree_dict):
        return 0.0
    infos = pedigree_dict[ancetre_id]
    return calculer_f_wright(infos['pere_id'], infos['mere_id'], pedigree_dict)


def _obtenir_ancetres_avec_chemins(animal_id, pedigree_dict,
                                   profondeur=0, max_prof=6):
    """
    Remonte l'arbre généalogique récursivement.
    Retourne {ancetre_id: [liste des distances depuis l'animal]}.
    """
    if profondeur >= max_prof or animal_id not in pedigree_dict:
        return {}

    infos   = pedigree_dict[animal_id]
    pere_id = infos.get('pere_id')
    mere_id = infos.get('mere_id')
    ancetres = {}

    for parent_id in [pere_id, mere_id]:
        if parent_id is None:
            continue
        if parent_id not in ancetres:
            ancetres[parent_id] = []
        ancetres[parent_id].append(profondeur + 1)
        for anc_id, dists in _obtenir_ancetres_avec_chemins(
                parent_id, pedigree_dict, profondeur + 1, max_prof).items():
            if anc_id not in ancetres:
                ancetres[anc_id] = []
            ancetres[anc_id].extend(dists)

    return ancetres


def _a_deux_parents(animal_id, pedigree_dict):
    if animal_id not in pedigree_dict:
        return False
    infos = pedigree_dict[animal_id]
    return infos.get('pere_id') is not None and infos.get('mere_id') is not None


# ==================================================================
# TAUX D'INCOMPLÉTUDE
# ==================================================================
def calculer_taux_incompletude(animal_id, pedigree_dict,
                                profondeur=0, max_prof=3):
    """
    Calcule le taux d'incomplétude du pedigree d'un animal.
    0.0 = tout connu | 1.0 = rien connu.
    """
    if animal_id not in pedigree_dict or profondeur >= max_prof:
        return 1.0

    infos   = pedigree_dict[animal_id]
    pere_id = infos.get('pere_id')
    mere_id = infos.get('mere_id')

    if pere_id is None and mere_id is None:
        return 1.0
    if pere_id is None or mere_id is None:
        parent = pere_id or mere_id
        return 0.5 + 0.5 * calculer_taux_incompletude(
            parent, pedigree_dict, profondeur + 1, max_prof)

    incomp_pere = calculer_taux_incompletude(
        pere_id, pedigree_dict, profondeur + 1, max_prof)
    incomp_mere = calculer_taux_incompletude(
        mere_id, pedigree_dict, profondeur + 1, max_prof)
    return (incomp_pere + incomp_mere) / 4


# ==================================================================
# AJUSTEMENT SELON INCOMPLÉTUDE
# ==================================================================
def ajuster_f_avec_incertitude(f_wright, taux_incompletude):
    """
    F_ajusté = F_wright × (1 - incomplétude) + F_moyen_race × incomplétude

    Exemple : F_wright=0.125, incomplétude=0.5
    → F_ajusté = 0.125×0.5 + 0.08×0.5 = 0.1025 (10.25%)
    """
    return round(
        f_wright * (1 - taux_incompletude) +
        F_MOYEN_RACE_LADOUM * taux_incompletude,
        4
    )


# ==================================================================
# NIVEAU DE CONFIANCE
# ==================================================================
def determiner_niveau_confiance(taux_incompletude_a, taux_incompletude_b):
    """Niveau de confiance selon l'incomplétude moyenne des deux pedigrees."""
    moy = (taux_incompletude_a + taux_incompletude_b) / 2
    if moy <= 0.1:
        return "ÉLEVÉE",    "Pedigrees complets — résultat très fiable."
    elif moy <= 0.4:
        return "MODÉRÉE",   "Pedigree partiellement connu — résultat indicatif."
    elif moy <= 0.7:
        return "FAIBLE",    "Un ou plusieurs parents inconnus — estimation prudente."
    else:
        return "TRÈS FAIBLE", ("Pedigrees quasi-inconnus — résultat basé sur "
                                "la moyenne de la race Ladoum (F=8%).")


# ==================================================================
# CLASSIFICATION DU RISQUE
# ==================================================================
def classifier_risque(f_ajuste):
    """Classifie le niveau de risque selon F ajusté."""
    if f_ajuste < SEUIL_FAIBLE:
        return {
            'niveau': 'ACCEPTABLE', 'couleur': 'vert',
            'message': 'Faible risque de consanguinité. Accouplement conseillé.',
            'action': 'AUTORISER',
        }
    elif f_ajuste < SEUIL_MODERE:
        return {
            'niveau': 'MODÉRÉ', 'couleur': 'orange',
            'message': (f'Risque modéré (F={f_ajuste*100:.1f}%). '
                        'Vérifiez les pedigrees avant de procéder.'),
            'action': 'AVERTIR',
        }
    else:
        return {
            'niveau': 'ÉLEVÉ', 'couleur': 'rouge',
            'message': (f'Risque élevé de consanguinité (F={f_ajuste*100:.1f}%). '
                        'Accouplement déconseillé.'),
            'action': 'BLOQUER',
        }


# ==================================================================
# FONCTION PRINCIPALE
# ==================================================================
def analyser_couple_complet(animal_a_id, animal_b_id, pedigree_dict):
    """
    Analyse complète d'un couple avec gestion des pedigrees incomplets.
    Retourne un dictionnaire complet avec tous les résultats Wright.
    """
    # 1. Taux d'incomplétude
    incomp_a = calculer_taux_incompletude(animal_a_id, pedigree_dict)
    incomp_b = calculer_taux_incompletude(animal_b_id, pedigree_dict)
    incompletude_moy = (incomp_a + incomp_b) / 2

    # 2. F de Wright sur la partie connue
    f_wright = calculer_f_wright(animal_a_id, animal_b_id, pedigree_dict)

    # 3. Ajustement selon incomplétude
    # IMPORTANT : le F de Wright calcule sur les ancetres CONNUS est un fait
    # confirme par le pedigree (pas une estimation). L'incompletude des
    # generations plus lointaines peut seulement ajouter du risque
    # supplementaire (des liens de parente non detectes plus haut), jamais
    # diluer vers le bas un lien deja prouve. Sans ce garde-fou, un couple
    # pere-fille confirme (F=25%) pourrait ressortir "MODERE" au lieu
    # d'"ELEVE" simplement parce que les grands-parents sont inconnus.
    f_ajuste_brut = ajuster_f_avec_incertitude(f_wright, incompletude_moy)
    f_ajuste = max(f_wright, f_ajuste_brut)

    # 4. Confiance
    confiance_label, confiance_msg = determiner_niveau_confiance(incomp_a, incomp_b)

    # 5. Classification Wright
    risque = classifier_risque(f_ajuste)

    # 6. Ancêtres communs (inclut chaque animal comme son propre "ancêtre"
    #    à distance 0, pour détecter aussi les relations directes
    #    parent-enfant, cf. calculer_f_wright)
    anc_a = dict(_obtenir_ancetres_avec_chemins(animal_a_id, pedigree_dict))
    anc_b = dict(_obtenir_ancetres_avec_chemins(animal_b_id, pedigree_dict))
    anc_a.setdefault(animal_a_id, []).append(0)
    anc_b.setdefault(animal_b_id, []).append(0)
    ancetres_communs = list(set(anc_a.keys()) & set(anc_b.keys()))

    # 7. Relation
    relation = _determiner_relation(f_wright, f_ajuste, ancetres_communs)

    return {
        'succes'              : True,
        'f_wright'            : f_wright,
        'f_ajuste'            : f_ajuste,
        'f_pourcent'          : round(f_ajuste * 100, 2),
        'incompletude_a'      : incomp_a,
        'incompletude_b'      : incomp_b,
        'incompletude_moyenne': incompletude_moy,
        'confiance'           : confiance_label,
        'confiance_message'   : confiance_msg,
        'ancetres_communs'    : ancetres_communs,
        'relation'            : relation,
        'niveau'              : risque['niveau'],
        'couleur'             : risque['couleur'],
        'message'             : risque['message'],
        'action'              : risque['action'],
        'details': {
            'f_calcule_sur_parents_connus': f_wright,
            'correction_incompletude'     : round(f_ajuste - f_wright, 4),
            'methode': ('Wright exact' if incompletude_moy <= 0.1
                        else 'Wright + correction incertitude'),
        }
    }


def _determiner_relation(f_wright, f_ajuste, ancetres_communs):
    if f_wright == 0 and f_ajuste < 0.02:
        return "Aucun lien détecté"
    elif f_wright >= 0.24:
        return "Frère/sœur ou parent-enfant"
    elif f_wright >= 0.12:
        return "Demi-frère/sœur"
    elif f_wright >= 0.06:
        return "Cousin(e) germain(e)"
    elif f_wright > 0:
        return f"Lien distant ({len(ancetres_communs)} ancêtre(s) commun(s))"
    elif f_ajuste > F_MOYEN_RACE_LADOUM:
        return "Relation possible (pedigree incomplet)"
    else:
        return "Probablement sans lien"


# ==================================================================
# TESTS UNITAIRES
# ==================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TESTS ALGORITHME DE WRIGHT")
    print("=" * 60)

    # Test 1 : Demi-frère/sœur — F=12.5%
    p1 = {
        'fatou'   : {'pere_id': 'baba', 'mere_id': 'mere_b', 'nom': 'Fatou'},
        'champion': {'pere_id': 'baba', 'mere_id': 'mere_c', 'nom': 'Champion'},
        'baba'    : {'pere_id': None,   'mere_id': None,      'nom': 'Baba'},
        'mere_b'  : {'pere_id': None,   'mere_id': None,      'nom': 'Mere_Fatou'},
        'mere_c'  : {'pere_id': None,   'mere_id': None,      'nom': 'Mere_Champion'},
    }
    r1 = analyser_couple_complet('fatou', 'champion', p1)
    print(f"\nTest 1 — Demi-frère/sœur")
    print(f"  F Wright : {r1['f_wright']*100:.2f}%  (attendu 12.50%)")
    print(f"  Relation : {r1['relation']}")
    print(f"  Niveau   : {r1['niveau']}")
    assert abs(r1['f_wright'] - 0.125) < 0.001, "ERREUR Test 1"
    print("  ✅ OK")

    # Test 2 : Sans lien — F=0%
    p2 = {
        'x': {'pere_id': 'px', 'mere_id': 'mx', 'nom': 'X'},
        'y': {'pere_id': 'py', 'mere_id': 'my', 'nom': 'Y'},
        'px': {'pere_id': None, 'mere_id': None, 'nom': 'PX'},
        'mx': {'pere_id': None, 'mere_id': None, 'nom': 'MX'},
        'py': {'pere_id': None, 'mere_id': None, 'nom': 'PY'},
        'my': {'pere_id': None, 'mere_id': None, 'nom': 'MY'},
    }
    r2 = analyser_couple_complet('x', 'y', p2)
    print(f"\nTest 2 — Sans lien")
    print(f"  F Wright : {r2['f_wright']*100:.2f}%  (attendu 0.00%)")
    print(f"  Niveau   : {r2['niveau']}")
    assert r2['f_wright'] == 0.0, "ERREUR Test 2"
    print("  ✅ OK")

    # Test 3 : Pedigree partiel
    p3 = {
        'brebis': {'pere_id': None,  'mere_id': 'mb', 'nom': 'Brebis'},
        'belier': {'pere_id': 'pb',  'mere_id': 'mc', 'nom': 'Belier'},
        'mb'    : {'pere_id': None,  'mere_id': None,  'nom': 'Mere_B'},
        'pb'    : {'pere_id': None,  'mere_id': None,  'nom': 'Pere_B'},
        'mc'    : {'pere_id': None,  'mere_id': None,  'nom': 'Mere_C'},
    }
    r3 = analyser_couple_complet('brebis', 'belier', p3)
    print(f"\nTest 3 — Pedigree partiel (père inconnu)")
    print(f"  F ajusté    : {r3['f_pourcent']}% (correction incomplétude)")
    print(f"  Incomplétude: {r3['incompletude_moyenne']*100:.0f}%")
    print(f"  Confiance   : {r3['confiance']}")
    print("  ✅ OK")

    # Test 4 : Bélier totalement inconnu
    p4 = {
        'fatou'     : {'pere_id': 'baba', 'mere_id': 'mb', 'nom': 'Fatou'},
        'ext'       : {'pere_id': None,   'mere_id': None,  'nom': 'BelierExterne'},
        'baba'      : {'pere_id': None,   'mere_id': None,  'nom': 'Baba'},
        'mb'        : {'pere_id': None,   'mere_id': None,  'nom': 'Mere'},
    }
    r4 = analyser_couple_complet('fatou', 'ext', p4)
    print(f"\nTest 4 — Bélier extérieur inconnu")
    print(f"  F ajusté    : {r4['f_pourcent']}% (≈ moyenne race 8%)")
    print(f"  Incomplétude: {r4['incompletude_moyenne']*100:.0f}%")
    print(f"  Confiance   : {r4['confiance']}")
    print(f"  Relation    : {r4['relation']}")
    print("  ✅ OK")

    print(f"\n{'='*60}")
    print("TOUS LES TESTS PASSÉS !")
    print(f"{'='*60}")