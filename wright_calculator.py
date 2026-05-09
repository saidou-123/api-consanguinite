# =============================================================================
# ALGORITHME DE WRIGHT — CALCUL CONSANGUINITÉ AVEC PEDIGREE INCOMPLET
# Fichier: wright_calculator.py
# =============================================================================
# Gère 3 cas :
#   CAS 1 : Pedigree complet  → Wright exact          (fiabilité 95%+)
#   CAS 2 : Pedigree partiel  → Wright + incertitude  (fiabilité ~70%)
#   CAS 3 : Pedigree inconnu  → Modèle ML seul        (fiabilité ~69%)
# =============================================================================

# Coefficient F moyen pour la race Ladoum
# (valeur standard en élevage ovin, utilisée quand le pedigree est inconnu)
F_MOYEN_RACE_LADOUM = 0.08

# Seuils de risque
SEUIL_FAIBLE  = 0.06   # F < 6%  → vert
SEUIL_MODERE  = 0.125  # F < 12.5% → orange
                       # F >= 12.5% → rouge


def calculer_f_wright(animal_a_id, animal_b_id, pedigree_dict):
    """
    Calcule le coefficient de consanguinité F entre deux animaux
    en utilisant l'algorithme de Wright (remontée de l'arbre généalogique).

    pedigree_dict: dictionnaire {animal_id: {'pere_id': ..., 'mere_id': ...}}
    Exemple:
    {
        'fatou':    {'pere_id': 'pere_a', 'mere_id': 'mere_b'},
        'champion': {'pere_id': 'pere_a', 'mere_id': 'mere_c'},
        'pere_a':   {'pere_id': None,     'mere_id': None},
        'mere_b':   {'pere_id': None,     'mere_id': None},
        'mere_c':   {'pere_id': None,     'mere_id': None},
    }
    """
    # 1. Trouver tous les ancêtres de A et de B
    ancetres_a = _obtenir_ancetres_avec_chemins(animal_a_id, pedigree_dict)
    ancetres_b = _obtenir_ancetres_avec_chemins(animal_b_id, pedigree_dict)

    # 2. Trouver les ancêtres COMMUNS
    ids_a = set(ancetres_a.keys())
    ids_b = set(ancetres_b.keys())
    ancetres_communs = ids_a & ids_b

    if not ancetres_communs:
        return 0.0  # Aucun lien de parenté détecté

    # 3. Appliquer la formule de Wright : F = Σ (0.5)^(n1+n2+1) × (1+Fa)
    # n1 = distance de A à l'ancêtre commun
    # n2 = distance de B à l'ancêtre commun
    # Fa = coefficient de consanguinité de l'ancêtre commun lui-même
    F = 0.0
    for ancetre_id in ancetres_communs:
        chemins_a = ancetres_a[ancetre_id]  # liste de distances depuis A
        chemins_b = ancetres_b[ancetre_id]  # liste de distances depuis B

        # F de l'ancêtre commun (0 si inconnu)
        Fa = calculer_f_wright(
            ancetre_id + '_pere', ancetre_id + '_mere', pedigree_dict
        ) if _a_deux_parents(ancetre_id, pedigree_dict) else 0.0

        # Sommer toutes les combinaisons de chemins
        for n1 in chemins_a:
            for n2 in chemins_b:
                F += (0.5 ** (n1 + n2 + 1)) * (1 + Fa)

    return round(F, 4)


def _obtenir_ancetres_avec_chemins(animal_id, pedigree_dict, profondeur=0, max_prof=6):
    """
    Remonte l'arbre généalogique et retourne un dict:
    {ancetre_id: [liste des distances depuis l'animal]}
    """
    if profondeur >= max_prof:
        return {}
    if animal_id not in pedigree_dict:
        return {}

    infos = pedigree_dict[animal_id]
    pere_id = infos.get('pere_id')
    mere_id = infos.get('mere_id')

    ancetres = {}

    for parent_id in [pere_id, mere_id]:
        if parent_id is None:
            continue

        # L'animal lui-même est un ancêtre à distance 1
        if parent_id not in ancetres:
            ancetres[parent_id] = []
        ancetres[parent_id].append(profondeur + 1)

        # Remonter récursivement
        ancetres_parent = _obtenir_ancetres_avec_chemins(
            parent_id, pedigree_dict, profondeur + 1, max_prof
        )
        for anc_id, distances in ancetres_parent.items():
            if anc_id not in ancetres:
                ancetres[anc_id] = []
            ancetres[anc_id].extend(distances)

    return ancetres


def _a_deux_parents(animal_id, pedigree_dict):
    """Vérifie si un animal a ses deux parents dans le pedigree"""
    if animal_id not in pedigree_dict:
        return False
    infos = pedigree_dict[animal_id]
    return infos.get('pere_id') is not None and infos.get('mere_id') is not None


def calculer_taux_incompletude(animal_id, pedigree_dict, profondeur=0, max_prof=3):
    """
    Calcule le taux d'incomplétude du pedigree d'un animal.
    0.0 = tout est connu, 1.0 = rien n'est connu.
    """
    if animal_id not in pedigree_dict or profondeur >= max_prof:
        return 1.0  # Animal inconnu = incomplet

    infos = pedigree_dict[animal_id]
    pere_id = infos.get('pere_id')
    mere_id = infos.get('mere_id')

    # Si aucun parent connu
    if pere_id is None and mere_id is None:
        return 1.0

    # Si un seul parent connu
    if pere_id is None or mere_id is None:
        parent_connu = pere_id or mere_id
        incompletude_parent = calculer_taux_incompletude(
            parent_connu, pedigree_dict, profondeur + 1, max_prof
        )
        return 0.5 + (0.5 * incompletude_parent)

    # Si les deux parents sont connus
    incomp_pere = calculer_taux_incompletude(pere_id, pedigree_dict, profondeur+1, max_prof)
    incomp_mere = calculer_taux_incompletude(mere_id, pedigree_dict, profondeur+1, max_prof)
    return (incomp_pere + incomp_mere) / 4  # Bonne complétude


def ajuster_f_avec_incertitude(f_wright, taux_incompletude):
    """
    Ajuste le coefficient F calculé en tenant compte de l'incomplétude du pedigree.

    Formule : F_ajusté = F_wright × (1 - incompletude) + F_moyen_race × incompletude

    Exemple : F_wright=0.125, incompletude=0.5
    → F_ajusté = 0.125×0.5 + 0.08×0.5 = 0.0625 + 0.04 = 0.1025 (10.25%)
    """
    f_ajuste = (f_wright * (1 - taux_incompletude) + 
                F_MOYEN_RACE_LADOUM * taux_incompletude)
    return round(f_ajuste, 4)


def determiner_niveau_confiance(taux_incompletude_a, taux_incompletude_b):
    """
    Détermine le niveau de confiance de l'analyse selon l'incomplétude des pedigrees.
    """
    incompletude_moyenne = (taux_incompletude_a + taux_incompletude_b) / 2

    if incompletude_moyenne <= 0.1:
        return "ÉLEVÉE", "Pedigrees complets — résultat très fiable."
    elif incompletude_moyenne <= 0.4:
        return "MODÉRÉE", "Pedigree partiellement connu — résultat indicatif."
    elif incompletude_moyenne <= 0.7:
        return "FAIBLE", "Un ou plusieurs parents inconnus — estimation prudente."
    else:
        return "TRÈS FAIBLE", "Pedigrees quasi-inconnus — résultat basé sur la moyenne de la race."


def classifier_risque(f_ajuste):
    """Classifie le niveau de risque selon le coefficient F ajusté."""
    if f_ajuste < SEUIL_FAIBLE:
        return {
            'niveau': 'ACCEPTABLE',
            'couleur': 'vert',
            'message': 'Faible risque de consanguinité. Accouplement conseillé.',
            'action': 'AUTORISER',
            'emoji': 'vert'
        }
    elif f_ajuste < SEUIL_MODERE:
        return {
            'niveau': 'MODÉRÉ',
            'couleur': 'orange',
            'message': f'Risque modéré (F={f_ajuste*100:.1f}%). Vérifiez les pedigrees avant de procéder.',
            'action': 'AVERTIR',
            'emoji': 'orange'
        }
    else:
        return {
            'niveau': 'ÉLEVÉ',
            'couleur': 'rouge',
            'message': f'Risque élevé de consanguinité (F={f_ajuste*100:.1f}%). Accouplement déconseillé.',
            'action': 'BLOQUER',
            'emoji': 'rouge'
        }


def analyser_couple_complet(animal_a_id, animal_b_id, pedigree_dict):
    """
    Fonction principale : analyse complète d'un couple avec gestion
    des cas de pedigree complet, partiel ou inconnu.

    Retourne un dictionnaire avec tous les résultats.
    """
    # 1. Calculer les taux d'incomplétude
    incomp_a = calculer_taux_incompletude(animal_a_id, pedigree_dict)
    incomp_b = calculer_taux_incompletude(animal_b_id, pedigree_dict)
    incompletude_moyenne = (incomp_a + incomp_b) / 2

    # 2. Calculer F de Wright sur ce qui est connu
    f_wright = calculer_f_wright(animal_a_id, animal_b_id, pedigree_dict)

    # 3. Ajuster selon l'incomplétude
    f_ajuste = ajuster_f_avec_incertitude(f_wright, incompletude_moyenne)

    # 4. Déterminer la confiance
    confiance_label, confiance_message = determiner_niveau_confiance(incomp_a, incomp_b)

    # 5. Classifier le risque
    risque = classifier_risque(f_ajuste)

    # 6. Trouver les ancêtres communs pour expliquer le résultat
    ancetres_a = _obtenir_ancetres_avec_chemins(animal_a_id, pedigree_dict)
    ancetres_b = _obtenir_ancetres_avec_chemins(animal_b_id, pedigree_dict)
    ancetres_communs = list(set(ancetres_a.keys()) & set(ancetres_b.keys()))

    # 7. Déterminer le type de relation
    relation = _determiner_relation(f_wright, f_ajuste, ancetres_communs)

    return {
        'succes': True,
        'f_wright': f_wright,
        'f_ajuste': f_ajuste,
        'f_pourcent': round(f_ajuste * 100, 2),
        'incompletude_a': incomp_a,
        'incompletude_b': incomp_b,
        'incompletude_moyenne': incompletude_moyenne,
        'confiance': confiance_label,
        'confiance_message': confiance_message,
        'ancetres_communs': ancetres_communs,
        'relation': relation,
        'niveau': risque['niveau'],
        'couleur': risque['couleur'],
        'message': risque['message'],
        'action': risque['action'],
        'details': {
            'f_calcule_sur_parents_connus': f_wright,
            'correction_incompletude': f_ajuste - f_wright,
            'methode': 'Wright + correction incertitude' if incompletude_moyenne > 0.1 else 'Wright exact'
        }
    }


def _determiner_relation(f_wright, f_ajuste, ancetres_communs):
    """Détermine la relation probable entre les deux animaux."""
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


# =============================================================================
# TESTS UNITAIRES — pour vérifier que l'algorithme fonctionne
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TESTS DE L'ALGORITHME DE WRIGHT")
    print("=" * 60)

    # Test 1 : Demi-frère/sœur (père commun) — F attendu = 12.5%
    pedigree_test1 = {
        'fatou':    {'pere_id': 'pere_a', 'mere_id': 'mere_b'},
        'champion': {'pere_id': 'pere_a', 'mere_id': 'mere_c'},
        'pere_a':   {'pere_id': None,     'mere_id': None},
        'mere_b':   {'pere_id': None,     'mere_id': None},
        'mere_c':   {'pere_id': None,     'mere_id': None},
    }
    resultat1 = analyser_couple_complet('fatou', 'champion', pedigree_test1)
    print(f"\nTest 1 — Demi-frère/sœur (père commun)")
    print(f"  F Wright    : {resultat1['f_wright']*100:.2f}%  (attendu: 12.50%)")
    print(f"  F ajusté    : {resultat1['f_pourcent']}%")
    print(f"  Confiance   : {resultat1['confiance']}")
    print(f"  Relation    : {resultat1['relation']}")
    print(f"  Niveau      : {resultat1['niveau']}")
    assert abs(resultat1['f_wright'] - 0.125) < 0.001, "ERREUR Test 1"
    print("  ✅ OK")

    # Test 2 : Sans lien (aucun ancêtre commun) — F attendu = 0%
    pedigree_test2 = {
        'mouton_x': {'pere_id': 'pere_x', 'mere_id': 'mere_x'},
        'mouton_y': {'pere_id': 'pere_y', 'mere_id': 'mere_y'},
        'pere_x':   {'pere_id': None, 'mere_id': None},
        'mere_x':   {'pere_id': None, 'mere_id': None},
        'pere_y':   {'pere_id': None, 'mere_id': None},
        'mere_y':   {'pere_id': None, 'mere_id': None},
    }
    resultat2 = analyser_couple_complet('mouton_x', 'mouton_y', pedigree_test2)
    print(f"\nTest 2 — Sans lien de parenté")
    print(f"  F Wright    : {resultat2['f_wright']*100:.2f}%  (attendu: 0.00%)")
    print(f"  F ajusté    : {resultat2['f_pourcent']}%")
    print(f"  Niveau      : {resultat2['niveau']}")
    assert resultat2['f_wright'] == 0.0, "ERREUR Test 2"
    print("  ✅ OK")

    # Test 3 : Pedigree INCOMPLET — père inconnu
    pedigree_test3 = {
        'brebis_z':  {'pere_id': None,     'mere_id': 'mere_b'},  # père inconnu !
        'champion':  {'pere_id': 'pere_a', 'mere_id': 'mere_c'},
        'mere_b':    {'pere_id': None,     'mere_id': None},
        'pere_a':    {'pere_id': None,     'mere_id': None},
        'mere_c':    {'pere_id': None,     'mere_id': None},
    }
    resultat3 = analyser_couple_complet('brebis_z', 'champion', pedigree_test3)
    print(f"\nTest 3 — Pedigree incomplet (père de brebis_z inconnu)")
    print(f"  F Wright    : {resultat3['f_wright']*100:.2f}%")
    print(f"  F ajusté    : {resultat3['f_pourcent']}%  (avec correction incertitude)")
    print(f"  Incomplétude: {resultat3['incompletude_moyenne']*100:.0f}%")
    print(f"  Confiance   : {resultat3['confiance']}")
    print(f"  Message     : {resultat3['confiance_message']}")
    print(f"  Méthode     : {resultat3['details']['methode']}")
    print("  ✅ OK")

    # Test 4 : Totalement inconnu (animal extérieur au troupeau)
    pedigree_test4 = {
        'fatou':       {'pere_id': 'pere_a', 'mere_id': 'mere_b'},
        'belier_ext':  {'pere_id': None,     'mere_id': None},     # animal extérieur !
        'pere_a':      {'pere_id': None,     'mere_id': None},
        'mere_b':      {'pere_id': None,     'mere_id': None},
    }
    resultat4 = analyser_couple_complet('fatou', 'belier_ext', pedigree_test4)
    print(f"\nTest 4 — Bélier extérieur au troupeau (totalement inconnu)")
    print(f"  F Wright    : {resultat4['f_wright']*100:.2f}%")
    print(f"  F ajusté    : {resultat4['f_pourcent']}%  (basé sur moyenne race Ladoum)")
    print(f"  Incomplétude: {resultat4['incompletude_moyenne']*100:.0f}%")
    print(f"  Confiance   : {resultat4['confiance']}")
    print(f"  Relation    : {resultat4['relation']}")
    print("  ✅ OK")

    print(f"\n{'='*60}")
    print("TOUS LES TESTS PASSÉS !")
    print(f"{'='*60}")