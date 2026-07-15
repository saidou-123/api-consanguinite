# -*- coding: utf-8 -*-
# =============================================================================
# API CONSANGUINITE OVINS LADOUM — v6 (Wright seul, sans ML)
# Fichier: api.py
# =============================================================================
# Changement v6 :
#   Retrait complet du Gradient Boosting (auparavant conserve a titre
#   informatif). Deux raisons :
#     1. Dataset d'entrainement synthetique juge non fiable par l'encadreur.
#     2. scikit-learn n'est pas compatible avec Python 3.14 (bloque le
#        deploiement local et potentiellement Railway selon l'image utilisee).
#   Decision et information reposent desormais UNIQUEMENT sur l'algorithme
#   de Wright applique au pedigree reel Supabase.
#
# Routes :
#   GET  /                    -> statut du serveur
#   GET  /sante                -> healthcheck Flutter
#   GET  /test                 -> test rapide Wright
#   POST /analyser-pedigree    -> Wright + pedigree Supabase  <- PRINCIPAL
# =============================================================================

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

from wright_calculator import analyser_couple_complet, classifier_risque, F_MOYEN_RACE_LADOUM
from pedigree_service import fusionner_pedigrees, extraire_noms_ancetres_communs

app = Flask(__name__)
CORS(app)


def _table_source(source):
    return 'animal_acheter' if source == 'achete' else 'nouveaux_nee'


def charger_profil_animal(supabase_client, animal_id, source):
    """
    Profil informatif uniquement (nom, sante, traits phenotypiques).
    N'influence jamais le calcul de risque, qui repose exclusivement
    sur l'algorithme de Wright applique au pedigree reel.
    """
    table = _table_source(source)
    try:
        res = (
            supabase_client.from_(table)
            .select('id,nom,pere_id,mere_id,score_sante,statut_fondateur,'
                    'couleur,taille_categorie,type_cornes,gabarit')
            .eq('id', str(animal_id))
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            animal = res.data[0]
            if animal.get('statut_fondateur') is None:
                animal['statut_fondateur'] = 1 if (
                    animal.get('pere_id') is None and animal.get('mere_id') is None
                ) else 0
            return animal
    except Exception as e:
        print(f"Avertissement profil {source}_{animal_id}: {e}")
    return {'nom': None, 'score_sante': None, 'statut_fondateur': 0,
            'couleur': None, 'taille_categorie': None,
            'type_cornes': None, 'gabarit': None}


def calculer_traits_communs(profil_a, profil_b):
    traits = ['couleur', 'taille_categorie', 'type_cornes', 'gabarit']
    return sum(
        1 for t in traits
        if profil_a.get(t) and profil_b.get(t)
        and str(profil_a[t]).lower() == str(profil_b[t]).lower()
    )


# ==================================================================
# ROUTE PRINCIPALE — Wright seul decide
# ==================================================================
@app.route('/analyser-pedigree', methods=['POST'])
def analyser_pedigree():
    try:
        d = request.get_json()
        if not d:
            return jsonify({'succes': False, 'erreur': 'Aucune donnee recue'}), 400

        brebis_id     = d.get('brebis_id')
        source_brebis = d.get('source_brebis', 'achete')
        belier_id     = d.get('belier_id')
        source_belier = d.get('source_belier', 'achete')

        if brebis_id is None or belier_id is None:
            return jsonify({'succes': False, 'erreur': 'brebis_id et belier_id requis'}), 400

        print(f"Wright: {source_brebis}_{brebis_id} x {source_belier}_{belier_id}")

        # ETAPE 1 : Connexion Supabase
        supabase_client = None
        supabase_ok = False
        try:
            from supabase import create_client
            supabase_client = create_client(
                os.environ['SUPABASE_URL'],
                os.environ['SUPABASE_KEY'],
            )
            supabase_ok = True
        except Exception as e:
            print(f"Avertissement Supabase: {e}")

        # ETAPE 2 : Profil informatif (n'influence jamais le calcul)
        if supabase_ok:
            profil_a = charger_profil_animal(supabase_client, brebis_id, source_brebis)
            profil_b = charger_profil_animal(supabase_client, belier_id, source_belier)
        else:
            profil_a = {'nom': None, 'score_sante': None, 'statut_fondateur': 0,
                        'couleur': None, 'taille_categorie': None,
                        'type_cornes': None, 'gabarit': None}
            profil_b = profil_a.copy()

        nb_traits_communs = calculer_traits_communs(profil_a, profil_b)

        # ETAPE 3 : Construire le pedigree reel depuis Supabase
        pedigree = {}
        pedigree_disponible = False
        noms_ancetres = []
        cle_a = f"{source_brebis}_{brebis_id}"
        cle_b = f"{source_belier}_{belier_id}"

        if supabase_ok:
            try:
                pedigree, cle_a, cle_b = fusionner_pedigrees(
                    supabase_client,
                    brebis_id, source_brebis,
                    belier_id, source_belier,
                )
                pedigree_disponible = True
                print(f"   Pedigree: {len(pedigree)} animaux charges")
            except Exception as e:
                print(f"Avertissement pedigree: {e}")

        # ETAPE 4 : WRIGHT — SEULE SOURCE DE DECISION
        if pedigree_disponible and len(pedigree) >= 2:
            rw = analyser_couple_complet(cle_a, cle_b, pedigree)
            noms_ancetres    = extraire_noms_ancetres_communs(pedigree, rw['ancetres_communs'])
            f_pourcent       = rw['f_pourcent']
            incompletude_moy = rw['incompletude_moyenne']
            incomp_a         = rw['incompletude_a']
            incomp_b         = rw['incompletude_b']
            confiance        = rw['confiance']
            confiance_msg    = rw['confiance_message']
            relation         = rw['relation']
            niveau           = rw['niveau']
            couleur          = rw['couleur']
            message          = rw['message']
            action           = rw['action']
            f_wright         = rw['f_wright']
            f_ajuste         = rw['f_ajuste']

            # Un seul animal totalement inconnu = analyse peu fiable,
            # meme si l'autre est parfaitement connu (moyenne trompeuse)
            belier_inconnu = incompletude_moy >= 0.9 or max(incomp_a, incomp_b) >= 0.9

            if incompletude_moy <= 0.1:
                methode = 'wright_exact'
            elif incompletude_moy <= 0.7 and max(incomp_a, incomp_b) < 0.9:
                methode = 'wright_partiel'
            else:
                methode = 'wright_moyen_race'
        else:
            # Pedigree totalement absent -> F moyen race, classe normalement
            # par les memes seuils que le reste (jamais force a ACCEPTABLE)
            risque_moyen     = classifier_risque(F_MOYEN_RACE_LADOUM)
            f_pourcent       = round(F_MOYEN_RACE_LADOUM * 100, 2)
            f_wright         = 0.0
            f_ajuste         = F_MOYEN_RACE_LADOUM
            incompletude_moy = 1.0
            confiance        = 'TRES FAIBLE'
            confiance_msg    = ('Pedigrees totalement inconnus. Resultat base sur la '
                                'moyenne estimee de la race Ladoum (F=8%). '
                                'Aucun lien de parente ne peut etre confirme ni exclu.')
            relation         = 'Inconnu — pedigree absent'
            methode          = 'wright_moyen_race'
            belier_inconnu   = True
            niveau           = risque_moyen['niveau']
            couleur          = risque_moyen['couleur']
            action           = risque_moyen['action']
            message          = ('Pedigree non disponible dans la base de donnees. '
                                'Aucun lien de parente detecte avec les informations actuelles. '
                                'Resultat base sur la moyenne de la race Ladoum (F=8%). '
                                'Pour une analyse precise, renseignez pere et mere de chaque animal.')

        # Cette note s'ajoute au message UNIQUEMENT quand aucune relation
        # concrete n'a ete trouvee (f_wright == 0). Si Wright a deja
        # confirme un lien (f_wright > 0, ex. pere-fille detecte malgre
        # des grands-parents inconnus), on ne touche JAMAIS a niveau,
        # couleur, action ou message : un lien confirme ne doit jamais
        # etre efface par une heuristique de donnees manquantes.
        if belier_inconnu and methode == 'wright_moyen_race' and f_wright == 0.0:
            message = ('Pedigree insuffisant pour un calcul fiable — calcul Wright limite. '
                       f'Estimation basee sur la moyenne race Ladoum (F=~{F_MOYEN_RACE_LADOUM*100:.0f}%). '
                       "Renseignez les parents des deux animaux pour ameliorer l'analyse.")

        print(f"   OK F={f_pourcent}% | {relation} | {niveau} | {methode}")

        return jsonify({
            'succes'               : True,
            'methode'              : methode,
            'f_pourcent'           : f_pourcent,
            'f_wright'             : round(float(f_wright), 4),
            'f_ajuste'             : round(float(f_ajuste), 4),
            'relation'             : relation,
            'ancetres_communs'     : noms_ancetres,
            'confiance'            : confiance,
            'confiance_message'    : confiance_msg,
            'incompletude_moyenne' : round(float(incompletude_moy), 2),
            'niveau'               : niveau,
            'resultat'             : niveau,
            'couleur'              : couleur,
            'message'              : message,
            'action'               : action,
            'belier_inconnu'       : belier_inconnu,
            'profil_informatif'    : {
                'nom_brebis'        : profil_a.get('nom'),
                'nom_belier'        : profil_b.get('nom'),
                'score_sante_brebis': profil_a.get('score_sante'),
                'score_sante_belier': profil_b.get('score_sante'),
                'nb_traits_communs' : nb_traits_communs,
            },
        }), 200

    except Exception as e:
        import traceback
        print(f"Erreur: {e}\n{traceback.format_exc()}")
        return jsonify({'succes': False, 'erreur': str(e)}), 500


@app.route('/', methods=['GET'])
def accueil():
    return jsonify({
        'statut' : 'en ligne', 'version': '6.0 — Wright Only (sans ML)',
        'algorithme_decisionnel': 'Wright (pedigrees Supabase)',
        'note': ('Le modele Gradient Boosting a ete retire (dataset synthetique '
                'juge non fiable, et incompatible avec Python 3.14+).'),
        'routes' : ['/analyser-pedigree', '/sante', '/test'],
    })


@app.route('/sante', methods=['GET'])
def sante():
    return jsonify({'statut': 'ok', 'version': '6.0', 'wright_actif': True})


@app.route('/test', methods=['GET'])
def test():
    pedigree_test = {
        'test_fatou'   : {'pere_id': 'test_baba', 'mere_id': 'test_mere1', 'nom': 'Fatou'},
        'test_champion': {'pere_id': 'test_baba', 'mere_id': 'test_mere2', 'nom': 'Champion'},
        'test_baba'    : {'pere_id': None, 'mere_id': None, 'nom': 'Baba'},
        'test_mere1'   : {'pere_id': None, 'mere_id': None, 'nom': 'Mere_Fatou'},
        'test_mere2'   : {'pere_id': None, 'mere_id': None, 'nom': 'Mere_Champion'},
    }
    res = analyser_couple_complet('test_fatou', 'test_champion', pedigree_test)
    return jsonify({
        'test'      : 'Fatou x Champion (pere commun: Baba)',
        'algorithme': 'Wright — seul decisionnel',
        'f_pourcent': res['f_pourcent'],
        'relation'  : res['relation'],
        'ancetres'  : res['ancetres_communs'],
        'niveau'    : res['niveau'],
        'f_attendu' : '12.50%',
        'ok'        : abs(res['f_wright'] - 0.125) < 0.001,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("API CONSANGUINITE v6 — WRIGHT SEUL (sans ML)")
    print("="*60)
    print(f"http://localhost:{port}")
    print(f"  SUPABASE_URL = {os.environ.get('SUPABASE_URL','NON DEFINIE')[:40]}")
    print(f"  SUPABASE_KEY = {'definie' if os.environ.get('SUPABASE_KEY') else 'NON DEFINIE'}")
    print("="*60+"\n")
    app.run(host='0.0.0.0', port=port, debug=False)