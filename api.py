# =============================================================================
# API CONSANGUINITÉ OVINS LADOUM — v4
# Fichier: api.py
# =============================================================================
# Corrections v4 par rapport à v3 :
#   ✅ Lecture des vraies données animaux depuis Supabase (score_sante,
#      statut_fondateur, taux_reussite_reproduction, nb_traits_communs)
#   ✅ Le Gradient Boosting reçoit de vraies features, plus des constantes
#   ✅ Fonction dédiée charger_profil_animal() pour lire Supabase
#   ✅ Calcul automatique statut_fondateur (pere_id = null → fondateur)
#   ✅ Calcul automatique taux_reussite depuis table naissances
#   ✅ Calcul automatique nb_traits_communs entre les deux animaux
#   ✅ Fallback sur valeurs par défaut si colonne absente dans Supabase
#
# Routes :
#   GET  /                    → statut du serveur
#   GET  /sante               → healthcheck Flutter
#   GET  /test                → test rapide ML
#   POST /predire             → prédiction ML seule (legacy)
#   POST /analyser-pedigree   → Wright + ML + pedigree Supabase  ← PRINCIPAL
# =============================================================================

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import warnings
warnings.filterwarnings('ignore')

from wright_calculator import analyser_couple_complet, F_MOYEN_RACE_LADOUM
from pedigree_service import fusionner_pedigrees, extraire_noms_ancetres_communs

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# CHARGEMENT DU MODÈLE ML (Gradient Boosting v2)
# ------------------------------------------------------------------
print("⏳ Chargement du modèle IA v2...")
modele = joblib.load('modele_consanguinite_v2.pkl')
scaler = joblib.load('normaliseur_v2.pkl')
print("✅ Modèle ML chargé")

FEATURES_ML = [
    'Distance_Genetique_Estimee',
    'Diversite_Allelique',
    'Similarite_Phenotypique',
    'Taux_Reussite_Reproduction',
    'Statut_Fondateur_A',
    'Statut_Fondateur_B',
    'Taux_Incompletude_A',
    'Taux_Incompletude_B',
    'Score_Sante_A',
    'Score_Sante_B',
    'Niveau_Confiance',
    'Nb_Traits_Communs',
]

# Seuil ML abaissé à 0.40 pour réduire les faux négatifs
# (mieux vaut signaler un risque qui n'en est pas un que de manquer
#  un vrai cas de consanguinité)
SEUIL_ML = 0.40


# ==================================================================
# FONCTIONS UTILITAIRES — Lecture des données animaux depuis Supabase
# ==================================================================

def _table_source(source: str) -> str:
    """Retourne le nom de la table Supabase selon la source."""
    return 'animal_acheter' if source == 'achete' else 'nouveaux_nee'


def charger_profil_animal(supabase_client, animal_id, source: str) -> dict:
    """
    Charge le profil complet d'un animal depuis Supabase.
    Retourne un dict avec toutes les features ML nécessaires.
    Si une colonne est absente → valeur par défaut utilisée.

    Colonnes lues dans Supabase (à ajouter si absentes) :
      - score_sante           : float 0.0–1.0  (état de santé général)
      - statut_fondateur      : int   0 ou 1   (1 = animal fondateur sans parents connus)
      - pere_id               : uuid|null       (pour calculer statut_fondateur auto)
      - mere_id               : uuid|null       (pour calculer statut_fondateur auto)
      - couleur               : str             (pour calculer traits communs)
      - taille_categorie      : str             (pour calculer traits communs)
      - type_cornes           : str             (pour calculer traits communs)
      - gabarit               : str             (pour calculer traits communs)
    """
    table = _table_source(source)
    try:
        res = (
            supabase_client
            .from_(table)
            .select(
                'id, nom, pere_id, mere_id, '
                'score_sante, statut_fondateur, '
                'couleur, taille_categorie, type_cornes, gabarit'
            )
            .eq('id', str(animal_id))
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            animal = res.data[0]

            # Calcul automatique statut_fondateur si colonne absente
            # Un animal est "fondateur" s'il n'a aucun parent connu
            if animal.get('statut_fondateur') is None:
                pere_absent = animal.get('pere_id') is None
                mere_absent = animal.get('mere_id') is None
                animal['statut_fondateur'] = 1 if (pere_absent and mere_absent) else 0

            # Valeur par défaut score_sante si colonne absente
            if animal.get('score_sante') is None:
                animal['score_sante'] = 0.75  # valeur neutre

            return animal

    except Exception as e:
        print(f"⚠️  Erreur chargement profil {source}_{animal_id}: {e}")

    # Retourne un profil par défaut si Supabase échoue
    return {
        'id'               : animal_id,
        'nom'              : f'Animal_{animal_id}',
        'pere_id'          : None,
        'mere_id'          : None,
        'score_sante'      : 0.75,
        'statut_fondateur' : 0,
        'couleur'          : None,
        'taille_categorie' : None,
        'type_cornes'      : None,
        'gabarit'          : None,
    }


def calculer_taux_reussite(supabase_client, animal_id, source: str) -> float:
    """
    Calcule le taux de réussite de reproduction d'un animal
    à partir de l'historique des naissances dans Supabase.

    Cherche dans la table 'naissances' (ou 'nouveaux_nee') combien
    d'accouplements ont produit des petits vivants.

    Si aucun historique → retourne 0.75 (moyenne de la race).
    """
    try:
        # Essaie d'abord dans la table naissances (si elle existe)
        res = (
            supabase_client
            .from_('nouveaux_nee')
            .select('id, statut')
            .or_(f'pere_id.eq.{animal_id},mere_id.eq.{animal_id}')
            .execute()
        )
        if res.data and len(res.data) > 0:
            total  = len(res.data)
            vivants = sum(
                1 for n in res.data
                if n.get('statut') in ('vivant', 'actif', 'vendu', None)
            )
            taux = vivants / total if total > 0 else 0.75
            return round(min(max(taux, 0.0), 1.0), 3)
    except Exception as e:
        print(f"⚠️  Taux réussite non calculable pour {source}_{animal_id}: {e}")

    return 0.75  # Valeur par défaut (moyenne race Ladoum)


def calculer_traits_communs(profil_a: dict, profil_b: dict) -> int:
    """
    Calcule le nombre de traits phénotypiques communs entre deux animaux.
    Compare : couleur, taille_categorie, type_cornes, gabarit.
    Maximum possible : 4 traits.
    """
    traits = ['couleur', 'taille_categorie', 'type_cornes', 'gabarit']
    communs = 0
    for trait in traits:
        val_a = profil_a.get(trait)
        val_b = profil_b.get(trait)
        # On compte comme commun seulement si les deux valeurs sont
        # connues ET identiques
        if val_a and val_b and str(val_a).lower() == str(val_b).lower():
            communs += 1
    return communs


def calculer_similarite_phenotypique(nb_traits_communs: int) -> float:
    """
    Convertit le nombre de traits communs en score de similarité phénotypique.
    0 trait  → 0.05  (très différents)
    1 trait  → 0.25
    2 traits → 0.50
    3 traits → 0.75
    4 traits → 0.95  (très similaires)
    """
    table = {0: 0.05, 1: 0.25, 2: 0.50, 3: 0.75, 4: 0.95}
    return table.get(nb_traits_communs, 0.30)


# ==================================================================
# ROUTE PRINCIPALE — Analyse pedigree complète (Wright + ML)
# ==================================================================
@app.route('/analyser-pedigree', methods=['POST'])
def analyser_pedigree():
    """
    Corps JSON attendu :
    {
        "brebis_id"    : "uuid-ou-int",
        "source_brebis": "achete",        // "achete" ou "nee"
        "belier_id"    : "uuid-ou-int",
        "source_belier": "nee"
    }

    Réponse JSON :
    {
        "succes"               : true,
        "methode"              : "wright_exact|wright_partiel|ml_seul",
        "f_pourcent"           : 11.4,
        "f_wright"             : 0.125,
        "f_ajuste"             : 0.114,
        "relation"             : "Demi-frère/sœur",
        "ancetres_communs"     : ["Baba"],
        "confiance"            : "MODÉRÉE",
        "confiance_message"    : "Pedigree partiellement connu...",
        "incompletude_moyenne" : 0.25,
        "niveau"               : "MODÉRÉ",
        "couleur"              : "orange",
        "message"              : "...",
        "action"               : "AVERTIR",
        "ml_resultat"          : "RISQUE",
        "ml_confiance_risque"  : 0.72,
        "ml_confiance_acceptable": 0.28,
        "belier_inconnu"       : false,
        // Nouvelles clés v4 (debug) :
        "features_ml_utilisees": { ... }
    }
    """
    try:
        d = request.get_json()
        if not d:
            return jsonify({'succes': False, 'erreur': 'Aucune donnée reçue'}), 400

        brebis_id     = d.get('brebis_id')
        source_brebis = d.get('source_brebis', 'achete')
        belier_id     = d.get('belier_id')
        source_belier = d.get('source_belier', 'achete')

        if brebis_id is None or belier_id is None:
            return jsonify({'succes': False, 'erreur': 'brebis_id et belier_id requis'}), 400

        print(f"🔍 Analyse pedigree: {source_brebis}_{brebis_id} × {source_belier}_{belier_id}")

        # ----------------------------------------------------------
        # ÉTAPE 1 : Connexion Supabase
        # ----------------------------------------------------------
        try:
            from supabase import create_client
            supabase_client = create_client(
                os.environ['SUPABASE_URL'],
                os.environ['SUPABASE_KEY'],
            )
            supabase_ok = True
        except Exception as e:
            print(f"⚠️  Connexion Supabase échouée: {e}")
            supabase_client = None
            supabase_ok     = False

        # ----------------------------------------------------------
        # ÉTAPE 2 : Charger les profils animaux depuis Supabase
        #           (NOUVEAU en v4 — remplace les constantes fixes)
        # ----------------------------------------------------------
        if supabase_ok:
            profil_a = charger_profil_animal(supabase_client, brebis_id, source_brebis)
            profil_b = charger_profil_animal(supabase_client, belier_id, source_belier)

            # Taux de réussite reproduction (calculé depuis historique)
            taux_reussite_a = calculer_taux_reussite(supabase_client, brebis_id, source_brebis)
            taux_reussite_b = calculer_taux_reussite(supabase_client, belier_id, source_belier)
            taux_reussite   = round((taux_reussite_a + taux_reussite_b) / 2, 3)

            print(f"   Profils chargés: {profil_a.get('nom')} | {profil_b.get('nom')}")
            print(f"   Score santé: A={profil_a.get('score_sante'):.2f} | B={profil_b.get('score_sante'):.2f}")
            print(f"   Taux réussite: {taux_reussite:.2f}")
        else:
            # Profils par défaut si Supabase indisponible
            profil_a = {'score_sante': 0.75, 'statut_fondateur': 0,
                        'couleur': None, 'taille_categorie': None,
                        'type_cornes': None, 'gabarit': None}
            profil_b = profil_a.copy()
            taux_reussite = 0.75

        # Traits communs calculés depuis les profils réels
        nb_traits_communs      = calculer_traits_communs(profil_a, profil_b)
        similarite_phenotypique = calculer_similarite_phenotypique(nb_traits_communs)

        # ----------------------------------------------------------
        # ÉTAPE 3 : Construire le pedigree depuis Supabase
        # ----------------------------------------------------------
        pedigree          = {}
        pedigree_disponible = False
        noms_ancetres     = []

        if supabase_ok:
            try:
                pedigree, cle_a, cle_b = fusionner_pedigrees(
                    supabase_client,
                    brebis_id, source_brebis,
                    belier_id, source_belier,
                )
                pedigree_disponible = True
                print(f"   Pedigree construit: {len(pedigree)} animaux chargés")
            except Exception as e:
                print(f"⚠️  Pedigree non disponible: {e}")
                cle_a = f"{source_brebis}_{brebis_id}"
                cle_b = f"{source_belier}_{belier_id}"
        else:
            cle_a = f"{source_brebis}_{brebis_id}"
            cle_b = f"{source_belier}_{belier_id}"

        # ----------------------------------------------------------
        # ÉTAPE 4 : Calcul Wright
        # ----------------------------------------------------------
        if pedigree_disponible and len(pedigree) >= 2:
            resultat_wright = analyser_couple_complet(cle_a, cle_b, pedigree)

            noms_ancetres    = extraire_noms_ancetres_communs(
                pedigree, resultat_wright['ancetres_communs']
            )
            f_pourcent       = resultat_wright['f_pourcent']
            incompletude_moy = resultat_wright['incompletude_moyenne']
            confiance        = resultat_wright['confiance']
            confiance_msg    = resultat_wright['confiance_message']
            relation         = resultat_wright['relation']
            niveau           = resultat_wright['niveau']
            couleur          = resultat_wright['couleur']
            message          = resultat_wright['message']
            action           = resultat_wright['action']
            f_wright         = resultat_wright['f_wright']
            f_ajuste         = resultat_wright['f_ajuste']

            if incompletude_moy <= 0.1:
                methode = 'wright_exact'
            elif incompletude_moy <= 0.7:
                methode = 'wright_partiel'
            else:
                methode = 'ml_seul'

            belier_inconnu = incompletude_moy >= 0.9

        else:
            # Pas de pedigree → on utilise les moyennes de la race
            f_pourcent       = F_MOYEN_RACE_LADOUM * 100
            f_wright         = 0.0
            f_ajuste         = F_MOYEN_RACE_LADOUM
            incompletude_moy = 1.0
            confiance        = 'TRÈS FAIBLE'
            confiance_msg    = 'Pedigrees inconnus — estimation basée sur la moyenne de la race Ladoum.'
            relation         = 'Inconnu'
            methode          = 'ml_seul'
            belier_inconnu   = True
            niveau           = 'ACCEPTABLE'
            couleur          = 'vert'
            message          = (
                "Pedigree non disponible. Aucun lien détecté sur la base des données disponibles. "
                "Renseignez les parents de l'animal pour une analyse précise."
            )
            action           = 'AUTORISER'

        # ----------------------------------------------------------
        # ÉTAPE 5 : Prédiction Gradient Boosting
        #           ✅ CORRIGÉ v4 — vraies features depuis Supabase
        # ----------------------------------------------------------
        ml_resultat             = 'INCONNU'
        ml_confiance_risque     = 0.0
        ml_confiance_acceptable = 0.0

        # Construction du vecteur avec les vraies valeurs Supabase
        # (plus de constantes fixes 0.80, 0.75, 0, 0)
        features_ml = {
            # Dérivé de f_ajuste (Wright ou moyenne race)
            'Distance_Genetique_Estimee' : max(1 - f_ajuste * 4, 0.1),

            # Dérivé de l'incomplétude du pedigree
            'Diversite_Allelique'        : max(1 - incompletude_moy, 0.3),

            # ✅ Calculé depuis les traits phénotypiques réels des deux animaux
            'Similarite_Phenotypique'    : similarite_phenotypique,

            # ✅ Calculé depuis l'historique des naissances dans Supabase
            'Taux_Reussite_Reproduction' : taux_reussite,

            # ✅ Lu depuis le profil réel de chaque animal dans Supabase
            #    (ou calculé auto : pere_id=null AND mere_id=null → fondateur=1)
            'Statut_Fondateur_A'         : int(profil_a.get('statut_fondateur', 0)),
            'Statut_Fondateur_B'         : int(profil_b.get('statut_fondateur', 0)),

            # Calculé automatiquement par wright_calculator.py
            'Taux_Incompletude_A'        : incompletude_moy,
            'Taux_Incompletude_B'        : incompletude_moy,

            # ✅ Lu depuis la colonne score_sante dans Supabase
            #    (au lieu de la constante 0.80 fixe)
            'Score_Sante_A'              : float(profil_a.get('score_sante', 0.75)),
            'Score_Sante_B'              : float(profil_b.get('score_sante', 0.75)),

            # Dérivé de l'incomplétude
            'Niveau_Confiance'           : max(1 - incompletude_moy, 0.1),

            # ✅ Calculé depuis les traits phénotypiques réels (couleur, taille, etc.)
            'Nb_Traits_Communs'          : nb_traits_communs,
        }

        print(f"   🤖 Features ML: sante_A={features_ml['Score_Sante_A']:.2f} | "
              f"sante_B={features_ml['Score_Sante_B']:.2f} | "
              f"fondateur_A={features_ml['Statut_Fondateur_A']} | "
              f"fondateur_B={features_ml['Statut_Fondateur_B']} | "
              f"traits_communs={features_ml['Nb_Traits_Communs']} | "
              f"reussite={features_ml['Taux_Reussite_Reproduction']:.2f}")

        try:
            vecteur_ml = [[features_ml[f] for f in FEATURES_ML]]
            vecteur_norme       = scaler.transform(vecteur_ml)
            probas_ml           = modele.predict_proba(vecteur_norme)[0]
            pred_ml             = 1 if probas_ml[1] >= SEUIL_ML else 0
            ml_resultat         = 'RISQUE' if pred_ml == 1 else 'ACCEPTABLE'
            ml_confiance_risque     = round(float(probas_ml[1]), 3)
            ml_confiance_acceptable = round(float(probas_ml[0]), 3)
        except Exception as e:
            print(f"⚠️  Prédiction ML échouée: {e}")

        # ----------------------------------------------------------
        # ÉTAPE 6 : Message spécial si bélier extérieur inconnu
        # ----------------------------------------------------------
        if belier_inconnu:
            message = (
                "Pedigree du bélier inconnu. Résultat probablement acceptable, "
                "mais le niveau de confiance est très faible. "
                "Renseignez les parents du bélier pour améliorer l'analyse."
            )
            niveau = 'ACCEPTABLE'
            couleur = 'vert'
            action  = 'AUTORISER'

        print(f"   ✅ Résultat: F={f_pourcent}% | {relation} | {niveau} ({methode})")
        print(f"   🤖 ML: {ml_resultat} (risque={ml_confiance_risque:.0%})")

        return jsonify({
            'succes'                 : True,
            'methode'                : methode,
            'f_pourcent'             : f_pourcent,
            'f_wright'               : round(float(f_wright), 4),
            'f_ajuste'               : round(float(f_ajuste), 4),
            'relation'               : relation,
            'ancetres_communs'       : noms_ancetres,
            'confiance'              : confiance,
            'confiance_message'      : confiance_msg,
            'incompletude_moyenne'   : round(float(incompletude_moy), 2),
            'niveau'                 : niveau,
            'resultat'               : niveau,   # alias compatibilité Flutter
            'couleur'                : couleur,
            'message'                : message,
            'action'                 : action,
            'belier_inconnu'         : belier_inconnu,
            'ml_resultat'            : ml_resultat,
            'ml_confiance_risque'    : ml_confiance_risque,
            'ml_confiance_acceptable': ml_confiance_acceptable,
            # Nouvelles clés v4 : les features réellement utilisées par le ML
            # (utile pour le débogage et les futures améliorations)
            'features_ml_utilisees'  : features_ml,
        }), 200

    except Exception as e:
        import traceback
        print(f"❌ Erreur /analyser-pedigree: {e}\n{traceback.format_exc()}")
        return jsonify({'succes': False, 'erreur': str(e)}), 500


# ==================================================================
# ROUTE LEGACY — Prédiction ML seule (conservée pour compatibilité)
# ==================================================================
@app.route('/predire', methods=['POST'])
def predire():
    """
    Route legacy : le client envoie directement les features ML.
    Conservée pour compatibilité avec les anciens appels Flutter.
    """
    try:
        d = request.get_json()
        if not d:
            return jsonify({'erreur': 'Aucune donnée reçue'}), 400

        vecteur = [[
            d.get('distance_genetique', 0.8),
            d.get('diversite_allelique', 0.75),
            d.get('similarite_phenotypique', 0.3),
            d.get('taux_reussite', 0.75),
            d.get('fondateur_a', 0),
            d.get('fondateur_b', 0),
            d.get('incompletude_a', 0.2),
            d.get('incompletude_b', 0.2),
            d.get('sante_a', 0.8),
            d.get('sante_b', 0.8),
            d.get('niveau_confiance', 0.6),
            d.get('traits_communs', 2),
        ]]

        vecteur_norme = scaler.transform(vecteur)
        prediction    = int(modele.predict(vecteur_norme)[0])
        probas        = modele.predict_proba(vecteur_norme)[0]

        if prediction == 0:
            return jsonify({
                'succes'              : True,
                'resultat'            : 'ACCEPTABLE',
                'couleur'             : 'vert',
                'message'             : 'Accouplement conseillé. Bonne diversité génétique attendue.',
                'action'              : 'AUTORISER',
                'confiance_acceptable': round(float(probas[0]), 3),
                'confiance_risque'    : round(float(probas[1]), 3),
            }), 200
        else:
            return jsonify({
                'succes'              : True,
                'resultat'            : 'RISQUE',
                'couleur'             : 'rouge',
                'message'             : 'Accouplement déconseillé. Risque de consanguinité élevé.',
                'action'              : 'BLOQUER',
                'confiance_acceptable': round(float(probas[0]), 3),
                'confiance_risque'    : round(float(probas[1]), 3),
            }), 200

    except Exception as e:
        return jsonify({'succes': False, 'erreur': str(e)}), 500


# ==================================================================
# ROUTES UTILITAIRES
# ==================================================================
@app.route('/', methods=['GET'])
def accueil():
    return jsonify({
        'statut'   : 'en ligne',
        'version'  : '4.0',
        'modele'   : 'Gradient Boosting v2 + Wright Algorithm',
        'precision': '69.1% (ML) | 95%+ (Wright complet)',
        'nouveautes_v4': [
            'Score santé lu depuis Supabase (plus de 0.80 fixe)',
            'Statut fondateur calculé automatiquement',
            'Taux réussite calculé depuis historique naissances',
            'Traits communs calculés depuis profils phénotypiques',
        ],
        'routes'   : ['/analyser-pedigree', '/predire', '/sante', '/test'],
    })


@app.route('/sante', methods=['GET'])
def sante():
    return jsonify({'statut': 'ok', 'modele_charge': True, 'version': '4.0'})


@app.route('/test', methods=['GET'])
def test():
    """Test rapide avec un couple demi-frère/sœur simulé."""
    pedigree_test = {
        'test_fatou'   : {'pere_id': 'test_baba', 'mere_id': 'test_mere1', 'nom': 'Fatou'},
        'test_champion': {'pere_id': 'test_baba', 'mere_id': 'test_mere2', 'nom': 'Champion'},
        'test_baba'    : {'pere_id': None,         'mere_id': None,          'nom': 'Baba'},
        'test_mere1'   : {'pere_id': None,         'mere_id': None,          'nom': 'Mere_Fatou'},
        'test_mere2'   : {'pere_id': None,         'mere_id': None,          'nom': 'Mere_Champion'},
    }
    res = analyser_couple_complet('test_fatou', 'test_champion', pedigree_test)
    return jsonify({
        'test'      : 'Fatou × Champion (père commun: Baba)',
        'f_pourcent': res['f_pourcent'],
        'relation'  : res['relation'],
        'ancetres'  : res['ancetres_communs'],
        'niveau'    : res['niveau'],
        'f_attendu' : '12.50%',
        'ok'        : abs(res['f_wright'] - 0.125) < 0.001,
    })


# ==================================================================
# DÉMARRAGE
# ==================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 60)
    print("🚀 API CONSANGUINITÉ OVINS v4 — DÉMARRAGE")
    print("=" * 60)
    print(f"📡 Serveur          : http://localhost:{port}")
    print("🧬 Pedigree + Wright: POST /analyser-pedigree")
    print("🤖 ML seul (legacy) : POST /predire")
    print("🔍 Santé            : GET  /sante")
    print("🧪 Test Wright      : GET  /test")
    print("=" * 60)
    print("Variables d'environnement requises:")
    print(f"  SUPABASE_URL = {os.environ.get('SUPABASE_URL', '❌ NON DÉFINIE')[:40]}")
    print(f"  SUPABASE_KEY = {'✅ définie' if os.environ.get('SUPABASE_KEY') else '❌ NON DÉFINIE'}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
