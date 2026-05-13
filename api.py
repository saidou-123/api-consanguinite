# =============================================================================
# API CONSANGUINITÉ OVINS LADOUM — v3
# Fichier: api.py
# =============================================================================
# Routes :
#   GET  /                    → statut du serveur
#   GET  /sante               → healthcheck Flutter
#   GET  /test                → test rapide ML
#   POST /predire             → prédiction ML seule (legacy)
#   POST /analyser-pedigree   → Wright + ML + pedigree Supabase  ← NOUVEAU
# =============================================================================
from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import warnings
warnings.filterwarnings('ignore')

# Wright algorithm (fichier local)
from wright_calculator import analyser_couple_complet, F_MOYEN_RACE_LADOUM

# Service pedigree Supabase
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


# ==================================================================
# ROUTE PRINCIPALE — Analyse pedigree complète (Wright + ML)
# ==================================================================
@app.route('/analyser-pedigree', methods=['POST'])
def analyser_pedigree():
    """
    Corps JSON attendu :
    {
        "brebis_id"    : 12,
        "source_brebis": "achete",   // "achete" ou "nee"
        "belier_id"    : 7,
        "source_belier": "nee"
    }

    Réponse JSON :
    {
        "succes"          : true,
        "methode"         : "wright_exact" | "wright_partiel" | "ml_seul",
        "f_pourcent"      : 11.4,
        "f_wright"        : 0.125,
        "f_ajuste"        : 0.114,
        "relation"        : "Demi-frère/sœur",
        "ancetres_communs": ["Baba"],
        "confiance"       : "MODÉRÉE",
        "confiance_message": "Pedigree partiellement connu...",
        "incompletude_moyenne": 0.25,
        "niveau"          : "MODÉRÉ",    // ACCEPTABLE | MODÉRÉ | ÉLEVÉ
        "couleur"         : "orange",    // vert | orange | rouge
        "message"         : "...",
        "action"          : "AVERTIR",   // AUTORISER | AVERTIR | BLOQUER
        // Résultat ML en complément :
        "ml_resultat"     : "RISQUE",
        "ml_confiance_risque"     : 0.72,
        "ml_confiance_acceptable" : 0.28,
        // Cas inconnu :
        "belier_inconnu"  : false
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

        # ----------------------------------------------------------
        # ÉTAPE 1 : Construire le pedigree depuis Supabase
        # ----------------------------------------------------------
        print(f"🔍 Analyse pedigree: {source_brebis}_{brebis_id} × {source_belier}_{belier_id}")

        try:
            from pedigree_service import fusionner_pedigrees, extraire_noms_ancetres_communs
            from supabase import create_client
            supabase_client = create_client(
                os.environ['SUPABASE_URL'],
                os.environ['SUPABASE_KEY'],
            )
            pedigree, cle_a, cle_b = fusionner_pedigrees(
                supabase_client,
                brebis_id, source_brebis,
                belier_id, source_belier,
            )
            pedigree_disponible = True
            print(f"   Pedigree construit: {len(pedigree)} animaux chargés")
        except Exception as e:
            print(f"⚠️  Pedigree non disponible: {e}")
            pedigree = {}
            cle_a = f"{source_brebis}_{brebis_id}"
            cle_b = f"{source_belier}_{belier_id}"
            pedigree_disponible = False

        # ----------------------------------------------------------
        # ÉTAPE 2 : Calcul Wright
        # ----------------------------------------------------------
        if pedigree_disponible and len(pedigree) >= 2:
            resultat_wright = analyser_couple_complet(cle_a, cle_b, pedigree)

            # Noms lisibles des ancêtres communs
            noms_ancetres = extraire_noms_ancetres_communs(
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

            # Méthode utilisée
            if incompletude_moy <= 0.1:
                methode = 'wright_exact'
            elif incompletude_moy <= 0.7:
                methode = 'wright_partiel'
            else:
                methode = 'ml_seul'

            belier_inconnu = incompletude_moy >= 0.9

        else:
            # Pas de pedigree du tout → on utilise les moyennes de la race
            f_pourcent       = F_MOYEN_RACE_LADOUM * 100
            f_wright         = 0.0
            f_ajuste         = F_MOYEN_RACE_LADOUM
            incompletude_moy = 1.0
            confiance        = 'TRÈS FAIBLE'
            confiance_msg    = 'Pedigrees inconnus — estimation basée sur la moyenne de la race Ladoum.'
            relation         = 'Inconnu'
            noms_ancetres    = []
            methode          = 'ml_seul'
            belier_inconnu   = True
            niveau           = 'ACCEPTABLE'
            couleur          = 'vert'
            message          = ("Pedigree non disponible. Aucun lien détecté sur la base des données disponibles. "
                                "Renseignez les parents de l'animal pour une analyse précise.")
            action           = 'AUTORISER'

        # ----------------------------------------------------------
        # ÉTAPE 3 : Compléter avec la prédiction ML
        # ----------------------------------------------------------
        ml_resultat             = 'INCONNU'
        ml_confiance_risque     = 0.0
        ml_confiance_acceptable = 0.0

        try:
            # APRÈS — corrigé
            vecteur_ml = [[
                max(1 - f_ajuste * 4, 0.1),         # Distance_Genetique_Estimee
                max(1 - incompletude_moy, 0.3),     # Diversite_Allelique
                min(f_ajuste * 3, 0.9),             # Similarite_Phenotypique
                0.75,                               # Taux_Reussite_Reproduction
                0, 0,                               # Statut_Fondateur_A, Statut_Fondateur_B
                incompletude_moy,                   # Taux_Incompletude_A
                incompletude_moy,                   # Taux_Incompletude_B
                0.80, 0.80,                         # Score_Sante_A, Score_Sante_B
                max(1 - incompletude_moy, 0.1),    # Niveau_Confiance
                len(noms_ancetres),                 # Nb_Traits_Communs
            ]]
            vecteur_norme       = scaler.transform(vecteur_ml)
            probas_ml           = modele.predict_proba(vecteur_norme)[0]
            # Seuil abaissé à 0.40 (au lieu de 0.50 par défaut) pour réduire
            # les faux négatifs : mieux vaut signaler un risque qui n'en est
            # pas un, que de manquer un vrai cas de consanguinité.
            SEUIL_ML = 0.40
            pred_ml             = 1 if probas_ml[1] >= SEUIL_ML else 0
            ml_resultat         = 'RISQUE' if pred_ml == 1 else 'ACCEPTABLE'
            ml_confiance_risque = round(float(probas_ml[1]), 3)
            ml_confiance_acceptable = round(float(probas_ml[0]), 3)
        except Exception as e:
            print(f"⚠️  Prédiction ML échouée: {e}")

        # ----------------------------------------------------------
        # ÉTAPE 4 : Message spécial si bélier extérieur inconnu
        # ----------------------------------------------------------
        if belier_inconnu:
            message  = ("Pedigree du bélier inconnu. Résultat probablement acceptable, "
                        "mais le niveau de confiance est très faible. "
                        "Renseignez les parents du bélier pour améliorer l'analyse.")
            niveau   = 'ACCEPTABLE'
            couleur  = 'vert'
            action   = 'AUTORISER'

        print(f"   ✅ Résultat: F={f_pourcent}% | {relation} | {niveau} ({methode})")

        return jsonify({
            'succes'              : True,
            'methode'             : methode,
            'f_pourcent'          : f_pourcent,
            'f_wright'            : round(float(f_wright), 4),
            'f_ajuste'            : round(float(f_ajuste), 4),
            'relation'            : relation,
            'ancetres_communs'    : noms_ancetres,
            'confiance'           : confiance,
            'confiance_message'   : confiance_msg,
            'incompletude_moyenne': round(float(incompletude_moy), 2),
            'niveau'              : niveau,   # clé principale
            'resultat'            : niveau,   # alias pour compatibilité Flutter
            'couleur'             : couleur,
            'message'             : message,
            'action'              : action,
            'belier_inconnu'      : belier_inconnu,
            'ml_resultat'         : ml_resultat,
            'ml_confiance_risque' : ml_confiance_risque,
            'ml_confiance_acceptable': ml_confiance_acceptable,
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
                'succes': True, 'resultat': 'ACCEPTABLE', 'couleur': 'vert',
                'message': 'Accouplement conseillé. Bonne diversité génétique attendue.',
                'action': 'AUTORISER',
                'confiance_acceptable': round(float(probas[0]), 3),
                'confiance_risque': round(float(probas[1]), 3),
            }), 200
        else:
            return jsonify({
                'succes': True, 'resultat': 'RISQUE', 'couleur': 'rouge',
                'message': 'Accouplement déconseillé. Risque de consanguinité élevé.',
                'action': 'BLOQUER',
                'confiance_acceptable': round(float(probas[0]), 3),
                'confiance_risque': round(float(probas[1]), 3),
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
        'version'  : '3.0',
        'modele'   : 'Gradient Boosting v2 + Wright Algorithm',
        'precision': '69.1% (ML) | 95%+ (Wright complet)',
        'routes'   : ['/analyser-pedigree', '/predire', '/sante', '/test'],
    })


@app.route('/sante', methods=['GET'])
def sante():
    return jsonify({'statut': 'ok', 'modele_charge': True})


@app.route('/test', methods=['GET'])
def test():
    """Test rapide avec un couple demi-frère/sœur simulé."""
    pedigree_test = {
        'test_fatou':    {'pere_id': 'test_baba', 'mere_id': 'test_mere1', 'nom': 'Fatou'},
        'test_champion': {'pere_id': 'test_baba', 'mere_id': 'test_mere2', 'nom': 'Champion'},
        'test_baba':     {'pere_id': None,         'mere_id': None,          'nom': 'Baba'},
        'test_mere1':    {'pere_id': None,         'mere_id': None,          'nom': 'Mere_Fatou'},
        'test_mere2':    {'pere_id': None,         'mere_id': None,          'nom': 'Mere_Champion'},
    }
    res = analyser_couple_complet('test_fatou', 'test_champion', pedigree_test)
    return jsonify({
        'test'          : 'Fatou × Champion (père commun: Baba)',
        'f_pourcent'    : res['f_pourcent'],
        'relation'      : res['relation'],
        'ancetres'      : res['ancetres_communs'],
        'niveau'        : res['niveau'],
        'f_attendu'     : '12.50%',
        'ok'            : abs(res['f_wright'] - 0.125) < 0.001,
    })


# ==================================================================
# DÉMARRAGE
# ==================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 API CONSANGUINITÉ OVINS v3 — DÉMARRAGE")
    print("=" * 60)
    print("📡 Serveur          : http://localhost:5000")
    print("🧬 Pedigree + Wright: POST /analyser-pedigree")
    print("🤖 ML seul (legacy) : POST /predire")
    print("🔍 Santé            : GET  /sante")
    print("🧪 Test Wright      : GET  /test")
    print("=" * 60)
    print("Variables d'environnement requises:")
    print(f"  SUPABASE_URL = {os.environ.get('SUPABASE_URL', '❌ NON DÉFINIE')[:40]}")
    print(f"  SUPABASE_KEY = {'✅ définie' if os.environ.get('SUPABASE_KEY') else '❌ NON DÉFINIE'}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)