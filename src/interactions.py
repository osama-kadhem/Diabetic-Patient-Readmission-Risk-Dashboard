"""Drug-drug interaction checker using the OpenFDA drug/label API."""
import requests
import warnings
import streamlit as st

@st.cache_data(show_spinner=False, ttl=3600)
def check_drug_interactions(medications: dict) -> list[dict]:
    """
    Cross-references the patient's active diabetes medications against the
    OpenFDA drug/label API to detect registered contraindications.
    """
    active_drugs = []
    valid_fda_drugs = ['insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
    
    for k, v in medications.items():
        if k.lower() in valid_fda_drugs and str(v).lower() not in ['no', 'none', 'false', '']:
            active_drugs.append(k.lower())
            
    alerts = []
    
    if len(active_drugs) < 2:
        return alerts
        
    # Cap at 3 pairs to stay within API rate limits
    pairs_checked = 0
    api_failed = False
    
    for i in range(len(active_drugs)):
        for j in range(i + 1, len(active_drugs)):
            if pairs_checked >= 3:
                break
                
            drug1 = active_drugs[i]
            drug2 = active_drugs[j]
            
            query = f'search=openfda.generic_name:"{drug1}"+AND+drug_interactions:"{drug2}"&limit=1'
            url = f"https://api.fda.gov/drug/label.json?{query}"
            
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    total_matches = data.get('meta', {}).get('results', {}).get('total', 0)
                    
                    if total_matches > 0:
                        alerts.append({
                            "level": "🚨 FDA DDI DETECTED",
                            "message": f"Real-time Clinical API (OpenFDA) confirms active interaction between {drug1.title()} and {drug2.title()}. Patient monitoring required.",
                            "color": "#dc2626"
                        })
                elif response.status_code == 404:
                    pass
                else:
                    api_failed = True
            except requests.exceptions.RequestException:
                api_failed = True
                
            pairs_checked += 1

    if api_failed and not alerts:
        # Local heuristic fallback
        has_insulin = 'insulin' in active_drugs
        has_sulfonylurea = 'glipizide' in active_drugs or 'glyburide' in active_drugs
        
        if has_insulin and has_sulfonylurea:
            alerts.append({
                "level": "⚠️ OFFLINE DDI ALERT",
                "message": "API Request Failed. Heuristic backup: Concurrent use of Insulin and Sulfonylureas increases severe hypoglycemia risk.",
                "color": "#d97706"
            })

    return alerts
