import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

MODELS = "models"
PROC   = "data/processed"
SITE   = os.path.expanduser("~/Desktop/octagon-ai")

model        = xgb.XGBClassifier()
model.load_model(os.path.join(MODELS, "xgb_model.json"))
FEATURE_COLS = joblib.load(os.path.join(MODELS, "feature_cols.pkl"))
features_df  = pd.read_csv(os.path.join(PROC, "features.csv"),   parse_dates=["date"])
full_df      = pd.read_csv(os.path.join(PROC, "fights_elo.csv"), parse_dates=["date"])
elo_df       = pd.read_csv(os.path.join(PROC, "elo_ratings.csv"))
features_df  = features_df.sort_values("date").reset_index(drop=True)
full_df      = full_df.sort_values("date").reset_index(drop=True)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_vegas_odds():
    key = os.getenv('ODDS_API_KEY')
    if not key: return {}
    url = (f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
           f"?regions=us&markets=h2h&oddsFormat=american&apiKey={key}")
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200: return {}
        data = resp.json()
        odds_map = {}
        for fight in data:
            for bm in fight['bookmakers']:
                for market in bm['markets']:
                    if market['key'] != 'h2h': continue
                    for outcome in market['outcomes']:
                        name = outcome['name'].lower().strip()
                        if name not in odds_map: odds_map[name] = []
                        odds_map[name].append(outcome['price'])
        return {n: round(np.mean(p)) for n, p in odds_map.items()}
    except: return {}

def fetch_card(event_name):
    for status in ["upcoming","completed"]:
        try:
            url  = f"http://ufcstats.com/statistics/events/{status}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, 'lxml')
            event_link = None
            event_text = None
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                if event_name.lower() in text.lower() or text.lower() in event_name.lower():
                    if fuzz.ratio(event_name.lower(), text.lower()) > 85:
                        event_link = a['href']
                        event_text = text
                        break
            if not event_link: continue
            resp = requests.get(event_link, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, 'lxml')
            title_tag = soup.find('h2', class_='b-content__title-headline')
            full_name = title_tag.get_text(strip=True) if title_tag else event_name
            date_tag  = soup.find('li', class_='b-list__box-list-item')
            event_date = date_tag.get_text(strip=True).replace('Date:','').strip() if date_tag else ''
            location_tags = soup.find_all('li', class_='b-list__box-list-item')
            location = location_tags[1].get_text(strip=True).replace('Location:','').strip() if len(location_tags) > 1 else ''
            fights = []
            for row in soup.select('tr.js-fight-details-click'):
                cols = row.find_all('td')
                if len(cols) < 2: continue
                links = cols[1].find_all('a')
                if len(links) < 2: continue
                fa = links[0].get_text(strip=True)
                fb = links[1].get_text(strip=True)
                if not fa or not fb: continue
                weight_col = cols[6].get_text(strip=True) if len(cols) > 6 else ''
                is_title   = 'title' in weight_col.lower()
                fights.append((fa, fb, is_title, weight_col))
            return fights, full_name, event_date, location
        except: continue
    return [], event_name, '', ''

def fuzzy_match_name(name):
    all_names = list(set(
        full_df['r_name'].str.strip().tolist() +
        full_df['b_name'].str.strip().tolist()
    ))
    result = process.extractOne(name, all_names, scorer=fuzz.ratio)
    return result[0] if result and result[1] > 75 else None

def find_fighter(name):
    name_lower = name.lower().strip()
    r = full_df[full_df['r_name'].str.lower().str.strip() == name_lower]
    b = full_df[full_df['b_name'].str.lower().str.strip() == name_lower]
    if len(r) > 0: return r.sort_values('date').iloc[-1], 'r', name
    if len(b) > 0: return b.sort_values('date').iloc[-1], 'b', name
    matched = fuzzy_match_name(name)
    if matched:
        r = full_df[full_df['r_name'].str.strip() == matched]
        b = full_df[full_df['b_name'].str.strip() == matched]
        if len(r) > 0: return r.sort_values('date').iloc[-1], 'r', matched
        if len(b) > 0: return b.sort_values('date').iloc[-1], 'b', matched
    return None, None, name

def get_elo(name):
    row = elo_df[elo_df['fighter'].str.lower().str.strip() == name.lower().strip()]
    if len(row) == 0:
        row = elo_df[elo_df['fighter'].str.lower().str.contains(name.lower().strip(), na=False)]
    return float(row.iloc[0]['elo']) if len(row) > 0 else 1500.0

def get_rolling_stats(name):
    name_lower = name.lower().strip()
    r = features_df[features_df['r_name'].str.lower().str.strip() == name_lower].sort_values('date')
    b = features_df[features_df['b_name'].str.lower().str.strip() == name_lower].sort_values('date')
    if len(r) > 0:   last, sign = r.iloc[-1], 1
    elif len(b) > 0: last, sign = b.iloc[-1], -1
    else:
        return {k: 0.0 for k in ['striker_r3','striker_r5','grappler_r3',
                                   'grappler_r5','win_streak','recent_form',
                                   'chaos','finish_rate','ko_layoff','durability']}
    return {
        'striker_r3':  sign * float(last.get('striker_score_diff_r3', 0) or 0),
        'striker_r5':  sign * float(last.get('striker_score_diff_r5', 0) or 0),
        'grappler_r3': sign * float(last.get('grappler_score_diff_r3', 0) or 0),
        'grappler_r5': sign * float(last.get('grappler_score_diff_r5', 0) or 0),
        'win_streak':  sign * float(last.get('win_streak_diff', 0) or 0),
        'recent_form': float(last.get('recent_form_diff', 0.5) or 0.5) if sign == 1
                       else 1 - float(last.get('recent_form_diff', 0.5) or 0.5),
        'chaos':       float(last.get('chaos_index_r', 0.5) or 0.5) if sign == 1
                       else float(last.get('chaos_index_b', 0.5) or 0.5),
        'finish_rate': sign * float(last.get('finish_rate_diff', 0) or 0),
        'ko_layoff':   float(last.get('ko_layoff_r', 0) or 0) if sign == 1
                       else float(last.get('ko_layoff_b', 0) or 0),
        'durability':  sign * float(last.get('durability_diff', 0) or 0),
    }

def classify_style(td_avg, sub_avg, str_acc, splm):
    try:
        td_avg=float(td_avg or 0); sub_avg=float(sub_avg or 0)
        str_acc=float(str_acc or 0); splm=float(splm or 0)
    except: return 'hybrid'
    if td_avg > 2.5 and sub_avg > 0.5:  return 'grappler'
    if td_avg > 2.5 and sub_avg <= 0.5: return 'wrestler'
    if splm > 5.0  and td_avg < 1.0:    return 'striker'
    return 'hybrid'

def encode_stance(r, b):
    r,b = str(r).lower(), str(b).lower()
    if r=='orthodox' and b=='southpaw': return 1
    if r=='southpaw' and b=='orthodox': return 2
    if r==b: return 3
    if 'switch' in r or 'switch' in b: return 4
    return 0

style_map = {
    ('striker','grappler'):1,('grappler','striker'):2,
    ('striker','wrestler'):3,('wrestler','striker'):4,
    ('striker','striker'):5,('grappler','grappler'):6,
    ('wrestler','wrestler'):7,('hybrid','hybrid'):8,
    ('striker','hybrid'):9,('hybrid','striker'):10,
    ('grappler','hybrid'):11,('hybrid','grappler'):12,
    ('wrestler','hybrid'):13,('hybrid','wrestler'):14,
    ('wrestler','grappler'):15,('grappler','wrestler'):16,
}

def american_to_prob(ml):
    try:
        ml=float(ml)
        if abs(ml) < 100: return None  # impossible moneyline, reject it
        return abs(ml)/(abs(ml)+100) if ml<0 else 100/(ml+100)
    except: return None

def calculate_ev(model_prob, moneyline, stake=100):
    vp = american_to_prob(moneyline)
    if vp is None: return None, None
    ml=float(moneyline)
    payout=(100/abs(ml)*stake) if ml<0 else (ml/100*stake)
    return (model_prob*payout)-((1-model_prob)*stake), model_prob-vp

def get_odds_for_fighter(name, odds_map):
    key = name.lower()
    if key in odds_map: return odds_map[key]
    keys = list(odds_map.keys())
    if keys:
        result = process.extractOne(key, keys, scorer=fuzz.ratio)
        if result and result[1] > 75: return odds_map[result[0]]
    return None

def predict_fight(name_a, name_b, title_fight=False):
    row_a, corner_a, matched_a = find_fighter(name_a)
    row_b, corner_b, matched_b = find_fighter(name_b)

    if row_a is None:
        print(f"  [SKIP] '{name_a}' — not found in dataset")
        return None
    if row_b is None:
        print(f"  [SKIP] '{name_b}' — not found in dataset")
        return None

    if matched_a != name_a:
        print(f"  [FUZZY] '{name_a}' matched to '{matched_a}'")
    if matched_b != name_b:
        print(f"  [FUZZY] '{name_b}' matched to '{matched_b}'")

    roll_a = get_rolling_stats(matched_a)
    roll_b = get_rolling_stats(matched_b)

    if all(v == 0.0 for v in roll_a.values()):
        print(f"  [WARN] '{matched_a}' has no rolling stats")
    if all(v == 0.0 for v in roll_b.values()):
        print(f"  [WARN] '{matched_b}' has no rolling stats")

    p,q = corner_a, corner_b
    def s(row,corner,col):
        try: return float(row.get(f'{corner}_{col}',0) or 0)
        except: return 0.0

    roll_a = get_rolling_stats(matched_a)
    roll_b = get_rolling_stats(matched_b)
    elo_a  = get_elo(matched_a)
    elo_b  = get_elo(matched_b)

    style_a = classify_style(s(row_a,p,'td_avg'),s(row_a,p,'sub_avg'),
                              s(row_a,p,'str_acc'),s(row_a,p,'splm'))
    style_b = classify_style(s(row_b,q,'td_avg'),s(row_b,q,'sub_avg'),
                              s(row_b,q,'str_acc'),s(row_b,q,'splm'))

    row = {
        'elo_diff':               elo_a-elo_b,
        'striker_score_diff_r3':  roll_a['striker_r3']-roll_b['striker_r3'],
        'striker_score_diff_r5':  roll_a['striker_r5']-roll_b['striker_r5'],
        'grappler_score_diff_r3': roll_a['grappler_r3']-roll_b['grappler_r3'],
        'grappler_score_diff_r5': roll_a['grappler_r5']-roll_b['grappler_r5'],
        'sig_str_acc_diff':  s(row_a,p,'sig_str_acc')-s(row_b,q,'sig_str_acc'),
        'str_def_diff':      s(row_a,p,'str_def')-s(row_b,q,'str_def'),
        'splm_diff':         s(row_a,p,'splm')-s(row_b,q,'splm'),
        'sapm_diff':         s(row_a,p,'sapm')-s(row_b,q,'sapm'),
        'kd_diff':           s(row_a,p,'kd')-s(row_b,q,'kd'),
        'td_acc_diff':       s(row_a,p,'td_acc')-s(row_b,q,'td_acc'),
        'td_def_diff':       s(row_a,p,'td_def')-s(row_b,q,'td_def'),
        'ctrl_diff':         s(row_a,p,'ctrl')-s(row_b,q,'ctrl'),
        'sub_att_diff':      s(row_a,p,'sub_att')-s(row_b,q,'sub_att'),
        'reach_diff':        s(row_a,p,'reach')-s(row_b,q,'reach'),
        'height_diff':       s(row_a,p,'height')-s(row_b,q,'height'),
        'age_diff':          0.0,
        'win_streak_diff':   roll_a['win_streak']-roll_b['win_streak'],
        'recent_form_diff':  roll_a['recent_form']-roll_b['recent_form'],
        'days_since_diff':   0.0,
        'finish_rate_diff':  roll_a['finish_rate']-roll_b['finish_rate'],
        'durability_diff':   roll_a['durability']-roll_b['durability'],
        'chaos_index_r':     roll_a['chaos'],
        'chaos_index_b':     roll_b['chaos'],
        'chaos_combined':   (roll_a['chaos']+roll_b['chaos'])/2,
        'ko_layoff_r':       roll_a['ko_layoff'],
        'ko_layoff_b':       roll_b['ko_layoff'],
        'style_matchup':     style_map.get((style_a,style_b),0),
        'stance_matchup':    encode_stance(s(row_a,p,'stance'),s(row_b,q,'stance')),
        'title_fight_flag':  int(title_fight),
        'wins_diff':         s(row_a,p,'wins')-s(row_b,q,'wins'),
        'losses_diff':       s(row_a,p,'losses')-s(row_b,q,'losses'),
    }

    X      = pd.DataFrame([row])[FEATURE_COLS]
    prob_a = float(model.predict_proba(X)[0][1])
    prob_b = 1-prob_a
    chaos  = (roll_a['chaos']+roll_b['chaos'])/2

    elo_gap        = abs(elo_a-elo_b)
    elo_says_a     = elo_a>elo_b
    model_says_a   = prob_a>prob_b
    elo_stat_clash = (elo_says_a!=model_says_a) and elo_gap>50
    prob_gap       = abs(prob_a-prob_b)

    if chaos>0.65 or elo_stat_clash:   conf='low'
    elif chaos>0.45 or (elo_gap<80 and prob_gap>0.3): conf='medium'
    else:                               conf='high'

    winner      = matched_a if prob_a>=prob_b else matched_b
    winner_prob = max(prob_a,prob_b)

    return {
        'fighter_a':    matched_a,
        'fighter_b':    matched_b,
        'prob_a':       round(prob_a*100,1),
        'prob_b':       round(prob_b*100,1),
        'winner':       winner,
        'winner_prob':  round(winner_prob*100,1),
        'confidence':   conf,
        'elo_a':        round(elo_a,1),
        'elo_b':        round(elo_b,1),
        'elo_diff':     round(elo_a-elo_b,1),
        'style_a':      style_a,
        'style_b':      style_b,
        'chaos':        round(chaos,3),
        'elo_clash':    elo_stat_clash,
        'title_fight':  title_fight,
    }

if __name__ == "__main__":
    event_name = sys.argv[1] if len(sys.argv) > 1 else "UFC Fight Night: Allen vs. Costa"

    print(f"Exporting predictions for: {event_name}")

    # Fetch odds
    print("Fetching odds...")
    odds_map = fetch_vegas_odds()

    # Fetch card
    print("Fetching card...")
    fights, full_name, event_date, location = fetch_card(event_name)

    if not fights:
        print(f"Could not fetch card for '{event_name}'")
        sys.exit(1)

    print(f"Found {len(fights)} fights")

    # Run predictions
    fight_data = []
    for fa, fb, is_title, weight_class in fights:
        print(f"\nProcessing: {fa} vs {fb}")
        result = predict_fight(fa, fb, is_title)
        if result is None:
            fight_data.append({
                'fighter_a': fa, 'fighter_b': fb,
                'not_found': True,
                'weight_class': weight_class,
                'title_fight': is_title
            })
            continue

        # Add odds and EV
        ml_a = get_odds_for_fighter(result['fighter_a'], odds_map)
        ml_b = get_odds_for_fighter(result['fighter_b'], odds_map)

        if ml_a and ml_b:
            ev_a, edge_a = calculate_ev(result['prob_a']/100, ml_a)
            ev_b, edge_b = calculate_ev(result['prob_b']/100, ml_b)
            result['odds_a']      = int(ml_a)
            result['odds_b']      = int(ml_b)
            result['ev_a']        = round(ev_a, 1)
            result['ev_b']        = round(ev_b, 1)
            result['edge_a']      = round(edge_a*100, 1)
            result['edge_b']      = round(edge_b*100, 1)
            result['vegas_prob_a']= round(american_to_prob(ml_a)*100, 1)
            result['vegas_prob_b']= round(american_to_prob(ml_b)*100, 1)
        else:
            result['odds_a'] = result['odds_b'] = None
            result['ev_a']   = result['ev_b']   = None
            result['edge_a'] = result['edge_b'] = None
            result['vegas_prob_a'] = result['vegas_prob_b'] = None

        result['weight_class'] = weight_class
        result['not_found']    = False
        fight_data.append(result)

    # Build EV leaderboard
    ev_plays = []
    for f in fight_data:
        if f.get('not_found'): continue
        for fighter, prob, ev, edge, odds, vp in [
            (f['fighter_a'], f['prob_a'], f.get('ev_a'), f.get('edge_a'), f.get('odds_a'), f.get('vegas_prob_a')),
            (f['fighter_b'], f['prob_b'], f.get('ev_b'), f.get('edge_b'), f.get('odds_b'), f.get('vegas_prob_b')),
        ]:
            if ev and ev > 0:
                ev_plays.append({
                    'fighter':    fighter,
                    'prob':       prob,
                    'ev':         round(ev,1),
                    'edge':       round(edge,1),
                    'odds':       odds,
                    'vegas_prob': vp,
                    'confidence': f['confidence']
                })
    ev_plays.sort(key=lambda x: x['ev'], reverse=True)

    # Build parlay legs
    high_legs = [f for f in fight_data
                 if not f.get('not_found') and f['confidence']=='high'
                 and f['winner_prob'] >= 60]
    med_legs  = [f for f in fight_data
                 if not f.get('not_found') and f['confidence'] in ['high','medium']
                 and f['winner_prob'] >= 60]

    def parlay_prob(legs):
        p = 1.0
        for l in legs: p *= l['winner_prob']/100
        return round(p*100,1)

    # Summary stats
    total      = len(fight_data)
    high_count = sum(1 for f in fight_data if not f.get('not_found') and f['confidence']=='high')
    med_count  = sum(1 for f in fight_data if not f.get('not_found') and f['confidence']=='medium')
    low_count  = sum(1 for f in fight_data if not f.get('not_found') and f['confidence']=='low')
    not_found  = sum(1 for f in fight_data if f.get('not_found'))

    # Final JSON
    output = {
        'event_name':   full_name,
        'event_slug':   full_name.replace(' ','_').replace(':','').replace('/',''),
        'event_date':   event_date,
        'location':     location,
        'generated_at': datetime.now().isoformat(),
        'model_stats': {
            'accuracy':      83.1,
            'roc_auc':       0.9229,
            'high_conf_acc': 94.8,
            'overfit_gap':   0.026
        },
        'summary': {
            'total_fights':  total,
            'high_conf':     high_count,
            'medium_conf':   med_count,
            'low_conf':      low_count,
            'not_found':     not_found,
            'ev_plays':      len(ev_plays)
        },
        'fights':   fight_data,
        'ev_plays': ev_plays[:10],
        'parlays': {
            'high_only': {
                'legs':  [{'fighter': f['winner'], 'prob': f['winner_prob']} for f in high_legs],
                'prob':  parlay_prob(high_legs),
                'count': len(high_legs)
            },
            'high_and_med': {
                'legs':  [{'fighter': f['winner'], 'prob': f['winner_prob']} for f in med_legs],
                'prob':  parlay_prob(med_legs),
                'count': len(med_legs)
            }
        }
    }

    # Save to octagon-ai site folder
    slug     = output['event_slug']
    out_path = os.path.join(SITE, "data", f"{slug}.json")
    os.makedirs(os.path.join(SITE, "data"), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Saved: {out_path}")

    # Also save an index of all events
    index_path = os.path.join(SITE, "data", "index.json")
    index = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    # Update or add this event
    existing = [i for i in index if i['slug'] != slug]
    existing.append({
        'slug':         slug,
        'name':         full_name,
        'date':         event_date,
        'location':     location,
        'generated_at': output['generated_at'],
        'summary':      output['summary']
    })
    existing.sort(key=lambda x: x['generated_at'], reverse=True)

    with open(index_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"Index updated: {index_path}")
    print(f"\nDone. Run this to publish:")
    print(f"  cd ~/Desktop/octagon-ai")
    print(f"  git add .")
    print(f'  git commit -m "Add {full_name}"')
    print(f"  git push")