import json
import glob
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime, timedelta

# Load odds dataset
odds = pd.read_csv(
    '/Users/zackramsey/Desktop/ufc-predictor/data/raw/ufc-master.csv',
    parse_dates=['date']
)
odds = odds[odds['R_odds'].notna() & odds['B_odds'].notna()].copy()

def implied(ml):
    ml = float(ml)
    return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

def find_odds(name_a, name_b, event_date):
    """Find odds for a fight by fuzzy matching names and date."""
    # Parse event date
    try:
        if isinstance(event_date, str):
            # Handle formats like "May 09, 2026" or "2026-05-09"
            for fmt in ['%B %d, %Y', '%Y-%m-%d', '%b %d, %Y']:
                try:
                    edate = datetime.strptime(event_date.strip(), fmt).date()
                    break
                except:
                    continue
            else:
                return None
        else:
            edate = event_date.date()
    except:
        return None

    # Search within 3 days of event date
    window = odds[
        (odds['date'].dt.date >= edate - timedelta(days=3)) &
        (odds['date'].dt.date <= edate + timedelta(days=3))
    ]
    if len(window) == 0:
        return None

    best_score = 0
    best_row = None
    flipped = False

    for _, row in window.iterrows():
        # Try straight match
        score_a = fuzz.ratio(name_a.lower(), row['R_fighter'].lower())
        score_b = fuzz.ratio(name_b.lower(), row['B_fighter'].lower())
        combined = (score_a + score_b) / 2

        # Try flipped match
        score_af = fuzz.ratio(name_a.lower(), row['B_fighter'].lower())
        score_bf = fuzz.ratio(name_b.lower(), row['R_fighter'].lower())
        combined_f = (score_af + score_bf) / 2

        if combined > best_score and combined > 70:
            best_score = combined
            best_row = row
            flipped = False
        if combined_f > best_score and combined_f > 70:
            best_score = combined_f
            best_row = row
            flipped = True

    if best_row is None:
        return None

    if flipped:
        return {
            'odds_a': int(best_row['B_odds']),
            'odds_b': int(best_row['R_odds']),
            'vegas_prob_a': round(implied(best_row['B_odds'])*100, 1),
            'vegas_prob_b': round(implied(best_row['R_odds'])*100, 1),
        }
    else:
        return {
            'odds_a': int(best_row['R_odds']),
            'odds_b': int(best_row['B_odds']),
            'vegas_prob_a': round(implied(best_row['R_odds'])*100, 1),
            'vegas_prob_b': round(implied(best_row['B_odds'])*100, 1),
        }

def calculate_ev(model_prob, ml, stake=100):
    ml = float(ml)
    payout = (100/abs(ml)*stake) if ml < 0 else (ml/100*stake)
    return round((model_prob*payout)-((1-model_prob)*stake), 1)

# Process all event JSONs
event_files = sorted(glob.glob('data/UFC*.json'))
total_filled = 0
total_failed = 0

for path in event_files:
    d = json.load(open(path))
    event_date = d.get('event_date', '')
    changed = 0

    for f in d['fights']:
        if f.get('not_found'): continue
        if f.get('odds_a') is not None: continue  # already has odds

        result = find_odds(f['fighter_a'], f['fighter_b'], event_date)
        if result is None:
            total_failed += 1
            continue

        # Write odds in
        f['odds_a'] = result['odds_a']
        f['odds_b'] = result['odds_b']
        f['vegas_prob_a'] = result['vegas_prob_a']
        f['vegas_prob_b'] = result['vegas_prob_b']

        # Compute EV
        prob_a = f['prob_a'] / 100
        prob_b = f['prob_b'] / 100
        f['ev_a'] = calculate_ev(prob_a, result['odds_a'])
        f['ev_b'] = calculate_ev(prob_b, result['odds_b'])
        f['edge_a'] = round(prob_a - result['vegas_prob_a']/100, 3)
        f['edge_b'] = round(prob_b - result['vegas_prob_b']/100, 3)

        changed += 1
        total_filled += 1

    if changed > 0:
        with open(path, 'w') as file:
            json.dump(d, file, indent=2, default=str)
        print(f"{path.split('/')[-1]}: +{changed} odds added")

print(f"\nTotal filled: {total_filled}")
print(f"Total failed: {total_failed}")
print(f"\nNow run: python3 score_event.py to rebuild accuracy.json")