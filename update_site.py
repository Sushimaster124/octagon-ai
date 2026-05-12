import subprocess
import os
import sys
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from rapidfuzz import fuzz

SITE      = os.path.dirname(os.path.abspath(__file__))
PREDICTOR = os.path.expanduser("~/Desktop/ufc-predictor")
HEADERS   = {'User-Agent': 'Mozilla/5.0'}

def get_all_ufc_events():
    events = []
    for status in ['upcoming', 'completed']:
        try:
            url  = f"http://ufcstats.com/statistics/events/{status}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, 'lxml')
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                if text and len(text) > 5 and 'UFC' in text:
                    events.append(text.strip())
        except Exception as e:
            print(f"  Error: {e}")
    return list(dict.fromkeys(events))

def get_exported_events():
    index_path = os.path.join(SITE, "data", "index.json")
    if not os.path.exists(index_path):
        return [], []
    with open(index_path) as f:
        index = json.load(f)
    return [e['name'] for e in index], index

def fetch_actual_results(event_name):
    """Scrape actual fight results from UFCStats"""
    try:
        url  = "http://ufcstats.com/statistics/events/completed"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'lxml')

        event_link = None
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            if fuzz.ratio(event_name.lower(), text.lower()) > 85:
                event_link = a['href']
                break

        if not event_link:
            return None

        resp = requests.get(event_link, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'lxml')

        results = {}
        for row in soup.select('tr.js-fight-details-click'):
            cols = row.find_all('td')
            if len(cols) < 2: continue
            links = cols[1].find_all('a')
            if len(links) < 2: continue
            fa = links[0].get_text(strip=True)
            fb = links[1].get_text(strip=True)
            result_col = cols[0].get_text(strip=True).lower()
            if 'win' in result_col:
                results[f"{fa}|{fb}"] = fa
                results[f"{fb}|{fa}"] = fa

        return results if results else None

    except Exception as e:
        print(f"  Results fetch error: {e}")
        return None

def score_predictions(event_data, actual_results):
    """Score model predictions against actual results"""
    correct = 0
    wrong   = 0
    total   = 0
    by_conf = {
        'high':   {'correct': 0, 'wrong': 0},
        'medium': {'correct': 0, 'wrong': 0},
        'low':    {'correct': 0, 'wrong': 0},
    }
    fight_scores = []

    for fight in event_data.get('fights', []):
        if fight.get('not_found'): continue

        fa   = fight['fighter_a']
        fb   = fight['fighter_b']
        pred = fight['winner']
        conf = fight.get('confidence', 'medium')

        actual = (actual_results.get(f"{fa}|{fb}") or
                  actual_results.get(f"{fb}|{fa}"))
        if not actual: continue

        is_correct = pred.lower().strip() == actual.lower().strip()
        total += 1

        if is_correct:
            correct += 1
            if conf in by_conf: by_conf[conf]['correct'] += 1
        else:
            wrong += 1
            if conf in by_conf: by_conf[conf]['wrong'] += 1

        fight_scores.append({
            'fighter_a':  fa,
            'fighter_b':  fb,
            'predicted':  pred,
            'actual':     actual,
            'correct':    is_correct,
            'confidence': conf,
            'prob':       fight.get('winner_prob', 50),
        })

    return {
        'total':         total,
        'correct':       correct,
        'wrong':         wrong,
        'accuracy':      round(correct/total*100, 1) if total > 0 else None,
        'by_confidence': by_conf,
        'fight_scores':  fight_scores,
    }

def update_accuracy_tracker(index):
    """Score all completed events and save accuracy.json"""
    print("\nUpdating accuracy tracker...")
    accuracy_path = os.path.join(SITE, "data", "accuracy.json")

    existing = {}
    if os.path.exists(accuracy_path):
        with open(accuracy_path) as f:
            existing = json.load(f)

    event_scores = existing.get('events', {})
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    for event in index:
        slug = event['slug']
        name = event['name']
        date = event.get('date', '')

        # Skip upcoming events
        if date >= today:
            continue

        # Skip if already scored
        if slug in event_scores:
            continue

        print(f"  Scoring: {name}")

        # Load event predictions
        event_path = os.path.join(SITE, "data", f"{slug}.json")
        if not os.path.exists(event_path):
            continue

        with open(event_path) as f:
            event_data = json.load(f)

        # Fetch actual results
        actual = fetch_actual_results(name)
        if not actual:
            print(f"    No results found yet — skipping")
            continue

        # Score predictions
        scores = score_predictions(event_data, actual)
        if scores['total'] == 0:
            print(f"    Could not match results — skipping")
            continue

        event_scores[slug] = {
            'name':         name,
            'date':         date,
            'scored_at':    datetime.now(timezone.utc).isoformat(),
            **scores
        }

        print(f"    {scores['correct']}/{scores['total']} correct "
              f"({scores['accuracy']}%)")

    # Compute overall stats
    all_correct = sum(e['correct'] for e in event_scores.values())
    all_total   = sum(e['total']   for e in event_scores.values())
    all_wrong   = sum(e['wrong']   for e in event_scores.values())

    by_conf_total = {
        'high':   {'correct': 0, 'wrong': 0},
        'medium': {'correct': 0, 'wrong': 0},
        'low':    {'correct': 0, 'wrong': 0},
    }

    for e in event_scores.values():
        for conf in ['high', 'medium', 'low']:
            bc = e.get('by_confidence', {}).get(conf, {})
            by_conf_total[conf]['correct'] += bc.get('correct', 0)
            by_conf_total[conf]['wrong']   += bc.get('wrong', 0)

    accuracy_data = {
        'updated_at':    datetime.now(timezone.utc).isoformat(),
        'overall': {
            'total':    all_total,
            'correct':  all_correct,
            'wrong':    all_wrong,
            'accuracy': round(all_correct/all_total*100, 1) if all_total > 0 else None,
        },
        'by_confidence': by_conf_total,
        'events':        event_scores,
    }

    with open(accuracy_path, 'w') as f:
        json.dump(accuracy_data, f, indent=2)

    print(f"  Overall: {all_correct}/{all_total} = "
          f"{accuracy_data['overall']['accuracy']}%")
    print(f"  Saved: {accuracy_path}")
    return accuracy_data

if __name__ == "__main__":
    print("=" * 55)
    print("  Octagon AI — Auto Site Updater")
    print("=" * 55)

    # Get all UFC events from UFCStats
    print("\nScraping UFCStats for all events...")
    all_events = get_all_ufc_events()
    print(f"  Found {len(all_events)} UFC events")

    # Get already exported events
    exported_names, index = get_exported_events()
    print(f"  Already exported: {len(exported_names)} events")

    # Find missing events
    missing = [e for e in all_events if e not in exported_names]
    print(f"  New events to export: {len(missing)}")

    if not missing:
        print("\n  Predictions up to date.")
    else:
        for event in missing:
            print(f"\nExporting: {event}")
            result = subprocess.run(
                [sys.executable,
                 os.path.join(PREDICTOR, "export_json.py"),
                 event],
                cwd=PREDICTOR,
                capture_output=True,
                text=True
            )
            if "Saved:" in result.stdout:
                print(f"  Done")
            elif "Could not fetch" in result.stdout:
                print(f"  Skipped — no fights yet")
            else:
                print(f"  Failed")
                if result.stderr:
                    print(f"  {result.stderr[:150]}")

    # Reload index after exports
    _, index = get_exported_events()

    # Update accuracy tracker
    update_accuracy_tracker(index)

    print(f"\nDone — https://sushimaster124.github.io/octagon-ai")