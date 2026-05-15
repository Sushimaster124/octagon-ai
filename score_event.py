"""
score_event.py — runs in GitHub Actions after each UFC event.
Scrapes results from UFCStats, scores predictions, updates accuracy.json.
"""
import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from rapidfuzz import fuzz

# In GitHub Actions, GITHUB_ACTIONS_SITE is set to github.workspace.
# Locally it falls back to the directory this file lives in.
SITE    = os.environ.get('GITHUB_ACTIONS_SITE', os.path.dirname(os.path.abspath(__file__)))
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def fetch_actual_results(event_name):
    """Scrape actual fight results from UFCStats completed events."""
    try:
        url  = "http://ufcstats.com/statistics/events/completed"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'lxml')

        event_link = None
        for a in soup.find_all('a', href=True):
            text = a.get_text(strip=True)
            if fuzz.ratio(event_name.lower(), text.lower()) > 80:
                event_link = a['href']
                break

        if not event_link:
            print(f"  Could not find '{event_name}' on UFCStats completed list")
            return None

        resp = requests.get(event_link, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'lxml')

        results = {}
        for row in soup.select('tr.js-fight-details-click'):
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            links = cols[1].find_all('a')
            if len(links) < 2:
                continue
            fa = links[0].get_text(strip=True)
            fb = links[1].get_text(strip=True)
            # First fighter listed is the winner on UFCStats
            results[f"{fa}|{fb}"] = fa
            results[f"{fb}|{fa}"] = fa

        print(f"  Found {len(results)//2} fight results")
        return results if results else None

    except Exception as e:
        print(f"  Results fetch error: {e}")
        return None


def score_predictions(event_data, actual_results):
    """Compare model predictions to actual results."""
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
        if fight.get('not_found'):
            continue

        fa   = fight['fighter_a']
        fb   = fight['fighter_b']
        pred = fight['winner']
        conf = fight.get('confidence', 'medium')

        actual = (actual_results.get(f"{fa}|{fb}") or
                  actual_results.get(f"{fb}|{fa}"))
        if not actual:
            continue

        is_correct = pred.lower().strip() == actual.lower().strip()
        total += 1

        if is_correct:
            correct += 1
            if conf in by_conf:
                by_conf[conf]['correct'] += 1
        else:
            wrong += 1
            if conf in by_conf:
                by_conf[conf]['wrong'] += 1

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
        'accuracy':      round(correct / total * 100, 1) if total > 0 else None,
        'by_confidence': by_conf,
        'fight_scores':  fight_scores,
    }


def main():
    print("=" * 55)
    print("  Octagon AI — Accuracy Scorer")
    print("=" * 55)

    index_path    = os.path.join(SITE, "data", "index.json")
    accuracy_path = os.path.join(SITE, "data", "accuracy.json")

    if not os.path.exists(index_path):
        print("No index.json found — nothing to score.")
        return

    with open(index_path) as f:
        index = json.load(f)

    existing = {}
    if os.path.exists(accuracy_path):
        with open(accuracy_path) as f:
            existing = json.load(f)

    event_scores = existing.get('events', {})
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    scored_any = False

    for event in index:
        slug = event.get('slug', '')
        name = event.get('name', '')
        date = event.get('date', '')

        # Skip upcoming events
        if date >= today:
            print(f"  Skipping upcoming: {name}")
            continue

        # Skip already scored
        if slug in event_scores:
            print(f"  Already scored: {name}")
            continue

        print(f"\nScoring: {name} ({date})")

        event_path = os.path.join(SITE, "data", f"{slug}.json")
        if not os.path.exists(event_path):
            print(f"  No prediction file found — skipping")
            continue

        with open(event_path) as f:
            event_data = json.load(f)

        actual = fetch_actual_results(name)
        if not actual:
            print(f"  Results not available yet — skipping")
            continue

        scores = score_predictions(event_data, actual)
        if scores['total'] == 0:
            print(f"  Could not match any fights — skipping")
            continue

        event_scores[slug] = {
            'name':      name,
            'date':      date,
            'scored_at': datetime.now(timezone.utc).isoformat(),
            **scores
        }

        print(f"  Result: {scores['correct']}/{scores['total']} correct "
              f"({scores['accuracy']}%)")
        scored_any = True

    if not scored_any:
        print("\nNo new events to score.")

    # Recompute overall stats
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
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'overall': {
            'total':    all_total,
            'correct':  all_correct,
            'wrong':    all_wrong,
            'accuracy': round(all_correct / all_total * 100, 1) if all_total > 0 else None,
        },
        'by_confidence': by_conf_total,
        'events':        event_scores,
    }

    with open(accuracy_path, 'w') as f:
        json.dump(accuracy_data, f, indent=2)

    print(f"\nOverall: {all_correct}/{all_total} = "
          f"{accuracy_data['overall']['accuracy']}%")
    print(f"Saved: {accuracy_path}")


if __name__ == "__main__":
    main()
