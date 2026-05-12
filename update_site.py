import subprocess
import os
import sys
import json
import requests
from bs4 import BeautifulSoup

SITE      = os.path.dirname(os.path.abspath(__file__))
PREDICTOR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_all_ufc_events():
    """Scrape all events from UFCStats"""
    events = []
    for status in ['upcoming', 'completed']:
        try:
            url  = f"http://ufcstats.com/statistics/events/{status}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, 'lxml')
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                if text and len(text) > 5 and ('UFC' in text or 'ufc' in text.lower()):
                    events.append(text.strip())
        except Exception as e:
            print(f"  Error scraping {status}: {e}")
    return list(dict.fromkeys(events))  # deduplicate

def get_exported_events():
    """Get list of already exported event slugs"""
    index_path = os.path.join(SITE, "data", "index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        index = json.load(f)
    return [e['name'] for e in index]

def slugify(name):
    return name.replace(' ','_').replace(':','').replace('/','').replace('.','')

if __name__ == "__main__":
    print("=" * 55)
    print("  Octagon AI — Auto Site Updater")
    print("=" * 55)

    # Get all UFC events from UFCStats
    print("\nScraping UFCStats for all events...")
    all_events = get_all_ufc_events()
    print(f"  Found {len(all_events)} UFC events on UFCStats")

    # Get already exported events
    exported = get_exported_events()
    print(f"  Already exported: {len(exported)} events")

    # Find missing events
    missing = [e for e in all_events if e not in exported]
    print(f"  New events to export: {len(missing)}")

    if not missing:
        print("\n  Everything is up to date.")
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

    print(f"\nDone — https://sushimaster124.github.io/octagon-ai")