import subprocess
import os
import sys

SITE = os.path.dirname(os.path.abspath(__file__))
PREDICTOR = os.path.dirname(os.path.abspath(__file__))

EVENTS = [
    "UFC Fight Night: Allen vs. Costa",
    "UFC 328: Chimaev vs. Strickland",
    "UFC Fight Night: Song vs. Figueiredo",
    "UFC 327: Prochazka vs. Ulberg",
    "UFC Fight Night: Sterling vs. Zalal",
    "UFC Fight Night: Burns vs. Malott",
    "UFC 326: Holloway vs. Oliveira 2",
    "UFC 325: Volkanovski vs. Lopes 2",
    "UFC 324: Gaethje vs. Pimblett",
    "UFC 323: Dvalishvili vs. Yan 2",
    "UFC 322: Della Maddalena vs. Makhachev",
    "UFC 321: Aspinall vs. Gane",
]

print("=" * 50)
print("  Octagon AI — Site Updater")
print("=" * 50)

for event in EVENTS:
    print(f"\nExporting: {event}")
    result = subprocess.run(
        [sys.executable, os.path.join(PREDICTOR, "export_json.py"), event],
        cwd=PREDICTOR,
        capture_output=True,
        text=True
    )
    if "Saved:" in result.stdout:
        print(f"  Done")
    else:
        print(f"  Skipped or failed")
        if result.stderr:
            print(f"  {result.stderr[:200]}")

print(f"\nDone — https://sushimaster124.github.io/octagon-ai")
# updated
