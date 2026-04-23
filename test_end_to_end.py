"""
test_end_to_end.py
══════════════════════════════════════════════════════════════════
End-to-End Performance Test Suite for NutraDiet26
Tests the full pipeline: Text Input → BERT Parser → Nutrition Lookup → Results

Run with:
    python test_end_to_end.py            # full suite with report
    python test_end_to_end.py --quick    # fast subset only
    python test_end_to_end.py --verbose  # show all details
══════════════════════════════════════════════════════════════════
"""

import sys
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# TEST CASE DEFINITIONS
# ─────────────────────────────────────────────────────────────────

@dataclass
class FoodExpectation:
    """Expected result for a single food item."""
    name_contains: str           # substring to find in matched name
    min_calories: float = 0.0
    max_calories: float = 9999.0
    min_weight: float = 1.0
    source: Optional[str] = None  # "text", "image", "text+image"


@dataclass
class TestCase:
    """A single end-to-end test scenario."""
    label: str
    text_input: Optional[str]
    moondream_input: Optional[str]
    expected_items: list[FoodExpectation]
    min_items: int = 1           # minimum number of items expected
    max_items: int = 20
    category: str = "general"   # for grouping in report


# ─────────────────────────────────────────────────────────────────
# TEST SUITE
# ─────────────────────────────────────────────────────────────────

TEST_CASES = [

    # ── SINGLE FOOD — TEXT ONLY ───────────────────────────────────
    TestCase(
        label="Single food: dal makhani text only",
        text_input="dal makhani 200g",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="dal makhani", min_calories=100, max_calories=600),
        ],
        min_items=1, max_items=2,
        category="text_only",
    ),
    TestCase(
        label="Single food: butter chicken text only",
        text_input="butter chicken 250g",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="butter chicken", min_calories=150, max_calories=700),
        ],
        min_items=1, max_items=2,
        category="text_only",
    ),
    TestCase(
        label="Single food: chapati with count",
        text_input="2 chapati",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="chapati", min_calories=50, max_calories=300),
        ],
        min_items=1, max_items=2,
        category="text_only",
    ),
    TestCase(
        label="Single food: idli with count",
        text_input="3 idlis",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="idli", min_calories=50, max_calories=300),
        ],
        min_items=1, max_items=2,
        category="text_only",
    ),
    TestCase(
        label="Single food: biryani with weight",
        text_input="1 plate of biryani around 300 grams",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="biryani", min_calories=200, max_calories=900),
        ],
        min_items=1, max_items=3,
        category="text_only",
    ),

    # ── MULTI FOOD — TEXT ONLY ────────────────────────────────────
    TestCase(
        label="Multi food: idli + sambar",
        text_input="2 idlis and 1 bowl sambar",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="idli"),
            FoodExpectation(name_contains="sambar"),
        ],
        min_items=2, max_items=4,
        category="text_only",
    ),
    TestCase(
        label="Multi food: dal makhani + roti",
        text_input="dal makhani 200g with 2 rotis",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="dal makhani"),
            FoodExpectation(name_contains="chapati"),
        ],
        min_items=2, max_items=4,
        category="text_only",
    ),
    TestCase(
        label="Multi food: rice + egg curry",
        text_input="150g boiled rice and egg curry",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="rice"),
            FoodExpectation(name_contains="egg"),
        ],
        min_items=2, max_items=4,
        category="text_only",
    ),
    TestCase(
        label="Multi food: three items",
        text_input="chapati, dal makhani, and boondi raita",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="chapati"),
            FoodExpectation(name_contains="dal"),
            FoodExpectation(name_contains="raita"),
        ],
        min_items=2, max_items=5,
        category="text_only",
    ),

    # ── MOONDREAM TEXT — IMAGE SIMULATION ─────────────────────────
    TestCase(
        label="Moondream: comma-separated list",
        text_input=None,
        moondream_input="1 small bowl of chutney, 300g of rice, 1 cup of curry",
        expected_items=[
            FoodExpectation(name_contains="chutney"),
            FoodExpectation(name_contains="rice"),
        ],
        min_items=2, max_items=5,
        category="moondream_only",
    ),
    TestCase(
        label="Moondream: filler sentence style",
        text_input=None,
        moondream_input="I can see a plate of biryani, some raita, and 2 pieces of naan.",
        expected_items=[
            FoodExpectation(name_contains="biryani"),
        ],
        min_items=1, max_items=5,
        category="moondream_only",
    ),
    TestCase(
        label="Moondream: sentence-per-food style",
        text_input=None,
        moondream_input=(
            "I had 2 pieces of idli. "
            "I had 1 small bowl of sambar. "
            "I had 1 cup of coconut chutney."
        ),
        expected_items=[
            FoodExpectation(name_contains="idli"),
            FoodExpectation(name_contains="sambar"),
        ],
        min_items=2, max_items=5,
        category="moondream_only",
    ),
    TestCase(
        label="Moondream: thali description",
        text_input=None,
        moondream_input=(
            "I had 1 full plate of rice, 1 bowl of dal, "
            "2 pieces of chapati, 1 small bowl of pickle."
        ),
        expected_items=[
            FoodExpectation(name_contains="rice"),
            FoodExpectation(name_contains="dal"),
            FoodExpectation(name_contains="chapati"),
        ],
        min_items=3, max_items=6,
        category="moondream_only",
    ),

    # ── MERGED — TEXT + MOONDREAM ─────────────────────────────────
    TestCase(
        label="Merge: text has priority for portions",
        text_input="dal makhani 300g and naan",
        moondream_input="1 cup of dal makhani, 2 pieces of naan, 300g of rice",
        expected_items=[
            FoodExpectation(name_contains="dal makhani", min_weight=250),
            FoodExpectation(name_contains="naan"),
        ],
        min_items=2, max_items=5,
        category="merged",
    ),
    TestCase(
        label="Merge: moondream fills missing portions",
        text_input="idli and sambar",
        moondream_input="3 pieces of idli, 1 bowl of sambar",
        expected_items=[
            FoodExpectation(name_contains="idli"),
            FoodExpectation(name_contains="sambar"),
        ],
        min_items=2, max_items=4,
        category="merged",
    ),
    TestCase(
        label="Merge: moondream adds extra item",
        text_input="butter chicken 200g",
        moondream_input="1 plate of butter chicken, 2 pieces of naan",
        expected_items=[
            FoodExpectation(name_contains="butter chicken"),
            FoodExpectation(name_contains="naan"),
        ],
        min_items=2, max_items=4,
        category="merged",
    ),

    # ── EDGE CASES ────────────────────────────────────────────────
    TestCase(
        label="Edge: unknown food handled gracefully",
        text_input="xyzfooddoesnotexist123",
        moondream_input=None,
        expected_items=[],   # may return something via fuzzy match — that's ok
        min_items=0, max_items=5,
        category="edge_cases",
    ),
    TestCase(
        label="Edge: very short input",
        text_input="rice",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="rice"),
        ],
        min_items=1, max_items=3,
        category="edge_cases",
    ),
    TestCase(
        label="Edge: both inputs empty strings handled",
        text_input="",
        moondream_input="",
        expected_items=[],
        min_items=0, max_items=0,
        category="edge_cases",
    ),
    TestCase(
        label="Edge: Indian food with weight in description",
        text_input="1 bowl of rajma around 200 grams with boiled rice 150g",
        moondream_input=None,
        expected_items=[
            FoodExpectation(name_contains="rajmah"),
            FoodExpectation(name_contains="rice"),
        ],
        min_items=2, max_items=4,
        category="edge_cases",
    ),
]

# ─────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    duration_ms: float
    parsed_items: list[dict]
    nutrition_results: list[dict]
    failures: list[str] = field(default_factory=list)
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────

def run_pipeline(text_input, moondream_input):
    """
    Runs the full NutraDiet26 pipeline and returns parsed + nutrition data.
    Mirrors exactly what app.py does in the /predict endpoint.
    """
    from llm.parser import parse_food_items
    from modules.nutrition import get_nutrition

    food_items = parse_food_items(
        text=text_input or None,
        moondream_text=moondream_input or None,
    )

    results = []
    for item in food_items:
        weight = item["quantity_grams"] if item["quantity_grams"] is not None else 100.0
        nutrients = get_nutrition(item["name"], weight)
        results.append({
            "food":       item["name"],
            "weight":     weight,
            "source":     item.get("source", "unknown"),
            "confidence": item.get("confidence", 0.0),
            "nutrients":  nutrients,
        })

    return food_items, results


# ─────────────────────────────────────────────────────────────────
# ASSERTION ENGINE
# ─────────────────────────────────────────────────────────────────

def evaluate(tc: TestCase, parsed_items, nutrition_results) -> tuple[bool, list[str]]:
    failures = []
    matched_names = [r["food"].lower() for r in nutrition_results]

    # ── Item count check ─────────────────────────────────────────
    n = len(nutrition_results)
    if n < tc.min_items:
        failures.append(f"Too few items: got {n}, expected at least {tc.min_items}")
    if n > tc.max_items:
        failures.append(f"Too many items: got {n}, expected at most {tc.max_items}")

    # ── Per-expected-item checks ──────────────────────────────────
    for exp in tc.expected_items:
        # Find matching result
        match = next(
            (r for r in nutrition_results if exp.name_contains.lower() in r["food"].lower()),
            None
        )
        if match is None:
            failures.append(f"Expected food containing '{exp.name_contains}' not found. Got: {matched_names}")
            continue

        n_cals = match["nutrients"].get("calories") or 0

        # Calorie range check (only when food was found in DB)
        if match["nutrients"].get("matched_name"):
            if n_cals < exp.min_calories:
                failures.append(
                    f"'{exp.name_contains}' calories {n_cals:.1f} below min {exp.min_calories}"
                )
            if n_cals > exp.max_calories:
                failures.append(
                    f"'{exp.name_contains}' calories {n_cals:.1f} above max {exp.max_calories}"
                )

        # Weight check
        if match["weight"] < exp.min_weight:
            failures.append(
                f"'{exp.name_contains}' weight {match['weight']:.1f}g below min {exp.min_weight}g"
            )

        # Source check
        if exp.source and match["source"] != exp.source:
            failures.append(
                f"'{exp.name_contains}' source is '{match['source']}', expected '{exp.source}'"
            )

    return len(failures) == 0, failures


# ─────────────────────────────────────────────────────────────────
# RUN ONE TEST
# ─────────────────────────────────────────────────────────────────

def run_test(tc: TestCase, verbose: bool = False) -> TestResult:
    start = time.perf_counter()
    try:
        parsed_items, nutrition_results = run_pipeline(tc.text_input, tc.moondream_input)
        duration_ms = (time.perf_counter() - start) * 1000
        passed, failures = evaluate(tc, parsed_items, nutrition_results)
        return TestResult(
            test_case=tc,
            passed=passed,
            duration_ms=duration_ms,
            parsed_items=parsed_items,
            nutrition_results=nutrition_results,
            failures=failures,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return TestResult(
            test_case=tc,
            passed=False,
            duration_ms=duration_ms,
            parsed_items=[],
            nutrition_results=[],
            failures=[f"Pipeline raised exception: {e}"],
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────

def print_report(results: list[TestResult], verbose: bool):
    total      = len(results)
    passed     = sum(1 for r in results if r.passed)
    failed     = total - passed
    total_ms   = sum(r.duration_ms for r in results)
    avg_ms     = total_ms / total if total else 0

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"

    print("\n" + "═" * 65)
    print("  NUTRADIET26 — END-TO-END TEST REPORT")
    print("═" * 65)

    # ── Per-category summary ──────────────────────────────────────
    categories = {}
    for r in results:
        cat = r.test_case.category
        categories.setdefault(cat, {"pass": 0, "fail": 0})
        if r.passed:
            categories[cat]["pass"] += 1
        else:
            categories[cat]["fail"] += 1

    print("\n📂 Results by Category:")
    for cat, counts in categories.items():
        total_cat = counts["pass"] + counts["fail"]
        bar_fill  = int((counts["pass"] / total_cat) * 20)
        bar       = "█" * bar_fill + "░" * (20 - bar_fill)
        print(f"   {cat:<20} [{bar}] {counts['pass']}/{total_cat}")

    # ── Per-test details ──────────────────────────────────────────
    print("\n📋 Individual Tests:\n")
    for r in results:
        status = PASS if r.passed else FAIL
        print(f"  {status}  {r.test_case.label}")
        print(f"         ⏱  {r.duration_ms:.0f}ms  |  "
              f"Items found: {len(r.nutrition_results)}")

        if verbose or not r.passed:
            # Show what was detected
            for item in r.nutrition_results:
                cals  = item["nutrients"].get("calories") or 0
                match = item["nutrients"].get("matched_name") or "NOT FOUND"
                conf  = item.get("confidence", 0)
                src   = item.get("source", "?")
                print(f"         → {item['food']:<35} "
                      f"{item['weight']:>6.0f}g  "
                      f"{cals:>7.1f} kcal  "
                      f"conf:{conf:.2f}  [{src}]")
                if verbose:
                    print(f"           DB match: {match}")

            # Show failures
            for f in r.failures:
                print(f"         ⚠️  {f}")

        print()

    # ── Performance breakdown ──────────────────────────────────────
    slowest = sorted(results, key=lambda r: r.duration_ms, reverse=True)[:3]
    print("⏱  Slowest Tests:")
    for r in slowest:
        print(f"   {r.duration_ms:>7.0f}ms  {r.test_case.label}")

    # ── Confidence stats ──────────────────────────────────────────
    all_confs = [
        item["confidence"]
        for r in results
        for item in r.nutrition_results
        if item.get("confidence")
    ]
    if all_confs:
        avg_conf = sum(all_confs) / len(all_confs)
        low_conf = sum(1 for c in all_confs if c < 0.65)
        print(f"\n🎯 BERT Confidence Stats:")
        print(f"   Average confidence : {avg_conf:.3f}")
        print(f"   Low-confidence (<0.65): {low_conf}/{len(all_confs)} items")

    # ── Match score stats ──────────────────────────────────────────
    all_scores = [
        item["nutrients"].get("match_score", 0)
        for r in results
        for item in r.nutrition_results
        if item["nutrients"].get("matched_name")
    ]
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        low_score = sum(1 for s in all_scores if s < 80)
        print(f"\n🔍 Nutrition DB Match Score Stats:")
        print(f"   Average match score : {avg_score:.1f}")
        print(f"   Weak matches (<80)  : {low_score}/{len(all_scores)} items")

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "═" * 65)
    accuracy = (passed / total * 100) if total else 0
    bar_fill = int(accuracy / 5)
    bar      = "█" * bar_fill + "░" * (20 - bar_fill)
    print(f"  TOTAL:  {passed}/{total} passed  [{bar}]  {accuracy:.1f}%")
    print(f"  TIME:   {total_ms:.0f}ms total  |  {avg_ms:.0f}ms avg per test")
    print("═" * 65)

    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed — check failures above.")
    print()

    return passed, failed


# ─────────────────────────────────────────────────────────────────
# SAVE JSON REPORT
# ─────────────────────────────────────────────────────────────────

def save_json_report(results: list[TestResult], path: str = "e2e_report.json"):
    report = {
        "summary": {
            "total":   len(results),
            "passed":  sum(1 for r in results if r.passed),
            "failed":  sum(1 for r in results if not r.passed),
            "avg_ms":  sum(r.duration_ms for r in results) / max(len(results), 1),
        },
        "tests": [
            {
                "label":       r.test_case.label,
                "category":    r.test_case.category,
                "passed":      r.passed,
                "duration_ms": round(r.duration_ms, 2),
                "items_found": len(r.nutrition_results),
                "failures":    r.failures,
                "results": [
                    {
                        "food":       item["food"],
                        "weight":     item["weight"],
                        "source":     item["source"],
                        "confidence": round(item["confidence"], 3),
                        "calories":   round(item["nutrients"].get("calories") or 0, 1),
                        "protein":    round(item["nutrients"].get("protein")  or 0, 1),
                        "carbs":      round(item["nutrients"].get("carbs")    or 0, 1),
                        "fat":        round(item["nutrients"].get("fat")      or 0, 1),
                        "db_match":   item["nutrients"].get("matched_name"),
                        "match_score": round(item["nutrients"].get("match_score") or 0, 1),
                    }
                    for item in r.nutrition_results
                ],
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📄 JSON report saved → {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NutraDiet26 End-to-End Test Suite")
    parser.add_argument("--quick",   action="store_true", help="Run only text_only + edge_cases")
    parser.add_argument("--verbose", action="store_true", help="Show full details for every test")
    parser.add_argument("--category", type=str, help="Run only a specific category")
    parser.add_argument("--json",    type=str, default="e2e_report.json", help="Path for JSON report")
    args = parser.parse_args()

    # ── Filter test cases ─────────────────────────────────────────
    cases = TEST_CASES
    if args.quick:
        cases = [tc for tc in cases if tc.category in ("text_only", "edge_cases")]
        print("🚀 Quick mode: running text_only + edge_cases only\n")
    elif args.category:
        cases = [tc for tc in cases if tc.category == args.category]
        print(f"🔍 Category filter: '{args.category}'\n")

    if not cases:
        print("❌ No test cases matched the filter.")
        sys.exit(1)

    print(f"🧪 Running {len(cases)} end-to-end tests for NutraDiet26...\n")

    # ── Run tests ─────────────────────────────────────────────────
    results = []
    for i, tc in enumerate(cases, 1):
        print(f"  [{i:02d}/{len(cases):02d}] {tc.label[:55]}...", end=" ", flush=True)
        result = run_test(tc, verbose=args.verbose)
        results.append(result)
        print("✅" if result.passed else f"❌ ({len(result.failures)} failure(s))")

    # ── Print report ──────────────────────────────────────────────
    passed, failed = print_report(results, verbose=args.verbose)

    # ── Save JSON ─────────────────────────────────────────────────
    save_json_report(results, path=args.json)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
