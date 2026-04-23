#!/usr/bin/env python3
"""
Test script to verify all 27 nutrients are being returned by the API
"""

import sys
sys.path.insert(0, '/app')

from modules.nutrition import get_nutrition, load_databases

# Load databases
print("Loading databases...")
load_databases()
print("✓ Databases loaded")

# Test nutrition lookup
test_foods = ['dal makhani', 'apple', 'rice', 'chicken', 'milk']
expected_nutrients = [
    'energy_kcal', 'protein', 'carbs', 'fat', 'fiber', 'sugars',
    'saturated_fat', 'monounsaturated_fat', 'polyunsaturated_fat', 'cholesterol',
    'calcium', 'phosphorus', 'magnesium', 'sodium', 'potassium', 
    'iron', 'copper', 'selenium', 'zinc',
    'vitamin_a', 'vitamin_c', 'vitamin_e',
    'thiamin', 'riboflavin', 'niacin', 'vitamin_b6', 'folate'
]

print(f"\nTesting {len(test_foods)} foods for all {len(expected_nutrients)} nutrients:")
print("-" * 70)

all_passed = True
for food in test_foods:
    nutrients = get_nutrition(food)
    if not nutrients:
        print(f"✗ {food}: No nutrition data found")
        all_passed = False
        continue
    
    found_count = sum(1 for n in expected_nutrients if nutrients.get(n) is not None)
    matched_name = nutrients.get('matched_name', 'N/A')
    source = nutrients.get('source', 'N/A')
    
    print(f"✓ {food:20} → {matched_name:20} ({source:5}) | {found_count:2}/{len(expected_nutrients)} nutrients")
    
    # Show sample values
    if found_count > 0:
        print(f"  Energy: {nutrients.get('energy_kcal', 0):.1f}kcal | Protein: {nutrients.get('protein', 0):.1f}g | Fiber: {nutrients.get('fiber', 0):.1f}g")

print("-" * 70)
print(f"\n{'✓ All tests passed!' if all_passed else '✗ Some tests failed'}")
