# --- Shared calorie guides (estimates) ---
# Note: values are typical averages for cooked foods; brands/recipes & added oils will vary.

BASE_BREAKFAST_OPTIONS_WITH_CALORIES = {
    "Option 1": {
        "text": "4 egg whites + 1 yolk (Boiled/Soft boiled/Scrambled)",
        "calories_kcal": 123,
    },
    "Option 2": {
        "text": "150g chicken breast (Grilled) - Can marinate with salt, pepper, or spices",
        "calories_kcal": 248,
    },
    "Option 3": {
        "text": "150g Ayam Brand Tuna Chunk in water (Squeeze 1 lime)",
        "calories_kcal": 174,
    },
    "Option 4": {
        "text": "200g Tempeh (Lightly fried/air fryer)",
        "calories_kcal": 386,
    },
    "Option 5": {
        "instructions": "Prepare the night before; eat for breakfast",
        "ingredients": [
            "150ml almond/coconut milk",
            "¼ cup chia seeds",
            "1 scoop sugar-free protein powder",
            "½ cup fresh/frozen berries (raspberries, strawberries, blueberries)",
        ],
        "calories_kcal": 413,
    },
    "Option 6": {
        "ingredients": [
            "½ cup frozen raspberries",
            "1 scoop sugar-free protein powder",
            "150ml coconut/almond milk",
        ],
        "calories_kcal": 180,
    },
}

MEDITERRANEAN_CALORIE_GUIDE = {
    # 120 g cooked protein choices
    "ProteinPer120g_kcal": {
        "ChickenBreast": 198,
        "LeanFish": 144,
        "LeanBeef": 300,
    },
    # 120 g cooked gentle starch
    "CarbPer120g_kcal": {
        "WhiteRice": 156,
        "Pasta": 157,
        "BoiledPotato": 104,
    },
    # 240 g non-starchy veg (typical mixed veg)
    "NonStarchyVegPer240g_kcal": 60,
    # Add-ons (choose one)
    "AddOn_kcal": {
        "Nuts": {"Almond_20g": 116, "Walnut_20g": 131},
        "DarkChocolate_10g": 60,
        "ChiaSeeds_30g": 146,
    },
    # Optional fruit (choose one)
    "Fruit_kcal": {"Apple_1medium": 95, "Orange_1medium": 62},
    # Healthy fat (if used as 1 tbsp)
    "HealthyFat_kcal": {"OliveOil_1tbsp": 119},
}


def _get_addon_kcal(add_on: str | None) -> int:
    """
    add_on can be:
      - "Nuts.Almond_20g"
      - "Nuts.Walnut_20g"
      - "DarkChocolate_10g"
      - "ChiaSeeds_30g"
      - None
    """
    if not add_on:
        return 0
    guide = MEDITERRANEAN_CALORIE_GUIDE["AddOn_kcal"]
    if "." in add_on:
        group, key = add_on.split(".", 1)
        return int(guide.get(group, {}).get(key, 0))
    return int(guide.get(add_on, 0))


def calc_mediterranean_meal_kcal(
    *,
    protein: str = "ChickenBreast",
    carb: str = "WhiteRice",
    add_on: str | None = "Nuts.Almond_20g",
    include_fruit: bool = False,
    fruit: str = "Apple_1medium",
    include_oil: bool = False,
) -> dict:
    """
    Returns a breakdown and total kcal for one 'PCOS Plate' (120g protein + 120g carb + 240g veg + add-on [+ fruit] [+ 1 tbsp oil]).
    """
    P = int(MEDITERRANEAN_CALORIE_GUIDE["ProteinPer120g_kcal"].get(protein, 0))
    C = int(MEDITERRANEAN_CALORIE_GUIDE["CarbPer120g_kcal"].get(carb, 0))
    V = int(MEDITERRANEAN_CALORIE_GUIDE["NonStarchyVegPer240g_kcal"])
    A = _get_addon_kcal(add_on)
    F = int(MEDITERRANEAN_CALORIE_GUIDE["Fruit_kcal"].get(fruit, 0)) if include_fruit else 0
    O = int(MEDITERRANEAN_CALORIE_GUIDE["HealthyFat_kcal"]["OliveOil_1tbsp"]) if include_oil else 0

    total = P + C + V + A + F + O
    return {
        "protein": {"choice": protein, "kcal": P},
        "carb": {"choice": carb, "kcal": C},
        "veg_240g": {"kcal": V},
        "add_on": {"choice": add_on, "kcal": A},
        "fruit": {"included": include_fruit, "choice": fruit if include_fruit else None, "kcal": F},
        "oil_1tbsp": {"included": include_oil, "kcal": O},
        "total_kcal": total,
    }


def get_breakfast_kcal(option_key: str) -> dict:
    """
    option_key: e.g., "Option 2"
    """
    opt = BASE_BREAKFAST_OPTIONS_WITH_CALORIES.get(option_key)
    if not opt:
        raise KeyError(f"Unknown breakfast option '{option_key}'. Valid keys: {list(BASE_BREAKFAST_OPTIONS_WITH_CALORIES)}")
    return {"option": option_key, "kcal": int(opt["calories_kcal"])}


def calc_day_kcal(
    *,
    breakfast_option: str,
    lunch_kwargs: dict,
    dinner_kwargs: dict,
) -> dict:
    """
    Compute a day's calories: breakfast + lunch + dinner (each lunch/dinner is a 'PCOS Plate' combo).
    """
    b = get_breakfast_kcal(breakfast_option)
    l = calc_mediterranean_meal_kcal(**lunch_kwargs)
    d = calc_mediterranean_meal_kcal(**dinner_kwargs)
    total = b["kcal"] + l["total_kcal"] + d["total_kcal"]
    return {"breakfast": b, "lunch": l, "dinner": d, "day_total_kcal": total}


FEMALE_MEAL_PLANS = {
    "PCOS Insulin Resistance": {
        "SugarFreeBreakfast": {
            "Guidelines": {
                "Timing": "Consume 1 hour after waking up",
                "Avoid": ["Sugar", "Honey", "Stevia"],
                "Goal": "30-40g clean protein to stabilize blood sugar",
            },
            "Options": BASE_BREAKFAST_OPTIONS_WITH_CALORIES,
        },
        "MediterraneanDiet": {
            "PCOSPlate": {
                "Protein": "25% animal protein (palm-size portion)",
                "GentleStarch": "25% gentle starch (~½ cup cooked)",
                "NonStarchyVeg": "50% non-starchy vegetables",
                "HealthyFat": "1 tbsp healthy fat",
            },
            "SuggestedMenu": [
                "120g chicken/fish/meat",
                "120g rice/pasta/potato",
                "240g non-starchy vegetables",
                "20g nuts (Almond/Walnut) / 10g Dark Chocolate (80%+) / 30g Chia Seeds",
                "1 apple/orange (if craving sweets)",
            ],
            "CaloriesGuide": MEDITERRANEAN_CALORIE_GUIDE,
            "CookingTips": "Reduce oil, avoid sugar, no fast food or MSG for the first 4 weeks",
        },
    },
    "PCOS Adrenal": {
        "SugarFreeBreakfast": {
            "Focus": "High protein, low carbohydrate",
            "Rules": {
                "Timing": "Consume 1 hour after waking up",
                "Avoid": ["Sugar", "Honey", "Stevia", "Other sweeteners"],
            },
            "Options": BASE_BREAKFAST_OPTIONS_WITH_CALORIES,
        },
        "MediterraneanDiet": {
            "PCOSPlate": {
                "Protein": "25% animal protein (palm-size)",
                "GentleStarch": "25% gentle starch (~½ cup cooked)",
                "NonStarchyVeg": "50% non-starchy vegetables (as much as possible)",
                "HealthyFat": "1 tbsp healthy fat",
            },
            "SuggestedMenu": [
                "120g chicken/fish/meat",
                "120g rice/pasta/potato (can replace with brown rice/basmati/wholemeal bread)",
                "240g non-starchy vegetables",
                "20g nuts (Almond/Walnut) / 10g dark chocolate (80%+) / 30g chia seeds",
                "1 apple/orange (if craving sweets)",
            ],
            "CaloriesGuide": MEDITERRANEAN_CALORIE_GUIDE,
            "CookingTips": "Less oil, no sugar, no fast food or MSG for the first 4 weeks",
        },
    },
    "PCOS Inflammation": {
        "SugarFreeBreakfast": {
            "Guidelines": {
                "Timing": "Consume 1 hour after waking up",
                "Avoid": ["Sugar", "Honey", "Stevia"],
                "Goal": "30-40g clean protein to stabilize blood sugar",
            },
            "Options": BASE_BREAKFAST_OPTIONS_WITH_CALORIES,
        },
        "MediterraneanDiet": {
            "PCOSPlate": {
                "Protein": "25% animal protein (palm-size portion)",
                "GentleStarch": "25% gentle starch (~½ cup cooked)",
                "NonStarchyVeg": "50% non-starchy vegetables",
                "HealthyFat": "1 tbsp healthy fat",
            },
            "SuggestedMenu": [
                "120g chicken/fish/meat",
                "120g rice/pasta/potato",
                "240g non-starchy vegetables",
                "20g nuts (Almond/Walnut) / 10g Dark Chocolate (80%+) / 30g Chia Seeds",
                "1 apple/orange (if craving sweets)",
            ],
            "CaloriesGuide": MEDITERRANEAN_CALORIE_GUIDE,
            "CookingTips": "Reduce oil, avoid sugar, no fast food or MSG for the first 4 weeks",
        },
    },
    "PCOS Post Birth Control": {
        "SugarFreeBreakfast": {
            "Focus": "High protein, low carbohydrate",
            "Rules": {
                "Timing": "Consume 1 hour after waking up",
                "Avoid": ["Sugar", "Honey", "Stevia", "Other sweeteners"],
            },
            "Options": BASE_BREAKFAST_OPTIONS_WITH_CALORIES,
        },
        "MediterraneanDiet": {
            "PCOSPlate": {
                "Protein": "25% animal protein (palm-size)",
                "GentleStarch": "25% gentle starch (~½ cup cooked)",
                "NonStarchyVeg": "50% non-starchy vegetables (as much as possible)",
                "HealthyFat": "1 tbsp healthy fat",
            },
            "SuggestedMenu": [
                "120g chicken/fish/meat",
                "120g rice/pasta/potato (can replace with brown rice/basmati/wholemeal bread)",
                "240g non-starchy vegetables",
                "20g nuts (Almond/Walnut) / 10g dark chocolate (80%+) / 30g chia seeds",
                "1 apple/orange (if craving sweets)",
            ],
            "CaloriesGuide": MEDITERRANEAN_CALORIE_GUIDE,
            "CookingTips": "Less oil, no sugar, no fast food or MSG for the first 4 weeks",
        },
    },
}


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Breakfast: Option 2 (150g grilled chicken)
    b = get_breakfast_kcal("Option 2")

    # Lunch: Chicken + Rice + Veg + Almonds + Apple (no oil)
    lunch = calc_mediterranean_meal_kcal(
        protein="ChickenBreast",
        carb="WhiteRice",
        add_on="Nuts.Almond_20g",
        include_fruit=True,
        include_oil=False,
    )

    # Dinner: LeanFish + Pasta + Veg + Dark Chocolate (no fruit, with oil)
    dinner = calc_mediterranean_meal_kcal(
        protein="LeanFish",
        carb="Pasta",
        add_on="DarkChocolate_10g",
        include_fruit=False,
        include_oil=True,
    )

    day = calc_day_kcal(
        breakfast_option="Option 2",
        lunch_kwargs=dict(
            protein="ChickenBreast",
            carb="WhiteRice",
            add_on="Nuts.Almond_20g",
            include_fruit=True,
            include_oil=False,
        ),
        dinner_kwargs=dict(
            protein="LeanFish",
            carb="Pasta",
            add_on="DarkChocolate_10g",
            include_fruit=False,
            include_oil=True,
        ),
    )

    print("Breakfast:", b)
    print("Lunch:", lunch)
    print("Dinner:", dinner)
    print("Day total kcal:", day["day_total_kcal"])

# Helper functions to work with meal plans
def get_female_meal_plan(goal: str):
    """
    Get a specific meal plan object by goal.

    Args:
        goal (str): One of
            - "PCOS Insulin Resistance"
            - "PCOS Adrenal"
            - "PCOS Inflammation"
            - "PCOS Post Birth Control"

    Returns:
        dict | None: The meal plan data for the goal (or None).
    """
    return FEMALE_MEAL_PLANS[goal]