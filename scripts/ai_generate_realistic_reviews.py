"""
================================================================================
ARCHETYPE-BASED REALISTIC REVIEW GENERATOR
================================================================================
Generates reviews based on REAL customer archetypes, not artificial emotions.

Archetypes reflect actual buying behavior:
- Enthusiast fan who loves the product
- Practical buyer focused on value
- Disappointed customer who expected more
- Casual satisfied customer
- Gift buyer
- Long-term subscriber
- First-time buyer
- Canceller explaining why they left

Each product gets 3 reviews from different archetypes, weighted by avg rating.

Output: 10,173 realistic reviews indistinguishable from human reviews

Usage:
    python ai_generate_realistic_reviews.py
================================================================================
"""

import json
import random
import time
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    print("❌ Missing dependency: python-dotenv")
    print("   Install deps in the project venv and run with it:")
    print("   python3 -m venv .venv")
    print("   .venv/bin/pip install -r requirements.txt")
    print("   .venv/bin/python3 scripts/ai_generate_realistic_reviews.py")
    raise

# Project root (repo): scripts/ -> parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/input/meta_Magazine_Subscriptions.jsonl"
OUTPUT_FILE = "data/output/ai_reviews_realistic.jsonl"
PROGRESS_FILE = "data/output/realistic_progress.json"
STATS_FILE = "data/output/realistic_stats.json"

PRODUCTS_PER_BATCH = 10
REVIEWS_PER_PRODUCT = 3
DELAY_BETWEEN_BATCHES = 2.5

API_PROVIDER = "groq"
MODEL = "llama-3.3-70b-versatile"

MULTI_MODEL = True
MODELS = [
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "llama-3.1-8b-instant"),
]


# ============================================================================
# CUSTOMER ARCHETYPES
# ============================================================================

ARCHETYPES = {
    # POSITIVE ARCHETYPES (tend toward 4-5 stars)
    "enthusiast": {
        "description": "Passionate fan who loves the product",
        "rating_range": (4, 5),
        "traits": [
            "Uses caps for emphasis (LOVE, AMAZING)",
            "Shares personal context for why they need it",
            "Mentions specific features they enjoy",
            "Recommends to others",
            "May have minor typos from typing fast",
            "Genuine excitement, not marketing speak"
        ],
        "example": "This mag is great! I LOVE ABC soaps. My college classes make me miss most episodes... I highly reccomend it."
    },
    
    "loyal_subscriber": {
        "description": "Long-term customer sharing their experience",
        "rating_range": (4, 5),
        "traits": [
            "Mentions how long they've subscribed",
            "Notes changes over time (good or bad)",
            "Feels ownership/connection to product",
            "Compares past vs present",
            "May mention renewal"
        ],
        "example": "Been subscribing for 5 years now. Still look forward to every issue. Quality has stayed consistent."
    },
    
    "gift_buyer": {
        "description": "Bought as gift for someone else",
        "rating_range": (4, 5),
        "traits": [
            "Mentions recipient (mom, dad, spouse, friend)",
            "Focuses on recipient's reaction",
            "May not know product details themselves",
            "Often brief and positive"
        ],
        "example": "Got this for my mother in law and she absolutely loves it. Perfect gift for her."
    },
    
    "casual_satisfied": {
        "description": "Happy but not overly enthusiastic",
        "rating_range": (4, 5),
        "traits": [
            "Brief and to the point",
            "Simple positive statements",
            "No detailed analysis",
            "Matter-of-fact tone"
        ],
        "example": "Good magazine. Enjoy reading it every month. Worth the subscription."
    },
    
    # NEUTRAL ARCHETYPES (tend toward 3 stars)
    "practical_buyer": {
        "description": "Focuses on value and utility",
        "rating_range": (3, 4),
        "traits": [
            "Mentions price/value",
            "Lists pros and cons",
            "Objective assessment",
            "Would recommend with caveats"
        ],
        "example": "Decent magazine for the price. Content is okay, not amazing. Good if you want light reading."
    },
    
    "comparison_shopper": {
        "description": "Compares to alternatives",
        "rating_range": (2, 4),
        "traits": [
            "References other magazines/products",
            "Explains why chose this one",
            "Notes differences",
            "May be switching from competitor"
        ],
        "example": "Not as comprehensive as Magazine X but cheaper. Gets the job done if you don't need all the extras."
    },
    
    "first_timer": {
        "description": "New customer sharing initial impressions",
        "rating_range": (3, 5),
        "traits": [
            "Mentions it's their first issue/purchase",
            "Initial impressions, not long-term view",
            "May express surprise (good or bad)",
            "Still forming opinion"
        ],
        "example": "Just got my first issue. So far so good, seems like quality content. Will update after a few more."
    },
    
    "mixed_feelings": {
        "description": "Genuinely conflicted about the product",
        "rating_range": (2, 4),
        "traits": [
            "Uses 'but' and 'however' frequently",
            "Acknowledges both good and bad",
            "Struggles to give clear recommendation",
            "Nuanced view"
        ],
        "example": "I want to love this magazine, I really do. The articles are great but the delivery is always late. Hard to recommend."
    },
    
    # NEGATIVE ARCHETYPES (tend toward 1-2 stars)
    "disappointed": {
        "description": "Expected more, feeling let down",
        "rating_range": (1, 3),
        "traits": [
            "Mentions expectations vs reality",
            "Feels misled or underwhelmed",
            "May reference reviews/marketing that set expectations",
            "Tone of letdown rather than anger"
        ],
        "example": "Based on other reviews I expected a lot more. The content is thin and repetitive. Disappointed."
    },
    
    "canceller": {
        "description": "Explaining why they cancelled/are cancelling",
        "rating_range": (1, 3),
        "traits": [
            "Past tense or explaining decision",
            "Lists reasons for leaving",
            "May mention how long they tried",
            "Sometimes wistful (used to be good)"
        ],
        "example": "Cancelled after 6 months. Quality went downhill, same articles recycled. Not worth it anymore."
    },
    
    "frustrated_complainer": {
        "description": "Angry about specific issues",
        "rating_range": (1, 2),
        "traits": [
            "Specific complaints (delivery, billing, quality)",
            "May use caps for emphasis",
            "Warns others",
            "Demands better"
        ],
        "example": "NEVER received half my issues. Customer service was useless. Save your money."
    },
    
    "brief_negative": {
        "description": "Short negative review",
        "rating_range": (1, 2),
        "traits": [
            "Very short, to the point",
            "Blunt assessment",
            "No detailed explanation",
            "Quick dismissal"
        ],
        "example": "Waste of money. Don't bother."
    }
}


# ============================================================================
# ARCHETYPE SELECTION BASED ON RATING
# ============================================================================

def get_archetype_weights(avg_rating: float) -> Dict[str, float]:
    """
    Select archetypes weighted by product's average rating.
    High-rated products get more positive archetypes.
    Low-rated products get more negative archetypes.
    """
    
    normalized = (avg_rating - 1) / 4  # 0-1 scale
    
    if normalized >= 0.875:  # ≥4.5 stars
        return {
            "enthusiast": 0.25,
            "loyal_subscriber": 0.20,
            "casual_satisfied": 0.20,
            "gift_buyer": 0.10,
            "first_timer": 0.10,
            "practical_buyer": 0.08,
            "mixed_feelings": 0.04,
            "disappointed": 0.02,
            "canceller": 0.01,
            "frustrated_complainer": 0.00,
            "brief_negative": 0.00,
            "comparison_shopper": 0.00
        }
    elif normalized >= 0.75:  # ≥4.0 stars
        return {
            "enthusiast": 0.18,
            "loyal_subscriber": 0.15,
            "casual_satisfied": 0.18,
            "gift_buyer": 0.08,
            "first_timer": 0.10,
            "practical_buyer": 0.12,
            "mixed_feelings": 0.08,
            "disappointed": 0.05,
            "comparison_shopper": 0.04,
            "canceller": 0.02,
            "frustrated_complainer": 0.00,
            "brief_negative": 0.00
        }
    elif normalized >= 0.625:  # ≥3.5 stars
        return {
            "enthusiast": 0.10,
            "loyal_subscriber": 0.10,
            "casual_satisfied": 0.12,
            "practical_buyer": 0.15,
            "mixed_feelings": 0.15,
            "first_timer": 0.10,
            "comparison_shopper": 0.08,
            "disappointed": 0.10,
            "canceller": 0.05,
            "gift_buyer": 0.03,
            "frustrated_complainer": 0.02,
            "brief_negative": 0.00
        }
    elif normalized >= 0.5:  # ≥3.0 stars
        return {
            "practical_buyer": 0.15,
            "mixed_feelings": 0.18,
            "disappointed": 0.15,
            "first_timer": 0.10,
            "comparison_shopper": 0.10,
            "canceller": 0.10,
            "casual_satisfied": 0.08,
            "enthusiast": 0.05,
            "frustrated_complainer": 0.05,
            "brief_negative": 0.02,
            "loyal_subscriber": 0.02,
            "gift_buyer": 0.00
        }
    elif normalized >= 0.375:  # ≥2.5 stars
        return {
            "disappointed": 0.22,
            "mixed_feelings": 0.15,
            "canceller": 0.15,
            "frustrated_complainer": 0.12,
            "practical_buyer": 0.10,
            "comparison_shopper": 0.08,
            "brief_negative": 0.08,
            "first_timer": 0.05,
            "casual_satisfied": 0.03,
            "enthusiast": 0.02,
            "loyal_subscriber": 0.00,
            "gift_buyer": 0.00
        }
    else:  # <2.5 stars
        return {
            "frustrated_complainer": 0.25,
            "disappointed": 0.20,
            "canceller": 0.18,
            "brief_negative": 0.15,
            "mixed_feelings": 0.10,
            "comparison_shopper": 0.07,
            "practical_buyer": 0.05,
            "first_timer": 0.00,
            "casual_satisfied": 0.00,
            "enthusiast": 0.00,
            "loyal_subscriber": 0.00,
            "gift_buyer": 0.00
        }


def select_archetypes_for_product(avg_rating: float, count: int = 3) -> List[str]:
    """Select diverse archetypes for a product."""
    
    weights = get_archetype_weights(avg_rating)
    archetypes = list(weights.keys())
    probs = list(weights.values())
    
    # Normalize probabilities
    total = sum(probs)
    probs = [p/total for p in probs]
    
    selected = []
    available = archetypes.copy()
    available_probs = probs.copy()
    
    for _ in range(count):
        if not available:
            break
            
        # Renormalize
        total = sum(available_probs)
        if total == 0:
            break
        norm_probs = [p/total for p in available_probs]
        
        choice = random.choices(available, weights=norm_probs)[0]
        selected.append(choice)
        
        # Remove to avoid duplicates
        idx = available.index(choice)
        available.pop(idx)
        available_probs.pop(idx)
    
    return selected


def get_rating_for_archetype(archetype: str, avg_rating: float) -> float:
    """Generate rating based on archetype and product average."""
    
    config = ARCHETYPES[archetype]
    min_r, max_r = config["rating_range"]
    
    # Add some variance based on product's actual average
    if archetype in ["enthusiast", "loyal_subscriber", "casual_satisfied", "gift_buyer"]:
        # Positive archetypes - bias toward higher end
        if avg_rating >= 4.5:
            return random.choices([4.0, 4.5, 5.0], weights=[0.1, 0.2, 0.7])[0]
        elif avg_rating >= 4.0:
            return random.choices([4.0, 4.5, 5.0], weights=[0.2, 0.4, 0.4])[0]
        else:
            return random.choices([4.0, 4.5, 5.0], weights=[0.4, 0.4, 0.2])[0]
    
    elif archetype in ["practical_buyer", "comparison_shopper", "first_timer", "mixed_feelings"]:
        # Neutral archetypes
        return random.choices([2.0, 2.5, 3.0, 3.5, 4.0], weights=[0.05, 0.15, 0.40, 0.25, 0.15])[0]
    
    elif archetype in ["disappointed", "canceller"]:
        # Negative but measured
        if avg_rating <= 3.0:
            return random.choices([1.0, 1.5, 2.0, 2.5], weights=[0.3, 0.3, 0.3, 0.1])[0]
        else:
            return random.choices([2.0, 2.5, 3.0], weights=[0.4, 0.4, 0.2])[0]
    
    elif archetype in ["frustrated_complainer", "brief_negative"]:
        # Very negative
        return random.choices([1.0, 1.5, 2.0], weights=[0.5, 0.3, 0.2])[0]
    
    return 3.0


# ============================================================================
# PROMPT GENERATION
# ============================================================================

SYSTEM_PROMPT = """

You generate realistic customer reviews that are indistinguishable from real human reviews.

CRITICAL - Write like REAL humans:
- Vary title length naturally (1 word to 15+ words)
- Include personal context and stories
- Use casual language, contractions, occasional typos
- Some reviews are short fragments, others are detailed
- Mix proper grammar with casual internet writing
- Reference specific product features when you know them
- Express genuine emotions, not templated sentiments
- NO marketing speak, NO "Overall", NO "In conclusion"

Each review has an ARCHETYPE that defines the customer's perspective:
- Match the archetype's traits and example style
- Generate a rating within the archetype's typical range
- Make each review feel like a unique person wrote it

OUTPUT: JSON array with each review containing:
- index (product number)  
- archetype (the customer type)
- rating (float, based on archetype)
- title (natural length - can be 1 word or 20 words)
- text (the review body - natural length)
- asin (product ID)"""


def create_batch_prompt(products_batch: List[Dict]) -> str:
    """Create prompt for batch with archetype specifications."""
    
    total_reviews = len(products_batch) * REVIEWS_PER_PRODUCT
    prompt = f"Generate {total_reviews} authentic customer reviews for {len(products_batch)} magazine subscriptions.\n\n"
    
    for i, product in enumerate(products_batch):
        title = product.get('title', 'Magazine')[:60]
        asin = product.get('parent_asin', '')
        avg = product.get('average_rating', 3.5)
        store = product.get('store', '')
        desc = product.get('description', [])
        
        # Clean description
        if isinstance(desc, list):
            desc_text = ' '.join(str(d) for d in desc)[:150]
        else:
            desc_text = str(desc)[:150] if desc else ''
        
        # Select archetypes
        archetypes = select_archetypes_for_product(avg, REVIEWS_PER_PRODUCT)
        product['_archetypes'] = archetypes
        
        prompt += f"━━━ Product {i}: {title} ━━━\n"
        prompt += f"ASIN: {asin} | Avg: {avg}★"
        if store:
            prompt += f" | {store}"
        prompt += "\n"
        if desc_text:
            prompt += f"{desc_text}\n"
        
        prompt += "\nWrite 3 reviews from these customer perspectives:\n"
        
        for j, archetype in enumerate(archetypes):
            config = ARCHETYPES[archetype]
            rating = get_rating_for_archetype(archetype, avg)
            product[f'_rating_{j}'] = rating
            product[f'_archetype_{j}'] = archetype
            
            traits = random.sample(config['traits'], min(2, len(config['traits'])))
            prompt += f"\n  [{i}.{j}] {archetype.upper()} ({rating}★)\n"
            prompt += f"       Style: {config['description']}\n"
            prompt += f"       Traits: {'; '.join(traits)}\n"
        
        prompt += "\n"
    
    prompt += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON array. No explanation.
Each object: {index, archetype, rating, title, text, asin}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return prompt


# ============================================================================
# API FUNCTIONS
# ============================================================================

def call_groq(prompt: str, model: str, api_key: str) -> Optional[str]:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=6000,
            temperature=0.9  # Higher for more variety
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" Error: {e}")
        return None


def call_together(prompt: str, model: str, api_key: str) -> Optional[str]:
    try:
        from together import Together
        client = Together(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=6000,
            temperature=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" Error: {e}")
        return None


def call_ollama(prompt: str, model: str) -> Optional[str]:
    try:
        import ollama
        
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': f"{SYSTEM_PROMPT}\n\n{prompt}"}],
            options={'temperature': 0.9, 'num_predict': 6000}
        )
        return response['message']['content']
    except Exception as e:
        print(f" Error: {e}")
        return None


def call_api(prompt: str, provider: str, model: str) -> Optional[str]:
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        return call_groq(prompt, model, api_key) if api_key else None
    elif provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY")
        return call_together(prompt, model, api_key) if api_key else None
    elif provider == "ollama":
        return call_ollama(prompt, model)
    return None


# ============================================================================
# PARSING & PROCESSING
# ============================================================================

def parse_response(text: str) -> Optional[List[Dict]]:
    if not text:
        return None
    
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else None
    except:
        return None


def generate_timestamp() -> int:
    start = datetime(2020, 1, 1)
    end = datetime(2025, 12, 31)
    delta = random.randint(0, int((end - start).total_seconds()))
    return int((start + timedelta(seconds=delta)).timestamp() * 1000)


def complete_review(review: Dict, product: Dict, model: str) -> Dict:
    """Build complete review record."""
    
    archetype = review.get('archetype', 'unknown')
    asin = product.get('parent_asin', '')
    rating = float(review.get('rating', 3.0))
    
    # Helpful votes vary by archetype
    if archetype in ['enthusiast', 'loyal_subscriber', 'frustrated_complainer']:
        helpful = random.choices([0, 1, 2, 3, 5, 8, 12], weights=[0.2, 0.2, 0.2, 0.15, 0.12, 0.08, 0.05])[0]
    elif archetype in ['detailed_critic', 'comparison_shopper']:
        helpful = random.choices([0, 1, 2, 4, 6, 10], weights=[0.15, 0.2, 0.25, 0.2, 0.12, 0.08])[0]
    else:
        helpful = random.choices([0, 1, 2, 3], weights=[0.5, 0.25, 0.15, 0.1])[0]
    
    return {
        "rating": rating,
        "title": review.get('title', '')[:200],
        "text": review.get('text', '')[:3000],
        "images": [],
        "asin": asin,
        "parent_asin": asin,
        "timestamp": generate_timestamp(),
        "verified_purchase": False,
        "helpful_vote": helpful,
        "_source": "ai_generated",
        "_model": model,
        "_archetype": archetype,
        "_product_title": product.get('title', '')[:60],
        "_product_avg_rating": product.get('average_rating', 0)
    }


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress() -> Dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_batches": [], "reviews_generated": 0}


def save_progress(progress: Dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def save_stats(stats: Dict):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("═" * 70)
    print("   ARCHETYPE-BASED REALISTIC REVIEW GENERATOR")
    print("   Human-like reviews from diverse customer perspectives")
    print("═" * 70)
    
    # Load metadata
    print(f"\n📦 Loading {INPUT_FILE}...")
    products = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                products.append(json.loads(line.strip()))
            except:
                continue
    
    total_products = len(products)
    total_reviews = total_products * REVIEWS_PER_PRODUCT
    
    print(f"   ✓ {total_products:,} products")
    print(f"   ✓ {total_reviews:,} reviews to generate")
    
    # Create batches
    batches = [products[i:i+PRODUCTS_PER_BATCH] 
               for i in range(0, len(products), PRODUCTS_PER_BATCH)]
    total_batches = len(batches)
    
    print(f"   ✓ {total_batches} batches ({PRODUCTS_PER_BATCH} products × {REVIEWS_PER_PRODUCT} reviews)")
    
    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed_batches", []))
    
    if completed:
        print(f"   ✓ Resuming: {len(completed)} batches done")
    
    # Time estimate
    remaining = total_batches - len(completed)
    est_min = (remaining * DELAY_BETWEEN_BATCHES) / 60
    print(f"\n⏱️  Est. time: {est_min:.1f} min ({remaining} batches)")
    
    # Stats (generated starts from prior run when resuming so progress file stays correct)
    prior_generated = int(progress.get("reviews_generated", 0) or 0)
    session_start_generated = prior_generated
    stats = {
        "total_products": total_products,
        "total_target": total_reviews,
        "generated": prior_generated,
        "by_archetype": {},
        "by_rating": {},
        "by_model": {},
        "failed": []
    }
    
    mode = 'a' if completed else 'w'
    model_idx = 0
    start = time.time()
    
    print(f"\n🔄 Generating...\n")
    
    with open(OUTPUT_FILE, mode, encoding='utf-8') as out:
        
        for batch_idx, batch in enumerate(batches):
            if batch_idx in completed:
                continue
            
            # Model selection
            if MULTI_MODEL:
                provider, model = MODELS[model_idx % len(MODELS)]
                model_idx += 1
            else:
                provider, model = API_PROVIDER, MODEL
            
            # Progress
            pct = 100 * (batch_idx + 1) / total_batches
            elapsed = time.time() - start
            done = batch_idx - len(completed) + 1
            eta = (elapsed / max(done, 1)) * (total_batches - batch_idx - 1)
            
            model_name = model.split('/')[-1][:12]
            print(f"   [{batch_idx+1:3d}/{total_batches}] {pct:5.1f}% | "
                  f"{len(batch)*3} reviews | {model_name} | "
                  f"ETA: {eta/60:.1f}m", end="", flush=True)
            
            # Generate
            prompt = create_batch_prompt(batch)
            response = call_api(prompt, provider, model)
            reviews = parse_response(response)
            
            if reviews:
                count = 0
                for review in reviews:
                    raw_idx = review.get("index", 0)
                    try:
                        idx = int(raw_idx)
                    except (TypeError, ValueError):
                        # Some models return "index" as non-numeric; skip those entries.
                        continue

                    if 0 <= idx < len(batch):
                        completed_review = complete_review(review, batch[idx], model)
                        out.write(json.dumps(completed_review, ensure_ascii=False) + '\n')
                        
                        # Stats
                        arch = completed_review.get('_archetype', 'unknown')
                        rat = str(completed_review.get('rating', 3.0))
                        stats["by_archetype"][arch] = stats["by_archetype"].get(arch, 0) + 1
                        stats["by_rating"][rat] = stats["by_rating"].get(rat, 0) + 1
                        stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
                        stats["generated"] += 1
                        count += 1
                
                print(f" ✓ {count}")
                
                completed.add(batch_idx)
                progress["completed_batches"] = list(completed)
                progress["reviews_generated"] = stats["generated"]
                save_progress(progress)
            else:
                print(f" ✗ FAILED")
                stats["failed"].append(batch_idx)
            
            if batch_idx < total_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)
    
    # Summary
    total_time = time.time() - start
    
    print("\n" + "═" * 70)
    print("   COMPLETE!")
    print("═" * 70)
    print(f"\n📊 Total in file: {stats['generated']:,} reviews "
          f"(+{stats['generated'] - session_start_generated:,} this run)")
    print(f"⏱️  Time: {total_time/60:.1f} minutes")
    print(f"📁 Output: {OUTPUT_FILE}")
    
    session_n = max(stats["generated"] - session_start_generated, 1)
    print(f"\n👥 Archetype Distribution (this run):")
    sorted_arch = sorted(stats["by_archetype"].items(), key=lambda x: -x[1])
    for arch, count in sorted_arch[:8]:
        pct = 100 * count / session_n
        bar = '█' * int(pct / 2)
        print(f"   {arch:20s} {bar} {count:,} ({pct:.1f}%)")
    
    print(f"\n⭐ Rating Distribution (this run):")
    for r in ["5.0", "4.5", "4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0"]:
        count = stats["by_rating"].get(r, 0)
        pct = 100 * count / session_n
        bar = '█' * int(pct / 2)
        print(f"   {r}★ {bar} {count:,} ({pct:.1f}%)")
    
    if stats["failed"]:
        print(f"\n⚠️  Failed: {len(stats['failed'])} batches (re-run to retry)")
    
    save_stats(stats)
    
    if not stats["failed"] and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    print(f"\n🎉 Done!")


if __name__ == "__main__":
    if API_PROVIDER == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("❌ Set GROQ_API_KEY in .env at project root or export it:")
        print(f"   {_PROJECT_ROOT / '.env'}")
        print("   Get free: https://console.groq.com")
        exit(1)
    
    main()