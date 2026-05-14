# services/context_ollama.py

import json
import os
import re
import time
import math
import traceback
import urllib.request
import urllib.error
from collections import Counter
from dotenv import load_dotenv
from typing import Dict, List, Optional, Callable

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_CONFIG = {
    "model_name": "gemma2:27b",
    "timeout_seconds": 500,
    "retry_attempts": 5,
    "retry_delay_sec": 5,
    "api_url": "http://localhost:11434/api/generate",
}

# ---------------------------------------------------------------------------
# Prompt — built dynamically from DB parameters via parameter_service
# ---------------------------------------------------------------------------

def _build_prompt_template(call_category=None, call_type=None):
    """Build the Ollama analysis prompt dynamically from the parameter database."""
    from services.parameter_service import get_parameters, get_agent_mappings, get_auto_weightages

    categories = get_parameters(call_category, call_type)
    agent_mappings = get_agent_mappings(call_category, call_type)
    auto_weightages = get_auto_weightages(call_category, call_type)

    # Build checklist_results keys (numeric scoring: 0 to max points)
    checklist_keys = []
    for cat in categories:
        for p in cat["parameters"]:
            checklist_keys.append(f'    "{p["key"]}": <0-{p["weight"]}>')

    checklist_json = ",\n".join(checklist_keys)

    # Build call_quality factors (double-brace escaped for .format())
    # Uses auto-normalized weightages based on points
    factors_lines = []
    for cat in categories:
        w = auto_weightages.get(cat["key"], 0)
        factors_lines.append(
            f'      {{{{"name":"{cat["label"]}", "score":<0-100>,"details":"<string>",'
            f'"icon":"{cat["icon"]}","status":"<pass|warning|fail>","weightage":{w}}}}}'
        )
    factors_json = ",\n".join(factors_lines)

    # Build agent_behavior category_scores (double-brace escaped for .format())
    agent_cats_lines = []
    for m in agent_mappings:
        agent_cats_lines.append(
            f'      {{{{"category":"{m["agent_category"]}", "score":<0-100>,"weightage":{m["weightage"]}}}}}'
        )
    agent_cats_json = ",\n".join(agent_cats_lines)

    # Build evaluation guidelines
    eval_guidelines = []
    param_count = 0
    for cat in categories:
        for p in cat["parameters"]:
            param_count += 1
            guide = p.get("evaluation_guide") or "Evaluate based on transcript evidence."
            eval_guidelines.append(f'  * {p["name"]} ({p["key"]}): {guide}')
    eval_text = "\n".join(eval_guidelines)

    # Use string concatenation instead of f-string to avoid brace escaping issues
    # {transcript} is the only placeholder — replaced by .format() at call time
    prompt = (
        'You are an expert call-center quality analyst.\n'
        'Analyze the customer service call transcript below and return ONLY a single valid JSON object.\n'
        'No markdown, no code fences, no explanation — raw JSON only.\n\n'
        'TRANSCRIPT:\n"""{transcript}"""\n\n'
        'Return this EXACT JSON structure (fill every field):\n\n'
        '{{\n'
        '  "checklist_results": {{\n'
        + checklist_json + '\n'
        '  }},\n\n'
        '  "summary": {{\n'
        '    "overall_score": <integer 0-100>,\n'
        '    "overall_grade": "<A|B|C|D|F>",\n'
        '    "word_count": <integer>,\n'
        '    "turn_count": <integer>\n'
        '  }},\n\n'
        '  "sentiment": {{\n'
        '    "score": <integer -100 to 100>,\n'
        '    "label": "<Very Positive|Positive|Neutral|Negative|Very Negative>",\n'
        '    "emoji": "<😊|🙂|😐|😟|😠>",\n'
        '    "positive_words": [{{"word":"<string>","count":<int>}}],\n'
        '    "negative_words": [{{"word":"<string>","count":<int>}}],\n'
        '    "progression": [<5 integers -100 to 100 for Start,Early,Mid,Late,End>]\n'
        '  }},\n\n'
        '  "call_quality": {{\n'
        '    "score": <integer 0-100>,\n'
        '    "rating": "<Excellent|Good|Fair|Poor>",\n'
        '    "factors": [\n'
        + factors_json + '\n'
        '    ]\n'
        '  }},\n\n'
        '  "agent_behavior": {{\n'
        '    "score": <integer 0-100>,\n'
        '    "rating": "<Exceptional|Proficient|Developing|Needs Training>",\n'
        '    "strengths": [{{"text":"<string>","category":"<string>"}}],\n'
        '    "improvements": [{{"text":"<string>","category":"<string>","priority":"<high|medium>"}}],\n'
        '    "category_scores": [\n'
        + agent_cats_json + '\n'
        '    ],\n'
        '    "detailed_metrics": [\n'
        '      {{"name":"Empathy",              "score":<0-100>,"max":100}},\n'
        '      {{"name":"Professional Language","score":<0-100>,"max":100}},\n'
        '      {{"name":"Problem Resolution",   "score":<0-100>,"max":100}},\n'
        '      {{"name":"Call Control",         "score":<0-100>,"max":100}},\n'
        '      {{"name":"Customer Focus",       "score":<0-100>,"max":100}}\n'
        '    ]\n'
        '  }},\n\n'
        '  "sensitive_words": [\n'
        '    {{"category":"Religious Content",   "words":["<word>"],"severity":"low",     "count":<int>,"icon":"🙏", "color":"low"}},\n'
        '    {{"category":"Profanity/Abuse",     "words":["<word>"],"severity":"high",    "count":<int>,"icon":"⚠️","color":"high"}},\n'
        '    {{"category":"Legal Threats",       "words":["<word>"],"severity":"critical","count":<int>,"icon":"⚖️","color":"critical"}},\n'
        '    {{"category":"Escalation Request",  "words":["<word>"],"severity":"high",    "count":<int>,"icon":"📈","color":"high"}},\n'
        '    {{"category":"Compliance Risk",     "words":["<word>"],"severity":"medium",  "count":<int>,"icon":"🛡️","color":"medium"}},\n'
        '    {{"category":"Customer Frustration","words":["<word>"],"severity":"medium",  "count":<int>,"icon":"😤","color":"medium"}},\n'
        '    {{"category":"Cancellation Intent", "words":["<word>"],"severity":"high",    "count":<int>,"icon":"🚪","color":"high"}}\n'
        '  ],\n\n'
        '  "alerts": [\n'
        '    {{"type":"<string>","severity":"<low|medium|high|critical>","message":"<string>","icon":"<emoji>"}}\n'
        '  ],\n\n'
        '  "tags": {\n'
        '    "type": ["Support Call","Inquiry","Complaint","Feedback","Escalation","Conversation"],\n'
        '    "tone": ["Polite","Neutral","Professional","Friendly","Frustrated","Confused","Urgent","Aggressive"],\n'
        '    "pattern": "<Short Responses|Detailed Discussion|Issue Resolution|Information Exchange|Repeated Clarification>",\n'
        '    "frequency": "<Very High|High|Medium|Low>",\n'
        '    "focus_area": "<Account Support|Billing|Technical Support|General Inquiry|Password Reset|Customer Assistance>",\n'
        '    "emotional_signal": "<Satisfied|Neutral|Frustrated|Confused|Interested|Concerned>"\n'
        '  },\n\n'
        '  "overall_sentiment": "<Positive|Neutral|Negative>",\n'
        '  "call_direction":    "<Outbound|Inbound — Outbound if agent references previous call/discussion or already knows caller details; Inbound if customer called for new enquiry>",\n'
        '  "admission_intent":  "<Hot|Warm|Cold>",\n'
        '  "fee_sensitivity":   "<High|Medium|Low>",\n'
        '  "business_action":   "<specific recommended action string>",\n'
        '  "context":           "<2-4 sentence call summary>",\n'
        '  "rating":            <float 1.0-10.0>\n'
        '}}\n\n'
        'Rules:\n'
        '- overall_score = weighted average: sentiment 15%%, call_quality 35%%, agent_behavior 35%%, checklist 15%%\n'
        f'- CRITICAL: You MUST output "checklist_results" as a simple object with {param_count} numeric scores. Each value is 0 to its max points (partial scores allowed based on evidence strength). Do not miss any.\n'
        '- Checklist Evaluation Guidelines:\n'
        + eval_text + '\n'
        '- call_quality.score = average of factors\n'
        '- agent_behavior.score = weighted average of category_scores based on their weightage\n'
        '- For sensitive_words: list only words actually found in transcript; use empty array [] if none found\n'
        '- alerts: include one entry per sensitive_words category that has count > 0, plus extra entries for very negative sentiment or very low agent score\n'
        '- word_count and turn_count: count from the transcript text itself\n'
        '- call_direction: IMPORTANT — carefully determine Outbound vs Inbound:\n'
        '  OUTBOUND indicators (if ANY of these are true, mark Outbound):\n'
        '    * If the "क्या मेरी बात" , "घर से बात हो रही है", "के पेरेंट्स से बात हो रही है" is involved in transcription mark is OUTBOUND.'
        '    * Agent mentions a previous call or discussion (लास्ट टाइम, पिछली बार, as discussed, पहले बात हुई थी, डिस्कशन हुआ था)\n'
        '    * Agent already knows student name, parent name, or course interest WITHOUT customer telling them first\n'
        '    * Agent says "I am calling regarding..." or "आपसे बात करनी थी regarding..."\n'
        '    * Agent references an exam that was already given (एग्जाम दिलवाया था)\n'
        
        '  INBOUND indicators:\n'
        '    * If the "आपका नाम बता दीजिए" , "आपका स्वागत है", "मैं आपकी क्या सहायता कर सकता हूँ", "मैं आपकी क्या सहायता कर सकती हूँ" is involved in transcription mark is INBOUND.'
        '    * Customer asks a fresh question (fees kaise jama hoga, admission kaise hoga)\n'
        '    * Customer drives the topic, agent only responds\n'
        '    * No reference to any prior interaction\n\n'
        'CRITICAL REQUIREMENT — CHECKLIST ITEMS:\n'
        f'- You MUST return ALL {param_count} checklist items inside checklist.items array.\n'
        '- Each item MUST have: id, name, passed (true/false), earned (numeric 0 to weight), evidence (array of phrases from transcript), weight, parent_category.\n'
        '- Score each param from 0 to its max points. Full points = fully satisfied. Partial points = partially satisfied. 0 = no evidence.\n'
        '- Set passed=true if earned >= 50% of weight, else passed=false.\n'
        '- Evaluate EACH item INDEPENDENTLY by searching for evidence in the transcript.\n'
        '- Do NOT shortcut by returning only checklist.score — the items array is MANDATORY.\n'
        '- For each item, try to include a short phrase from the transcript as evidence. If no direct quote, evidence can be empty [].\n'
        '- tags must ALWAYS be filled based on transcript behavior and tone.\n'
        '- Never return empty tags object.\n'
        '- If conversation is normal, use tone=["Neutral"] and emotional_signal="Neutral".\n'
        '- tags object is MANDATORY. Never leave it empty.\n'
        '- tags must ALWAYS contain meaningful values based on the conversation.\n'
        '- Never return empty arrays in tags.\n'
        '- Detect actual customer interaction tone from transcript.\n'
        '- If call is normal customer support conversation use:\n'
        '  tone=["Professional","Polite"]\n'
        '  type=["Support Call"]\n'
        '  emotional_signal="Neutral"\n'
        '- Return ONLY the JSON object, nothing else.\n'
    )
    return prompt

# Keep a module-level fallback for backward compat (used if _build_prompt_template fails)
ANALYSIS_PROMPT_TEMPLATE = None
try:
    ANALYSIS_PROMPT_TEMPLATE = _build_prompt_template()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Stop-word set for local word-cloud computation
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
    "is","are","was","were","be","been","have","has","do","does","did","will",
    "would","could","should","this","that","these","those","i","you","he","she",
    "it","we","they","me","him","her","us","them","my","your","his","its","our",
    "their","what","which","who","when","where","why","how","all","each","some",
    "no","not","just","also","now","here","then","um","uh","okay","ok","yeah",
    "yes","well","right","going","get","got","go","know","think","see","want",
    "need","take","make","come","let","say","said","im","youre","dont","cant",
    "wont","thats","so","sir","ma","am","hm","na","hai","mein","aap","kya",
}

_POSITIVE_WORDS = {
    "thank","thanks","appreciate","great","good","excellent","wonderful","helpful",
    "perfect","amazing","fantastic","happy","pleased","satisfied","resolved",
    "solved","love","best","awesome","brilliant","outstanding","superb","delighted",
    "grateful","impressed","sure","yes",
}

_NEGATIVE_WORDS = {
    "not","no","never","bad","worst","terrible","horrible","awful","poor",
    "disappointing","frustrated","angry","upset","unhappy","dissatisfied",
    "problem","issue","wrong","fail","failed","complaint","hate","annoyed",
    "irritated","confused","difficult","complicated","broken","error",
}

# ---------------------------------------------------------------------------
# Local helpers — computed from raw transcript text, not from the model
# ---------------------------------------------------------------------------

def debug_transcript_format(transcript: str, job_id=None, status_obj=None):
    """Call this temporarily to diagnose speaker label issues on server."""
    lines = transcript.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    non_empty = [l.strip() for l in lines if l.strip()]
    log_step(job_id, f"[DEBUG] Total lines: {len(non_empty)}", status_obj)
    for i, line in enumerate(non_empty[:5]):  # Print first 5 lines
        log_step(job_id, f"[DEBUG] Line {i}: repr={repr(line[:80])}", status_obj)



def _compute_speaker_analysis(transcript: str) -> Dict:
    """Split lines by Agent/Customer prefix and compute word counts."""
    
    # ✅ FIX: Handle ALL newline variants including literal \n strings
    transcript = (
        transcript
        .replace("\\n", "\n")   # literal backslash-n → real newline  ← KEY FIX
        .replace("\r\n", "\n")  # Windows CRLF
        .replace("\r", "\n")    # old Mac CR
    )
    
    lines = [l.strip() for l in transcript.split("\n") if l.strip()]
    agent_words, customer_words = 0, 0
    agent_turns, customer_turns = 0, 0

    AGENT_PATTERN = re.compile(
        r"^(agent|representative|support|rep|advisor|consultant)\s*:",
        re.IGNORECASE
    )
    CUSTOMER_PATTERN = re.compile(
        r"^(customer|caller|client|user|member)\s*:",
        re.IGNORECASE
    )

    for line in lines:
        if AGENT_PATTERN.match(line):
            text = re.sub(r"^[^:]+:\s*", "", line)
            agent_words += len(text.split())
            agent_turns += 1
        elif CUSTOMER_PATTERN.match(line):
            text = re.sub(r"^[^:]+:\s*", "", line)
            customer_words += len(text.split())
            customer_turns += 1
        else:
            customer_words += len(line.split())
            customer_turns += 1

    total = agent_words + customer_words or 1
    ap = round(agent_words / total * 100)
    cp = 100 - ap
    return {
        "agent":    {"word_count": agent_words, "percentage": ap, "turn_count": agent_turns},
        "customer": {"word_count": customer_words, "percentage": cp, "turn_count": customer_turns},
        "talk_ratio": f"{ap}:{cp}",
    }


def _compute_word_cloud(transcript: str) -> List[Dict]:
    """Build a simple word-frequency cloud (top 40 meaningful words)."""
    # ✅ Same fix here too
    transcript = transcript.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    words = re.sub(r"[^a-z\s]", " ", transcript.lower()).split()
    freq: Counter = Counter(w for w in words if len(w) > 2 and w not in _STOP_WORDS)
    result = []
    for word, count in freq.most_common(40):
        if word in _POSITIVE_WORDS:
            category = "positive"
        elif word in _NEGATIVE_WORDS:
            category = "negative"
        else:
            category = "neutral"
        result.append({"word": word, "count": count, "category": category})
    return result


def _merge_local_fields(ollama_result: Dict, transcript: str) -> Dict:
    # ✅ Normalize here once, so downstream functions get clean text
    transcript = transcript.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    
    lines = [l for l in transcript.split("\n") if l.strip()]
    word_count = len(transcript.split())

    ollama_result["speaker_analysis"] = _compute_speaker_analysis(transcript)
    ollama_result["word_cloud"]       = _compute_word_cloud(transcript)

    if "summary" not in ollama_result:
        ollama_result["summary"] = {}
    ollama_result["summary"]["word_count"]  = word_count
    ollama_result["summary"]["turn_count"]  = len(lines)

    return ollama_result

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[str]:
    """Strip markdown fences and locate the outermost JSON object."""
    # Remove code fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    # Find first { … last }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


# ---------------------------------------------------------------------------
# Math Enforcer (Overrides LLM math hallucinations to guarantee perfect formulas)
# ---------------------------------------------------------------------------

def _enforce_math_scores(result: Dict, call_category=None, call_type=None) -> Dict:
    """
    Recalculates ALL scores deterministically from the LLM's checklist boolean flags.
    Covers: checklist items, call_quality factors, agent_behavior category_scores.
    No LLM-generated numbers are trusted for scoring — only passed/failed booleans.

    Now reads parameter definitions dynamically from the database.
    """
    from services.parameter_service import get_parameters, get_agent_mappings, get_auto_weightages

    categories = get_parameters(call_category, call_type)
    agent_mappings = get_agent_mappings(call_category, call_type)
    auto_weightages = get_auto_weightages(call_category, call_type)

    # ── Build max_weights from DB parameters ─────────────────────────────────
    max_weights = {}
    for cat in categories:
        for p in cat["parameters"]:
            max_weights[p["key"]] = {"name": p["name"], "weight": p["weight"]}

    # ── Build cat_map from DB categories (auto-normalized weightages) ────────
    cat_map = {}
    for cat in categories:
        param_keys = [p["key"] for p in cat["parameters"]]
        total_weight = sum(p["weight"] for p in cat["parameters"])
        cat_map[cat["label"]] = {
            "items": param_keys,
            "max": total_weight,
            "icon": cat["icon"],
            "weightage": auto_weightages.get(cat["key"], 0),
        }

    # ── Build agent_cat_map from DB agent mappings ───────────────────────────
    agent_cat_map = {}
    for m in agent_mappings:
        agent_cat_map[m["agent_category"]] = {
            "items": m["parameter_keys"],
            "weightage": m["weightage"],
        }

    # ── 1. Read scores from LLM output (numeric or boolean) ────────────────
    score_map = result.get("checklist_results", {})

    # Fallback: if LLM returned the old checklist.items array instead
    if not score_map and isinstance(result.get("checklist"), dict):
        for old_item in result.get("checklist", {}).get("items", []):
            if isinstance(old_item, dict) and "id" in old_item:
                score_map[old_item["id"]] = old_item.get("earned", old_item.get("earned_weight",
                    old_item["weight"] if old_item.get("passed") else 0))

    # ── 2. Build checklist.items with weight + earned_weight ─────────────────
    items        = []
    earned_total = 0
    max_total    = 0
    for iid, info in max_weights.items():
        raw_score = score_map.get(iid, 0)
        # Handle both boolean (legacy) and numeric scores
        if isinstance(raw_score, bool):
            earned_weight = info["weight"] if raw_score else 0
        else:
            earned_weight = min(int(round(float(raw_score))), info["weight"])
            earned_weight = max(earned_weight, 0)
        is_passed     = earned_weight >= (info["weight"] * 0.5)
        earned_total += earned_weight
        max_total    += info["weight"]
        items.append({
            "id":            iid,
            "name":          info["name"],
            "passed":        is_passed,
            "evidence":      [],
            "weight":        info["weight"],
            "earned_weight": earned_weight,
        })

    # checklist.score = earned / actual total (97) * 100
    checklist_score = round((earned_total / max_total) * 100) if max_total else 0
    grade = (
        "A" if checklist_score >= 90 else
        "B" if checklist_score >= 75 else
        "C" if checklist_score >= 60 else
        "D" if checklist_score >= 40 else "F"
    )
    result["checklist"] = {"score": checklist_score, "grade": grade, "items": items}

    # ── 3. Rebuild call_quality.factors ──────────────────────────────────────
    call_quality = result.get("call_quality", {})
    details_map  = {
        f.get("name"): f.get("details", "")
        for f in call_quality.get("factors", [])
        if isinstance(f, dict) and f.get("name")
    }

    # Build earned lookup from items
    earned_lookup = {item["id"]: item["earned_weight"] for item in items}

    new_factors = []
    for cat_name, cat_info in cat_map.items():
        earned = sum(
            earned_lookup.get(iid, 0)
            for iid in cat_info["items"]
        )
        cat_score = round((earned / cat_info["max"]) * 100) if cat_info["max"] > 0 else 0
        status    = "pass" if cat_score >= 75 else ("warning" if cat_score >= 50 else "fail")
        new_factors.append({
            "name":     cat_name,
            "score":    cat_score,
            "details":  details_map.get(cat_name, "Evaluated based on checklist parameters."),
            "icon":     cat_info["icon"],
            "status":   status,
            "weightage": cat_info["weightage"],
        })

    # call_quality.score = weighted average of factor scores (total weightage = 100)
    total_w  = sum(f["weightage"] for f in new_factors)
    cq_score = round(sum(f["score"] * f["weightage"] for f in new_factors) / total_w) if total_w else 0
    call_quality["factors"] = new_factors
    call_quality["score"]   = cq_score
    call_quality["rating"]  = (
        "Excellent" if cq_score >= 90 else
        "Good"      if cq_score >= 75 else
        "Fair"      if cq_score >= 60 else "Poor"
    )
    result["call_quality"] = call_quality

    # ── 4. Rebuild agent_behavior.category_scores from checklist scores ──────
    agent_behavior = result.get("agent_behavior", {})
    new_cat_scores = []
    for cat_name, cat_info in agent_cat_map.items():
        cat_items   = cat_info["items"]
        cat_max     = sum(max_weights[iid]["weight"] for iid in cat_items)
        cat_earned  = sum(
            earned_lookup.get(iid, 0)
            for iid in cat_items
        )
        # Score 0-100: proportion of sub-params passed, scaled to 100
        cat_score = round((cat_earned / cat_max) * 100) if cat_max > 0 else 0
        new_cat_scores.append({
            "category": cat_name,
            "score":    cat_score,
            "weightage": cat_info["weightage"],
        })

    agent_behavior["category_scores"] = new_cat_scores

    # agent_behavior.score = weighted average of category scores
    ab_total_w   = sum(c["weightage"] for c in new_cat_scores)
    ab_score     = round(sum(c["score"] * c["weightage"] for c in new_cat_scores) / ab_total_w) if ab_total_w else 0
    ab_rating    = (
        "Exceptional"    if ab_score >= 90 else
        "Proficient"     if ab_score >= 75 else
        "Developing"     if ab_score >= 50 else
        "Needs Training"
    )
    agent_behavior["score"]  = ab_score
    agent_behavior["rating"] = ab_rating
    result["agent_behavior"] = agent_behavior

    # ── 5. Recompute summary.overall_score ───────────────────────────────────
    # Weights: sentiment 15%, call_quality 35%, agent_behavior 35%, checklist 15%
    sent_raw  = result.get("sentiment", {}).get("score", 0)
    sent_norm = round((sent_raw + 100) / 2)  # map -100..100 → 0..100

    overall_score = round(
        sent_norm       * 0.15 +
        cq_score        * 0.35 +
        ab_score        * 0.35 +
        checklist_score * 0.15
    )
    overall_grade = (
        "A" if overall_score >= 90 else
        "B" if overall_score >= 75 else
        "C" if overall_score >= 60 else
        "D" if overall_score >= 50 else "F"
    )
    result.setdefault("summary", {})["overall_score"] = overall_score
    result["summary"]["overall_grade"] = overall_grade

    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_step(job_id, message, status_obj=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [JOB:{job_id}] {message}"
    print(log_message)
    if status_obj and job_id in status_obj:
        status_obj[job_id].setdefault("logs", []).append(log_message)


# ---------------------------------------------------------------------------
# Ollama HTTP call
# ---------------------------------------------------------------------------

def run_ollama_analysis(
    transcript_text: str,
    job_id=None,
    status_obj=None,
    retry_count: int = 0,
) -> Optional[Dict]:
    """Send transcript to Ollama and return parsed analytics dict."""
    try:
        log_step(job_id, f"Starting Ollama analysis with {OLLAMA_CONFIG['model_name']}…", status_obj)

        if not transcript_text or not transcript_text.strip():
            log_step(job_id, "ERROR: Empty transcript provided", status_obj)
            return None

        # --- FIX: TRUNCATE INPUT IF EXTREMELY LARGE ---
        # If the transcript is massive, Gemma/Llama may lose the plot. 
        # We take the first 3000 words which usually covers the meat of the call.
        words = transcript_text.split()
        if len(words) > 3000:
            log_step(job_id, f"Trimming transcript from {len(words)} to 3000 words for stability.", status_obj)
            transcript_text = " ".join(words[:3000])

        # Build prompt dynamically from DB parameters
        try:
            prompt_template = _build_prompt_template()
        except Exception as pt_err:
            log_step(job_id, f"WARNING: Dynamic prompt build failed ({pt_err}), using cached fallback", status_obj)
            prompt_template = ANALYSIS_PROMPT_TEMPLATE
        prompt = prompt_template.format(transcript=transcript_text)

        payload = json.dumps({
            "model":  OLLAMA_CONFIG["model_name"],
            "prompt": prompt,
            "stream": False,
            "think":  False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 8000,
                "num_ctx": 8192
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_CONFIG["api_url"],
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_CONFIG["timeout_seconds"]) as resp:
            raw = resp.read().decode("utf-8")

        http_resp    = json.loads(raw)
        model_output = http_resp.get("response", "").strip()

        if not model_output:
            log_step(job_id, f"ERROR: Ollama returned empty response. done_reason={http_resp.get('done_reason')}", status_obj)
            return None

        log_step(job_id, f"Ollama response: {len(model_output)} chars, done_reason={http_resp.get('done_reason')}", status_obj)

        # If truncated (done_reason=length), try to fix the JSON
        if http_resp.get('done_reason') == 'length':
            log_step(job_id, "⚠️  Response truncated (done_reason=length) — attempting JSON repair", status_obj)
            # Try to close any open braces/brackets
            depth_brace = model_output.count('{') - model_output.count('}')
            depth_bracket = model_output.count('[') - model_output.count(']')
            model_output = model_output.rstrip().rstrip(',')
            model_output += ']' * max(0, depth_bracket) + '}' * max(0, depth_brace)

        # --- EXTRACTION ---
        json_text = _extract_json(model_output)
        if not json_text:
            log_step(job_id, f"ERROR: Could not extract JSON from response. First 300 chars: {model_output[:300]}", status_obj)
            # Try parsing the raw output directly
            json_text = model_output

        try:
            parsed = json.loads(json_text)
            log_step(job_id, f"✅ JSON parsed ({len(parsed)} keys)", status_obj)
            return parsed
        except json.JSONDecodeError as e:
            log_step(job_id, f"ERROR: JSON parse failed: {e}. Last 200 chars: ...{json_text[-200:]}", status_obj)
            if retry_count < OLLAMA_CONFIG["retry_attempts"]:
                log_step(job_id, f"Retrying ({retry_count + 1}/{OLLAMA_CONFIG['retry_attempts']})…", status_obj)
                return run_ollama_analysis(transcript_text, job_id, status_obj, retry_count + 1)
            return None

    except urllib.error.URLError as e:
        log_step(job_id, f"ERROR: Cannot reach Ollama — {e}", status_obj)
        return None
    except TimeoutError:
        log_step(job_id, f"ERROR: Ollama timed out after {OLLAMA_CONFIG['timeout_seconds']}s", status_obj)
        if retry_count < OLLAMA_CONFIG["retry_attempts"]:
            log_step(job_id, f"Retrying ({retry_count + 1}/{OLLAMA_CONFIG['retry_attempts']})…", status_obj)
            import time as _time
            _time.sleep(5)
            return run_ollama_analysis(transcript_text, job_id, status_obj, retry_count + 1)
        return None
    except Exception as e:
        log_step(job_id, f"ERROR in run_ollama_analysis: {type(e).__name__}: {e}", status_obj)
        import traceback
        log_step(job_id, traceback.format_exc(), status_obj)
        return None


# ---------------------------------------------------------------------------
# Fallback result (all fields present so frontend never crashes)
# ---------------------------------------------------------------------------

def _build_fallback(transcript: str) -> Dict:
    lines = [l for l in transcript.split("\n") if l.strip()]
    wc    = len(transcript.split())
    return {
        "summary": {
            "overall_score": 0, "overall_grade": "F",
            "word_count": wc,   "turn_count": len(lines),
        },
        "sentiment": {
            "score": 0, "label": "Neutral", "emoji": "😐",
            "positive_words": [], "negative_words": [],
            "progression": [0, 0, 0, 0, 0],
        },
        "call_quality": {
            "score": 0, "rating": "Poor",
            "factors": [
                {"name": "Call Opening & Customer Identification", "score": 0, "details": "Manual review required", "icon": "🚪", "status": "fail", "weightage": 10},
                {"name": "Issue / VOC Identification", "score": 0, "details": "Manual review required", "icon": "🔍", "status": "fail", "weightage": 15},
                {"name": "FCR Achieved", "score": 0, "details": "Manual review required", "icon": "✅", "status": "fail", "weightage": 20},
                {"name": "Communication Skills", "score": 0, "details": "Manual review required", "icon": "💬", "status": "fail", "weightage": 20},
                {"name": "Call Handling Efficiency", "score": 0, "details": "Manual review required", "icon": "⏱️", "status": "fail", "weightage": 15},
                {"name": "Resolution Compliance", "score": 0, "details": "Manual review required", "icon": "🛡️", "status": "fail", "weightage": 20},
            ],
        },
        "agent_behavior": {
            "score": 0, "rating": "Needs Training",
            "strengths": [], "improvements": [],
            "category_scores": [
                {"category": "Greeting", "score": 0, "weightage": 10},
                {"category": "Empathy", "score": 0, "weightage": 15},
                {"category": "Proactive", "score": 0, "weightage": 10},
                {"category": "Closing", "score": 0, "weightage": 10},
                {"category": "Attitude", "score": 0, "weightage": 10},
                {"category": "Urgency", "score": 0, "weightage": 10},
                {"category": "Knowledge", "score": 0, "weightage": 15},
                {"category": "Accountability", "score": 0, "weightage": 10},
                {"category": "Communication", "score": 0, "weightage": 10}
            ],
            "detailed_metrics": [
                {"name": n, "score": 0, "max": 100} for n in
                ["Empathy","Professional Language","Problem Resolution","Call Control","Customer Focus"]
            ],
        },
        "checklist_results": {},
        "sensitive_words": [
            {"category": "Religious Content",    "words": [], "severity": "low",      "count": 0, "icon": "🙏",  "color": "low"},
            {"category": "Profanity/Abuse",      "words": [], "severity": "high",     "count": 0, "icon": "⚠️", "color": "high"},
            {"category": "Legal Threats",        "words": [], "severity": "critical", "count": 0, "icon": "⚖️", "color": "critical"},
            {"category": "Escalation Request",   "words": [], "severity": "high",     "count": 0, "icon": "📈", "color": "high"},
            {"category": "Compliance Risk",      "words": [], "severity": "medium",   "count": 0, "icon": "🛡️", "color": "medium"},
            {"category": "Customer Frustration", "words": [], "severity": "medium",   "count": 0, "icon": "😤", "color": "medium"},
            {"category": "Cancellation Intent",  "words": [], "severity": "high",     "count": 0, "icon": "🚪", "color": "high"},
        ],
        "tags": {},
        "alerts": [],
        "speaker_analysis": _compute_speaker_analysis(transcript),
        "word_cloud":       _compute_word_cloud(transcript),
        "overall_sentiment": "Unknown",
        "admission_intent":  "Unknown",
        "fee_sensitivity":   "Unknown",
        "business_action":   "Manual review required",
        "context":           "Model output could not be parsed. Manual review required.",
        "rating":            0.0,
    }


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_transcript_file(file_path: str, job_id=None, status_obj=None) -> Optional[str]:
    try:
        if not os.path.exists(file_path):
            log_step(job_id, f"ERROR: File not found: {file_path}", status_obj)
            return None
        log_step(job_id, f"Reading transcript: {file_path}", status_obj)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            log_step(job_id, "WARNING: Transcript file is empty", status_obj)
            return None
        log_step(job_id, f"Transcript loaded: {len(text)} chars", status_obj)
        return text
    except Exception as e:
        log_step(job_id, f"Error reading file: {e}\n{traceback.format_exc()}", status_obj)
        return None


def save_analysis_result(result: Dict, output_path: str, job_id=None, status_obj=None) -> bool:
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log_step(job_id, f"✅ Result saved to: {output_path}", status_obj)
        return True
    except Exception as e:
        log_step(job_id, f"Error saving result: {e}\n{traceback.format_exc()}", status_obj)
        return False


# ---------------------------------------------------------------------------
# Main entry point (called by audio_processor.py STEP 4)
# ---------------------------------------------------------------------------

def _chunk_transcript(transcript: str, chunk_size: int = 2500) -> List[str]:
    """Break long transcript into manageable chunks for the LLM."""
    words = transcript.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def analyze_transcription(transcript_text: str, job_id=None, status_obj=None) -> Optional[Dict]:
    """
    Main entry point. Merges LLM output with reliable local data.
    """
    if not transcript_text:
        return None

    # 1. Run the LLM Analysis
    result = run_ollama_analysis(transcript_text, job_id, status_obj)

    # 2. Get the Fallback/Template Schema
    fallback = _build_fallback(transcript_text)

    # 3. FIX: GUARANTEE SCHEMA INTEGRITY
    # If the LLM failed or dropped keys, we merge what we have into the fallback template.
    if not result:
        log_step(job_id, "Using full fallback schema due to LLM failure.", status_obj)
        result = fallback
    else:
        # Check for specific missing top-level keys reported in your error
        critical_keys = ["summary", "sentiment", "call_quality", "agent_behavior", "checklist_results", "tags"]
        for key in critical_keys:
            if key not in result or not result[key]:
                log_step(job_id, f"⚠️  LLM dropped '{key}' — injecting fallback schema", status_obj)
                result[key] = fallback[key]

    # 4. ALWAYS OVERWRITE WITH ACCURATE LOCAL DATA
    # Don't trust LLM for counts/ratios on large files.
    result = _merge_local_fields(result, transcript_text)

    # ---------------------------------------------------
    # GUARANTEE NON-EMPTY TAGS
    # ---------------------------------------------------
    tags = result.get("tags", {})

    if not tags.get("type") or len(tags.get("type", [])) == 0:

        transcript_lower = transcript_text.lower()

        if any(word in transcript_lower for word in [
            "password", "account", "reset", "login"
        ]):
            tags["type"] = ["Support Call"]

        elif any(word in transcript_lower for word in [
            "problem", "issue", "complaint"
        ]):
            tags["type"] = ["Complaint"]

        else:
            tags["type"] = ["Conversation"]

    if not tags.get("tone") or len(tags.get("tone", [])) == 0:

        if any(word in transcript_text.lower() for word in [
            "thanks", "thank you", "sure"
        ]):
            tags["tone"] = ["Polite"]

        else:
            tags["tone"] = ["Neutral"]

    if not tags.get("pattern"):
        tags["pattern"] = "Information Exchange"

    if not tags.get("frequency"):
        tags["frequency"] = "Low"

    if not tags.get("focus_area"):

        if "password" in transcript_text.lower():
            tags["focus_area"] = "Password Reset"

        else:
            tags["focus_area"] = "General Inquiry"

    if not tags.get("emotional_signal"):
        tags["emotional_signal"] = "Neutral"

    result["tags"] = tags

    # 5. RECALCULATE MATH
    # Forces scores to match the boolean flags
    result = _enforce_math_scores(result, call_category=None, call_type=None)

    return result


# ---------------------------------------------------------------------------
# Complete pipeline (Read → Analyze → Save)
# ---------------------------------------------------------------------------

def process_transcription_and_analysis(
    transcript_file_path: str,
    output_file: str = "analysis_result.json",
    job_id: str = None,
    status_obj: Dict = None,
    check_for_stop: Callable = None,
) -> Optional[Dict]:
    """Read transcript file → analyze with Ollama → save full analytics JSON."""
    if check_for_stop is None:
        check_for_stop = lambda: False
    if status_obj is None:
        status_obj = {}

    try:
        if job_id not in status_obj:
            status_obj[job_id] = {"status": "processing", "steps": {}, "logs": []}

        log_step(job_id, "=" * 60, status_obj)
        log_step(job_id, "STARTING FULL CALL ANALYTICS PIPELINE", status_obj)
        log_step(job_id, "=" * 60, status_obj)

        if check_for_stop():
            return None

        transcript_text = read_transcript_file(transcript_file_path, job_id, status_obj)
        if not transcript_text:
            raise RuntimeError("Failed to read transcript file")

        if check_for_stop():
            return None

        result = analyze_transcription(transcript_text, job_id, status_obj)
        if not result:
            raise RuntimeError("Analysis failed")

        if check_for_stop():
            return None

        if not save_analysis_result(result, output_file, job_id, status_obj):
            raise RuntimeError("Failed to save result")

        status_obj[job_id]["status"] = "completed"
        status_obj[job_id]["result"] = result

        log_step(job_id, "✅ PIPELINE COMPLETED SUCCESSFULLY", status_obj)
        return result

    except Exception as e:
        log_step(job_id, f"Pipeline error: {e}\n{traceback.format_exc()}", status_obj)
        status_obj[job_id]["status"] = "failed"
        status_obj[job_id]["error"]  = str(e)
        return None


# ---------------------------------------------------------------------------
# Example / debug
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    status = {}
    job_id = "demo_job_1"
    result = process_transcription_and_analysis(
        transcript_file_path="demo1.txt",
        output_file="demo_analysis_result.json",
        job_id=job_id,
        status_obj=status,
    )
    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False))