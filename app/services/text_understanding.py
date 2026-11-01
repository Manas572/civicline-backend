"""Small, transparent category/urgency pass for English, Hindi, Hinglish and Marathi."""
import re

CATEGORY_TERMS = {
    "pothole": ("pothole", "road hole", "gadda", "gadde", "गड्ढा", "गड्ढे", "खड्डा", "खड्डे"),
    "water_leak": ("water leak", "leaking water", "pipe burst", "burst pipe", "water flowing", "pani leak", "पानी रिस", "पानी बह", "पाणी गळ", "गळती", "नल से पानी"),
    "garbage": ("garbage", "trash", "rubbish", "waste pile", "kachra", "कचरा", "घाण"),
    "streetlight": ("streetlight", "street light", "street lamp", "light pole", "not lighting", "बत्ती", "स्ट्रीट लाइट", "दिवा बंद", "दिवा लागत नाही"),
    "bridge_crack": ("bridge crack", "cracked bridge", "bridge damage", "bridge collapse", "pul crack", "पुल में दरार", "पुल टूट", "पुलाला तडा", "पूलाला तडा", "पूल तुट"),
}
CRITICAL_TERMS = (
    "live wire", "exposed wire", "electric shock", "sparking", "spark", "gas leak", "gas smell",
    "gas hiss", "fire", "collapse", "collapsed", "injured", "accident", "fatal",
    "करंट", "खुला तार", "तार में आग", "चिंगारी", "गैस रिस", "गैस की गंध", "गैसचा वास",
    "अपघात", "दुर्घटना", "ढासळ", "कोसळ", "जखमी", "खतरनाक", "ख़तरनाक", "तुरंत मदद",
)
HIGH_TERMS = (
    "urgent", "emergency", "dangerous", "deep", "large", "flooding", "flooded", "school", "hospital",
    "highway", "traffic", "blocked", "no light", "तत्काल", "तुरंत", "खतरा", "गंभीर", "बड़ा गड्ढा",
    "शाळा", "शाळेजवळ", "रुग्णालय", "तातडीचे", "धोकादायक", "मोठा खड्डा", "शाळा जवळ",
)
MODERATE_TERMS = ("please fix", "repair", "broken", "leak", "परेशानी", "समस्या", "दुरुस्ती", "बंद आहे", "गळत आहे")


def _matching(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def analyze_report_text(text: str | None) -> dict:
    normalized = (text or "").lower().strip()
    devanagari = bool(re.search(r"[\u0900-\u097f]", normalized))
    if devanagari:
        marathi_markers = ("आहे", "नाही", "पाणी", "खड्डा", "शाळा", "रुग्णालय", "दिवा", "जवळ")
        language = "Marathi" if any(term in normalized for term in marathi_markers) else "Hindi"
    elif any(term in normalized for term in ("gadda", "pani", "kachra", "bijli", "khadda", "pul", "safai")):
        language = "Hinglish"
    elif normalized:
        language = "English"
    else:
        language = "unknown"

    categories = []
    for issue, terms in CATEGORY_TERMS.items():
        matches = _matching(normalized, terms)
        if matches:
            categories.append({"issue_type": issue, "matched_terms": matches})
    categories.sort(key=lambda item: (
        -len(item["matched_terms"]),
        min((normalized.find(term) for term in item["matched_terms"] if normalized.find(term) >= 0), default=10**6),
    ))
    critical = _matching(normalized, CRITICAL_TERMS)
    high = _matching(normalized, HIGH_TERMS)
    moderate = _matching(normalized, MODERATE_TERMS)
    urgency, urgency_score = ("critical", 100) if critical else ("high", 75) if high else (
        ("moderate", 45) if moderate else ("routine", 15))
    return {
        "language": language,
        "issue_type": categories[0]["issue_type"] if len(categories) == 1 else None,
        "categories": categories,
        "urgency": urgency,
        "urgency_score": urgency_score,
        "urgency_terms": list(dict.fromkeys(critical + high + moderate)),
    }
