"""
Smart-answer helpers (pure functions, no network, standard library only).

These make the bot answer well no matter which model is behind it:
  * detect_language  - Hindi / Hinglish / English, so replies match the user
  * classify         - what kind of question is this, and how hard is it?
  * clean_output     - strip <think> blocks and other model junk
  * quality_check    - spot genuinely bad answers (empty, looping, wrong
                       language, leaked prompt, ...) so a stronger model can
                       be tried
  * merge_continuation - stitch a cut-off answer to its continuation

Note: the quality gate only looks at *quality*. It never retries because a
model declined a request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.ai.models import TaskType

# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------
_HINGLISH = frozenset(
    """
    kya hai hain nahi nhi kaise kese kyun kyu mujhe mera meri mere tum tumhara
    tumhe aap aapka bhai yaar karo kro krna karna chahiye batao bata banao
    dikhao samjhao samjha likho likh hoga hogi raha rahi rha rhi toh aur lekin
    mein ko ka ki ke ek koi kuch sab abhi phir fir bahut bohot thoda zyada kaam
    kaun kab kahan kitna kitni hoon hu hun tha thi wala wali accha achha theek
    thik bolo chalo dekho suno hum humare unko usko isko isme usme jaise vaise
    kyunki isliye matlab samajh pata lagta lagega milega banega hota hoti hote
    rakhna rakho daalo kare kiya kiye karenge karunga sakta sakte sakti
    """.split()
)

LANGUAGE_HINTS = {
    "hindi": (
        "The user writes in Hindi (Devanagari script). Reply in Hindi "
        "(Devanagari) unless they ask for another language."
    ),
    "hinglish": (
        "The user writes Hinglish (Hindi in Roman letters mixed with English). "
        "Reply in natural Hinglish the same way: Roman letters, never Devanagari."
    ),
    "english": "Reply in English.",
    "other": "Reply in the same language the user wrote in.",
}


def _script_counts(text: str) -> tuple[int, int, int]:
    """(latin, devanagari, other) letter counts."""
    latin = deva = other = 0
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250:
            latin += 1
        elif 0x900 <= o <= 0x97F:
            deva += 1
        else:
            other += 1
    return latin, deva, other


def detect_language(text: str) -> str:
    """Return 'hindi' | 'hinglish' | 'english' | 'other'."""
    latin, deva, other = _script_counts(text or "")
    total = latin + deva + other
    if total == 0:
        return "english"
    if deva / total > 0.3:
        return "hindi"
    if other / total > 0.3:
        return "other"
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if len(words) >= 2:
        hits = sum(1 for w in words if w in _HINGLISH)
        if hits >= 2 or hits / len(words) >= 0.25:
            return "hinglish"
    return "english"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Classification:
    task: TaskType
    difficulty: str  # "easy" | "normal" | "hard"
    language: str


_GREETING = re.compile(
    r"^\s*(hi+|hello+|hey+|hii+|namaste|namaskar|good\s+(morning|evening|"
    r"night|afternoon)|thanks?|thank you|thx|ok(ay)?|okk+|hmm+|bye|gm|gn|"
    r"kaise ho|kya haal|how are you|sup|yo|shukriya|dhanyawad)\W*$",
    re.I,
)

# (regex, weight). A task wins if its score reaches the threshold.
_CODE_WEIGHTS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"```"), 4),
    (re.compile(
        r"\b(python|javascript|typescript|java|kotlin|flutter|react|node\.?js|"
        r"html|css|sql|mysql|postgres|sqlite|json|regex|git|github|docker|"
        r"traceback|stack ?trace|exception|compile[rd]?|syntax|debug|"
        r"algorithm|coding|programming|pip install|npm|bug)\b", re.I), 2),
    (re.compile(r"\b(code|program|api|function|error)\b", re.I), 1),
    (re.compile(r"(\bdef \w+\(|\bimport \w+|console\.log|print\(|=>|;\s*$)", re.M), 2),
]
_TASK_PATTERNS: dict[TaskType, tuple[list[tuple[re.Pattern, int]], int]] = {
    TaskType.CODE: (_CODE_WEIGHTS, 2),
    TaskType.FREELANCE: ([
        (re.compile(r"\b(freelanc\w*|fiverr|upwork)\b", re.I), 3),
        (re.compile(r"\b(proposal|client|gig|invoice)\b", re.I), 2),
    ], 2),
    TaskType.CONTENT: ([
        (re.compile(r"\b(youtube|reels?|shorts|instagram|tiktok|caption|"
                    r"hashtags?|thumbnail|newsletter|blog)\b", re.I), 2),
        (re.compile(r"\b(video|content) (idea|script)s?\b|\bscript for\b", re.I), 2),
    ], 2),
    TaskType.SUMMARY: ([
        (re.compile(r"\b(summari[sz]e|summary|tl;?dr|key points|saar|"
                    r"sankshep)\b|short mein|in short", re.I), 3),
    ], 3),
    TaskType.REASONING: ([
        (re.compile(r"\d\s*[-+*/^×÷x]\s*\d"), 3),
        (re.compile(r"\b(calculate|solve|prove|derive|equation|integral|"
                    r"derivative|probability|puzzle|riddle|logic|percent(age)?|"
                    r"step by step|pros and cons|compare|analy[sz]e|strategy|"
                    r"reasoning)\b", re.I), 2),
        (re.compile(r"\bwhy\b", re.I), 1),
    ], 2),
}
_TASK_ORDER = [
    TaskType.CODE, TaskType.FREELANCE, TaskType.CONTENT,
    TaskType.SUMMARY, TaskType.REASONING,
]
_HARD_HINT = re.compile(
    r"\b(architecture|design a (system|database|schema)|step by step|prove|"
    r"optimi[sz]e|refactor|migrate|business plan|strategy|traceback|"
    r"explain in detail|in depth|roadmap)\b",
    re.I,
)


def _score(task: TaskType, text: str) -> int:
    patterns, _thr = _TASK_PATTERNS[task]
    return sum(w for rx, w in patterns if rx.search(text))


def classify(text: str, *, has_image: bool = False) -> Classification:
    text = text or ""
    lang = detect_language(text)
    if has_image:
        return Classification(TaskType.VISION, "normal", lang)

    stripped = text.strip()
    best, best_score = TaskType.CHAT, 0
    for task in _TASK_ORDER:
        s = _score(task, text)
        if s >= _TASK_PATTERNS[task][1] and s > best_score:
            best, best_score = task, s

    easy = bool(_GREETING.match(stripped)) or (
        len(stripped) <= 12 and "?" not in stripped
        and not any(c.isdigit() for c in stripped)
    )
    hard = (
        len(stripped) > 700
        or stripped.count("?") >= 3
        or len(re.findall(r"^\s*(\d+[.)]|[-*•])\s", stripped, re.M)) >= 3
        or bool(_HARD_HINT.search(stripped))
        or (best == TaskType.CODE and best_score >= 6 and len(stripped) > 300)
        or (best == TaskType.REASONING and best_score >= 5)
    )
    difficulty = "easy" if easy and not hard else ("hard" if hard else "normal")
    return Classification(best, difficulty, lang)


PREFER_BY_DIFFICULTY = {"easy": "speed", "normal": "balanced", "hard": "quality"}


# --------------------------------------------------------------------------
# Output cleaning
# --------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_output(text: str) -> str:
    """Remove <think> blocks, control chars and excess blank lines."""
    if not text:
        return ""
    t = _THINK_BLOCK.sub("", text)
    m = re.search(r"<(think|thinking)>", t, re.I)
    if m:  # opened but never closed
        before = t[: m.start()].strip()
        t = before if before else re.sub(r"</?(think|thinking)>", "", t, flags=re.I)
    t = re.sub(r"</?(think|thinking)>", "", t, flags=re.I)
    t = _CONTROL.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def has_unclosed_fence(text: str) -> bool:
    fences = [ln for ln in (text or "").splitlines() if ln.strip().startswith("```")]
    return len(fences) % 2 == 1


def merge_continuation(prev: str, extra: str) -> str:
    """Join a cut-off answer with its continuation."""
    prev = prev.rstrip()
    extra = (extra or "").strip()
    if not extra:
        return prev
    # The model often repeats the end of the old text: drop that overlap.
    for n in range(min(len(prev), len(extra), 120), 14, -1):
        if prev.endswith(extra[:n]):
            extra = extra[n:].lstrip()
            break
    if not extra:
        return prev
    if has_unclosed_fence(prev):
        return prev + "\n" + extra
    if prev and prev[-1].isalnum() and extra[0].isalnum():
        return prev + " " + extra
    return prev + extra


# --------------------------------------------------------------------------
# Quality gate
# --------------------------------------------------------------------------
@dataclass
class GateResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


_PROMPT_LEAK = re.compile(
    r"you are \"universal ai\"|## core identity|## answer policy \(internal\)",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _is_looping(answer: str, task: TaskType) -> bool:
    min_len, min_reps = (25, 5) if task == TaskType.CODE else (15, 4)
    counts: dict[str, int] = {}
    for ln in answer.splitlines():
        ln = ln.strip()
        if len(ln) >= min_len:
            counts[ln] = counts.get(ln, 0) + 1
    if any(c >= min_reps for c in counts.values()):
        return True
    words = re.findall(r"\w+", answer.lower())
    if len(words) >= 80:
        grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
        if len(set(grams)) / len(grams) < 0.3:
            return True
    return False


def quality_check(
    user_text: str,
    answer: str,
    *,
    task: TaskType = TaskType.CHAT,
    language: str = "english",
) -> GateResult:
    issues: list[str] = []
    ans = (answer or "").strip()
    if not ans:
        return GateResult(False, ["empty"])

    if len(user_text.strip()) >= 80 and len(ans) < 20:
        issues.append("too_short")
    if len(ans) >= 20 and _norm(ans) == _norm(user_text):
        issues.append("echo")
    if _PROMPT_LEAK.search(ans):
        issues.append("prompt_leak")
    if len(ans) >= 20 and ans.count("�") / len(ans) > 0.02:
        issues.append("garbled")
    if _is_looping(ans, task):
        issues.append("repetition")

    latin, deva, other = _script_counts(ans)
    letters = latin + deva + other
    if letters >= 30:
        if language == "hindi" and deva / letters < 0.03:
            issues.append("wrong_language")
        elif language in ("english", "hinglish") and (deva + other) / letters > 0.4:
            issues.append("wrong_language")
    return GateResult(not issues, issues)
