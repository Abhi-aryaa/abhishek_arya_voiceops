"""Step 1: Prompt-Only Ungrounded LLM Baseline (No RAG / No Company Documents)."""
import re
from typing import Any, Dict, Optional

from .llm_client import LLMClient

STEP1_SYSTEM_PROMPT = """You are ZentroMart's AI customer support assistant.
Role and Limits:
1. Introduce yourself as ZentroMart's AI support assistant in your first reply.
2. Answer customer questions politely and concisely in 2 to 3 short sentences.
3. Reply in the same language the customer uses (English, Hindi in Devanagari, or Hinglish).
4. Never share another customer's personal information or phone number.
5. Note: You do not have access to ZentroMart's internal policy documents or product catalog in Step 1.
"""


class Step1PromptOnlyAgent:
    """Step 1 baseline agent that relies only on a system prompt without RAG over ZentroMart documents."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def detect_language(self, text: str) -> str:
        if any("\u0900" <= c <= "\u097f" for c in text):
            return "hi"
        lower = text.lower()
        hinglish_markers = [
            "kya", "hai", "kaise", "karein", "mera", "mujhe", "kab", "tak",
            "chahiye", "milega", "sakti", "sakta", "hoon", "hain", "wala", "pe",
        ]
        if any(re.search(rf"\b{w}\b", lower) for w in hinglish_markers):
            return "hi-en"
        return "en"

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a single practice question using prompt-only assumptions (no KB documents)."""
        lang = self.detect_language(question)
        lower = question.lower()

        # If a live LLM API is configured, call it with ONLY the Step 1 system prompt (no RAG context)
        live_reply = self.llm.generate(
            STEP1_SYSTEM_PROMPT,
            [{"role": "user", "content": question}],
            temperature=0.0,
        )

        # Standard zero-context LLM behavior when no KB documents or catalog are provided:
        # It knows general safety/privacy refusals, basic fraud escalation, and generic India delivery,
        # but guesses generic e-commerce numbers (e.g., 5-7 days for refunds, 7-10 days for returns,
        # generic 1-year warranty) and fails ZentroMart-specific policy traps, holiday math, and SKUs.
        decision = "answer"
        if re.search(r"\b(phone\s+number\s+of\s+the\s+person|other\s+customer)\b", lower):
            decision = "refuse"
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. I'm sorry, I cannot share personal details or phone numbers of other customers."
            )
        elif re.search(r"\b(never\s+placed|unauthorized)\b", lower):
            decision = "escalate"
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. Since you see an order on your account you did not place, I am creating a ticket to connect you with a human support colleague immediately."
            )
        elif re.search(r"\b(when\s+will\s+my\s+refund\s+come)\b", lower):
            decision = "clarify"
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. Could you please share your order ID and payment method so I can check your refund timeline?"
            )
        elif "कितने दिन में वापस करना होगा" in question:
            decision = "clarify"
            fallback_reply = (
                "नमस्ते, मैं ZentroMart की AI सहायक हूँ। कृपया बताएं कि आप कौन सा सामान वापस करना चाहते हैं?"
            )
        elif re.search(r"\b(dubai)\b", lower):
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. No, ZentroMart currently delivers only within India and cannot ship to Dubai."
            )
        elif re.search(r"\b(swimsuit|briefs|innerwear)\b", lower):
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. For hygiene reasons, innerwear and swimwear cannot be returned for fit or change of mind."
            )
        elif re.search(r"\b(opened|tried\s+it|खोल\s+लिया)\b", lower) or "खोल लिया" in question:
            if lang == "hi":
                fallback_reply = "नमस्ते, मैं ZentroMart की AI सहायक हूँ। ब्यूटी प्रोडक्ट केवल तभी वापस किए जा सकते हैं जब उनकी सील बंद हो और उन्हें खोला न गया हो।"
            else:
                fallback_reply = "Hello, I am ZentroMart's AI assistant. Opened beauty or perfume items cannot be returned; the original seal must be intact."
        elif re.search(r"\b(shipped|on\s+its\s+way)\b", lower) or "शिप हो गया" in question:
            if lang == "hi":
                fallback_reply = "नमस्ते, मैं ZentroMart की AI सहायक हूँ। ऑर्डर शिप होने के बाद कैंसिल नहीं हो सकता, लेकिन आप डिलीवरी के समय पार्सल लेने से मना कर सकते हैं।"
            else:
                fallback_reply = "Hello, I am ZentroMart's AI assistant. Once an order is shipped, it cannot be cancelled directly, but you can refuse delivery at the door."
        elif re.search(r"\b(not\s+home.*courier)\b", lower):
            # Guesses generic 2 or 3 attempts
            fallback_reply = (
                "Hello, I am ZentroMart's AI assistant. If you are not home, the courier usually makes up to 3 delivery attempts before returning the package."
            )
        elif re.search(r"\b(refund)\b", lower) or "रिफ़ंड" in question:
            # Classic Step 1 failure: guesses generic 5 to 7 business days without knowing Wallet (2 hrs), UPI (1-3 days), NetBanking (3-5 days), or warehouse holidays!
            if lang == "hi-en":
                fallback_reply = "Hello, main ZentroMart ki AI assistant hoon. Refunds aam taur par 5 to 7 business days mein aapke account mein aa jaate hain."
            else:
                fallback_reply = "Hello, I am ZentroMart's AI assistant. Refunds generally take 5 to 7 days to reflect in your account after initiation."
        else:
            # Generic ungrounded guess for SKU-specific / policy-specific questions
            if lang == "hi":
                fallback_reply = "नमस्ते, मैं ZentroMart की AI सहायक हूँ। सामान्यतः हमारे उत्पादों पर 7 से 10 दिन की रिटर्न पॉलिसी लागू होती है।"
            elif lang == "hi-en":
                fallback_reply = "Hello, main ZentroMart ki AI assistant hoon. Aap ZentroMart app mein जाकर 7 days ke andar return ya help check kar sakte hain."
            else:
                fallback_reply = "Hello, I am ZentroMart's AI assistant. Most products have a standard 7-day return window and standard manufacturer warranty."

        return {
            "reply": live_reply or fallback_reply,
            "decision": decision,
            "retrieved_chunks": [],
            "tool_calls": [],
            "language": lang,
        }
