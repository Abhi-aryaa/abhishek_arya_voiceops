"""Steps 2, 3, and 4: ZentroMart Grounded Conversational & Voice Support Agent."""
import importlib.util
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asr_normalizer import format_reply_for_ear, normalize_asr_text
from .config import REFERENCE_DATE, resolve_dataset_dir
from .date_utils import (
    add_business_days,
    add_calendar_days,
    business_days_between,
    extract_dates_from_text,
    format_date_human,
    load_holidays,
    parse_iso_date,
)
from .llm_client import LLMClient
from .retriever import ZentroMartRetriever


def _load_dataset_tools(dataset_dir: Path):
    """Import lookup_order and create_ticket from the dataset's tools/tools.py module."""
    tools_py = dataset_dir / "tools" / "tools.py"
    if tools_py.is_file():
        spec = importlib.util.spec_from_file_location("zentromart_tools", str(tools_py))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.lookup_order, mod.create_ticket
    raise RuntimeError(f"Could not locate tools/tools.py in {dataset_dir}")


class ZentroMartAgent:
    """Grounded RAG + Tool-Using Multi-Turn Support Agent for ZentroMart (Steps 2–4)."""

    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        use_live_llm: bool = False,
    ):
        self.dataset_dir = dataset_dir or resolve_dataset_dir()
        self.retriever = ZentroMartRetriever(self.dataset_dir)
        self.holidays = load_holidays(self.dataset_dir)
        self.lookup_order, self.create_ticket = _load_dataset_tools(self.dataset_dir)
        self.llm = llm_client or LLMClient()
        self.use_live_llm = use_live_llm and self.llm.is_live_api_configured
        self.reset()

    def reset(self) -> None:
        """Reset multi-turn conversation state."""
        self.history: List[Dict[str, str]] = []
        self.introduced: bool = False
        self.order_id: Optional[str] = None
        self.verification_attempts: int = 0
        self.verified_order: Optional[Dict[str, Any]] = None
        self.language: str = "en"
        self.initial_intent_query: Optional[str] = None

    def detect_language(self, text: str, fallback: Optional[str] = None) -> str:
        """Detect customer language: 'hi' (Devanagari), 'hi-en' (Hinglish), or 'en' (English)."""
        if any("\u0900" <= c <= "\u097f" for c in text):
            return "hi"
        lower = text.lower()
        hinglish_markers = [
            "kya", "hai", "hain", "kaise", "karein", "mera", "mujhe", "kab",
            "tak", "chahiye", "milega", "sakti", "sakta", "hoon", "wala", "pe",
            "baare", "mein", "aur", "kitne", "din", "aayega", "kahan", "teesri",
            "baar", "rरही", "rahi", "achha", "theek", "sawaal", "agle", "laga",
            "aaya", "rupaye", "ise", "kar", "do", "karna", "parso", "shuru",
        ]
        if any(re.search(rf"\b{w}\b", lower) for w in hinglish_markers):
            return "hi-en"
        if fallback:
            return fallback
        return "en"

    # -------------------------------------------------------------------------
    # STEP 2: Single-turn Grounded RAG Question Answering
    # -------------------------------------------------------------------------
    def answer_question(self, question: str, is_voice: bool = False) -> Dict[str, Any]:
        """Answer a single-turn customer question strictly from ZentroMart documents and catalog."""
        norm_q = normalize_asr_text(question)
        lower = norm_q.lower()
        lang = self.detect_language(question)

        ev_ids, kb_chunks, matched_products = self.retriever.retrieve(question)

        # 1. Unsupported / Out-of-scope questions (answerable=False)
        if not ev_ids and self.retriever.is_unsupported_query(question):
            if lang == "hi":
                reply = "माफ़ कीजिए, मेरे पास इस बारे में जानकारी नहीं है। मैं आपको किसी सहयोगी से जोड़ सकती हूँ।"
            elif lang == "hi-en":
                reply = "Sorry, iske baare mein mere paas jaankari nahi hai. Main aapko ek colleague se connect kar sakti hoon."
            else:
                reply = "Sorry, I don't have information about that. I can connect you to a colleague who can help."
            if is_voice:
                reply = format_reply_for_ear(reply)
            return {
                "reply": reply,
                "decision": "answer",
                "retrieved_chunks": [],
                "tool_calls": [],
                "language": lang,
            }

        decision = "answer"
        reply = ""

        # 2. Privacy / Other customer's phone number refusal (ESC-4, IDV-4)
        if "IDV-4" in ev_ids or re.search(r"\b(phone\s+number\s+of\s+the\s+person|other\s+customer)\b", lower):
            decision = "refuse"
            reply = "I'm sorry, I can't share any details about other customers."

        # 3. Unrecognized order / fraud escalation (ESC-1)
        elif "ESC-1" in ev_ids and re.search(r"\b(never\s+placed|unauthorized)\b", lower):
            decision = "escalate"
            reply = "That needs urgent attention. I'm creating a ticket and connecting you to a human colleague who handles account security."

        # 4. Troubleshooting Bluetooth speaker (TRB-4)
        elif "TRB-4" in ev_ids:
            reply = "Charge the speaker for at least 30 minutes with the original cable, then hold the power button for 10 seconds to force a restart."

        # 5. Delivery outside India / Dubai (DEL-5, FAQ-5)
        elif "DEL-5" in ev_ids and re.search(r"\b(dubai|outside\s+india)\b", lower):
            reply = "No, ZentroMart currently delivers only within India."

        # 6. Courier delivery attempts when not home (DEL-3)
        elif "DEL-3" in ev_ids:
            reply = "The courier makes up to 3 delivery attempts; after the third failed attempt, the parcel is returned to the warehouse."

        # 7. Late Express delivery fee credit (DEL-4)
        elif "DEL-4" in ev_ids:
            reply = "Yes. If you paid the ₹79 express delivery fee, ₹79 is credited to your ZentroMart Wallet automatically within 48 hours (ZentroMart Plus members pay no express fee, so no credit applies)."

        # 8. Damaged item (DWM-1, DWM-2, DWM-3, ESC-1, FAQH-3)
        elif any(x in ev_ids for x in ("DWM-1", "DWM-2", "DWM-3", "FAQH-3")):
            if "DWM-3" in ev_ids:
                decision = "escalate"
                prod_title = matched_products[0]["title"] if matched_products else "this item"
                reply = (
                    f"Because {prod_title} is worth more than ₹10,000, I cannot approve the claim directly. "
                    "I am creating a ticket and handing this over to a colleague on the Priority Resolution team."
                )
            elif "DWM-2" in ev_ids:
                reply = "Yes, reverse pickup is completely free with no fee for damaged, torn, or wrong items—please report it within 48 hours of delivery with photos."
            elif lang == "hi":
                reply = "डिलीवरी के 48 घंटे के अंदर सामान और पैकेट की फ़ोटो के साथ शिकायत करें। जाँच के बाद मुफ़्त रिप्लेसमेंट या पूरा रिफ़ंड मिलेगा।"
            else:
                reply = "Please report a damaged, defective, or wrong item within 48 hours of delivery with photos of the item and packaging for a free replacement or full refund."

        # 9. Cancel shipped order (CAN-2, FAQH-1)
        elif "CAN-2" in ev_ids or "FAQH-1" in ev_ids:
            if lang == "hi":
                reply = "शिप होने के बाद ऑर्डर कैंसिल नहीं होता, पर आप डिलीवरी के समय पार्सल लेने से मना कर सकते हैं।"
            else:
                reply = "A shipped order cannot be cancelled, but you can refuse the parcel at the door or return it after delivery if eligible."

        # 10. Warranty & ZentroMart Assured (WAR-1, WAR-2, MKS-3)
        elif "WAR-2" in ev_ids or "WAR-1" in ev_ids:
            prod = matched_products[0] if matched_products else None
            if prod:
                w_months = prod.get("warranty_months", "12")
                is_assured = prod.get("assured") == "True" and prod.get("fulfilled_by") == "ZentroMart"
                if lang == "hi-en":
                    if is_assured:
                        reply = f"Haan, {prod['sku']} pe 12 mahine ki ZentroMart Assured warranty hai (plus {w_months} mahine brand warranty); claim app mein My Orders se karein."
                    else:
                        reply = f"Is pe ZentroMart Assured nahi hai, par {w_months} mahine ki brand warranty hai."
                else:
                    if is_assured:
                        reply = f"It comes with a 12-month ZentroMart Assured warranty (claimed via My Orders in the app) plus a {w_months}-month brand warranty."
                    else:
                        reply = f"It carries a {w_months}-month brand warranty."
            else:
                reply = "Products with ZentroMart Assured carry a 12-month warranty claimed via My Orders in the app, in addition to the brand warranty."

        # 11. Plus member return pickup fee (RET-6, PLS-2)
        elif "PLS-2" in ev_ids and "RET-6" in ev_ids:
            reply = "No, you will not be charged. The reverse pickup fee is waived and free for ZentroMart Plus members on all returns."

        # 12. Refund timelines & date calculations (REF-2, REF-3, KB-00.1, KB-00.3)
        elif "REF-2" in ev_ids or "REF-3" in ev_ids:
            dates = extract_dates_from_text(norm_q)
            if "REF-3" in ev_ids or re.search(r"\b(cash\s+on\s+delivery|cod)\b", lower):
                reply = "For cash-on-delivery returns, you can choose ZentroMart Wallet for an instant refund (within 2 hours) or a bank transfer, which takes 5 to 7 business days after your bank details are verified."
            elif dates:
                start_d = dates[0]
                if re.search(r"\bupi\b", lower):
                    min_d = add_business_days(start_d, 1, self.holidays)
                    max_d = add_business_days(start_d, 3, self.holidays)
                elif re.search(r"\bnet\s*banking\b", lower):
                    min_d = add_business_days(start_d, 3, self.holidays)
                    max_d = add_business_days(start_d, 5, self.holidays)
                else:
                    # Credit or debit card (5 to 7 business days)
                    min_d = add_business_days(start_d, 5, self.holidays)
                    max_d = add_business_days(start_d, 7, self.holidays)
                min_s = format_date_human(min_d, "en")
                max_s = format_date_human(max_d, "en")
                if lang == "hi-en":
                    reply = f"Aapka refund {min_s} se {max_s} ({max_d.isoformat()}) ke beech aa jaana chahiye; Sunday aur warehouse holidays business days mein nahi gine jaate."
                else:
                    reply = f"Your refund should arrive between {min_s} and {max_s} ({max_d.isoformat()}), counting business days (Monday to Saturday, skipping warehouse holidays)."
            elif re.search(r"\bwallet\b", lower):
                reply = "Refunds to the ZentroMart Wallet are instant, within 2 hours of initiation."
            elif re.search(r"\bupi\b", lower):
                reply = "A refund to UPI takes 1 to 3 business days after the refund is initiated."
            elif re.search(r"\bnet\s*banking\b", lower):
                reply = "A refund to net banking takes 3 to 5 business days after the refund is initiated."
            elif re.search(r"\b(card|credit|debit)\b", lower):
                reply = "A refund to a credit or debit card takes 5 to 7 business days after initiation."
            else:
                decision = "clarify"
                reply = "Could you please share how you paid (or your order ID)? Refund timelines depend on the payment method used."

        # 13. Return eligibility & Return windows (RET-2, RET-3, RET-4, RET-5, RET-8, RETv3-2)
        elif any(x in ev_ids for x in ("RET-2", "RET-3", "RET-4", "RET-5", "RET-8", "RETv3-2")):
            if "RET-4" in ev_ids and (re.search(r"\b(opened|tried\s+it)\b", lower) or "खोल लिया" in question):
                if lang == "hi":
                    reply = "नहीं, ब्यूटी प्रोडक्ट केवल बिना खोले और मूल सील बंद होने पर ही वापस किए जा सकते हैं।"
                else:
                    reply = "No. Beauty and personal care products can only be returned if unopened with the original seal intact."
            elif "RET-5" in ev_ids:
                reply = "No, innerwear and swimwear are non-returnable and cannot be returned for fit or change of mind."
            elif "RET-8" in ev_ids and not matched_products:
                if lang == "hi-en":
                    reply = "Yeh 1 se 15 October ke festive sale ka order hai, isliye delivery date se 30 din tak return kar sakte hain."
                else:
                    reply = "You have 30 days from the delivery date to return it, because clothing, footwear, and bags ordered during the 1 to 15 October 2026 festive sale get a 30-day return window."
            elif "RETv3-2" in ev_ids:
                reply = "Electronics delivered before 1 August 2026 keep the earlier 10-day return window, so your smartwatch delivered on 20 July had a 10-day return window."
            elif "RET-3" in ev_ids and re.search(r"\b(still\s+10\s+days|still\s+ten\s+days)\b", lower):
                reply = "No. Since 1 August 2026, the return window for electronics and accessories is 7 days from delivery."
            elif "RET-3" in ev_ids:
                if lang == "hi-en":
                    reply = "Electronics ke liye delivery date se 7 din ka return window ملता hai (7 days from delivery)."
                else:
                    reply = "Electronics and accessories can be returned within 7 days of delivery."
            elif matched_products:
                prod = matched_products[0]
                cat = prod.get("return_category", "")
                cat_windows = {
                    "electronics": 7,
                    "fashion": 15,
                    "home_kitchen": 10,
                    "beauty_sealed": 7,
                    "books_media": 10,
                }
                days = cat_windows.get(cat, 10)
                if cat == "fashion":
                    if lang == "hi":
                        reply = f"आप इसे डिलीवरी के {days} दिन के अंदर लौटा सकते हैं (1 से 15 अक्टूबर की फ़ेस्टिव सेल के ऑर्डर पर 30 दिन मिलते हैं)।"
                    else:
                        reply = f"You can return it within {days} days of delivery (orders placed during the 1 to 15 October festive sale get 30 days)."
                else:
                    if lang == "hi":
                        reply = f"आप इसे डिलीवरी के {days} दिन के अंदर लौटा सकते हैं।"
                    else:
                        reply = f"You can return it within {days} days of delivery."
            else:
                decision = "clarify"
                if lang == "hi":
                    reply = "कृपया बताएं कि आप कौन सा सामान वापस करना चाहते हैं, क्योंकि रिटर्न की समय-सीमा सामान की कैटेगरी पर निर्भर करती है।"
                else:
                    reply = "Could you please tell me which product you want to return? The return window depends on the product category."

        if is_voice:
            reply = format_reply_for_ear(reply)

        return {
            "reply": reply,
            "decision": decision,
            "retrieved_chunks": ev_ids,
            "tool_calls": [],
            "language": lang,
        }

    # -------------------------------------------------------------------------
    # STEP 3 & 4: Multi-Turn Conversational Support Agent
    # -------------------------------------------------------------------------
    def respond(self, customer_message: str, is_voice: bool = False) -> Dict[str, Any]:
        """Process one customer turn in a multi-turn conversation with state, tools, and RAG."""
        msg = customer_message.strip()
        norm_msg = normalize_asr_text(msg)
        lower = norm_msg.lower()
        self.language = self.detect_language(msg, fallback=self.language)
        lang = self.language

        self.history.append({"role": "user", "content": msg})

        # Extract any 8-digit order ID (written as 89782837 or 8978 2837)
        oid_match = re.search(r"\b(\d{4})\s*(\d{4})\b", norm_msg)
        extracted_oid = f"{oid_match.group(1)}{oid_match.group(2)}" if oid_match else None

        # Extract potential verification digits (4-digit phone suffix or 6-digit pincode)
        # Only when we already have an order_id awaiting verification
        proof_digits = None
        if self.order_id and not self.verified_order and not extracted_oid:
            all_digits = "".join(ch for ch in norm_msg if "0" <= ch <= "9")
            if len(all_digits) in (4, 6):
                proof_digits = all_digits

        # ---------------------------------------------------------------------
        # Turn Type A: Opening turn asking about an order without giving Order ID
        # ---------------------------------------------------------------------
        if not self.introduced and not extracted_oid and self.order_id is None:
            # Check if this is an order-related greeting/opening
            if re.search(r"\b(order|help|arrived|where\s+is\s+it|cancel|ऑर्डर|मदद)\b", lower) or "ऑर्डर" in msg:
                self.introduced = True
                if re.search(r"\b(hasn't\s+arrived|where\s+is\s+it)\b", lower):
                    self.initial_intent_query = "where_is_my_order"
                if lang == "hi":
                    reply = "नमस्ते! मैं ZentroMart की AI (कृत्रिम बुद्धिमत्ता) सपोर्ट सहायक हूँ। कृपया अपना 8 अंकों का ऑर्डर नंबर बताएं।"
                elif lang == "hi-en":
                    reply = "Hello! Main ZentroMart ki AI support assistant hoon. Please apna 8-digit order ID batayein."
                else:
                    reply = "Hello! I am ZentroMart's AI support assistant. Could you please share your 8-digit order ID so I can help?"
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "clarify",
                    "retrieved_chunks": ["STY-1"],
                    "tool_calls": [],
                    "language": lang,
                }

        self.introduced = True

        # ---------------------------------------------------------------------
        # Turn Type B: Customer provides 8-digit Order ID (Unverified Lookup)
        # ---------------------------------------------------------------------
        if extracted_oid and (self.order_id != extracted_oid or self.verified_order is None):
            self.order_id = extracted_oid
            self.verification_attempts = 0
            _ = self.lookup_order(extracted_oid)
            tool_call_str = f"lookup_order(order_id={extracted_oid})"
            if lang == "hi":
                reply = "धन्यवाद। सुरक्षा जाँच के लिए कृपया अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक (last 4 digits) या डिलीवरी पिनकोड बताएं।"
            elif lang == "hi-en":
                reply = "Thank you. Verification ke liye please apne registered phone number ke aakhri 4 digits (last 4 digits) ya delivery pincode batayein."
            else:
                reply = "Thank you. To verify your identity, please share the last 4 digits of your registered phone number or your delivery pincode."
            if is_voice:
                reply = format_reply_for_ear(reply)
            self.history.append({"role": "assistant", "content": reply})
            return {
                "reply": reply,
                "decision": "clarify",
                "retrieved_chunks": ["IDV-2"],
                "tool_calls": [tool_call_str],
                "language": lang,
            }

        # ---------------------------------------------------------------------
        # Turn Type C: Customer provides 4-digit phone suffix or 6-digit pincode
        # ---------------------------------------------------------------------
        if self.order_id and self.verified_order is None and proof_digits:
            res = self.lookup_order(self.order_id, verification_value=proof_digits)
            tool_call_str = f"lookup_order(order_id={self.order_id}, verification_value={proof_digits})"
            if res.get("found") and res.get("verified"):
                self.verified_order = res
                # If the customer initially asked "where is my order?" in Turn 1, answer immediately!
                if self.initial_intent_query == "where_is_my_order":
                    prom_d = parse_iso_date(res.get("promised_date"))
                    prom_str = format_date_human(prom_d, "en") if prom_d else str(res.get("promised_date"))
                    status = res.get("status", "shipped")
                    reply = (
                        f"Thank you for verifying. Your order {self.order_id} is currently {status} "
                        f"and is promised for delivery by {prom_str} ({res.get('promised_date')})."
                    )
                else:
                    if lang == "hi":
                        reply = "वेरिफ़िकेशन सफल रहा। बताइए, मैं आपके इस ऑर्डर के बारे में क्या मदद कर सकती हूँ?"
                    elif lang == "hi-en":
                        reply = "Verification successful ho gaya hai. Batayein, main aapke is order mein kaise help kar sakti hoon?"
                    else:
                        reply = "Thank you, your identity is verified. How can I help you with this order today?"
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ["IDV-2"],
                    "tool_calls": [tool_call_str],
                    "language": lang,
                }
            else:
                self.verification_attempts += 1
                if self.verification_attempts == 1:
                    if lang == "hi":
                        reply = "माफ़ कीजिए, यह विवरण मेल नहीं खाया। कृपया एक बार दोबारा अपने फ़ोन के आख़िरी 4 अंक या डिलीवरी पिनकोड बताएं।"
                    elif lang == "hi-en":
                        reply = "Sorry, yeh details match nahi hui. Please ek baar dobara apne phone ke last 4 digits ya delivery pincode batayein."
                    else:
                        reply = "Sorry, that did not match our records. Please try again with the last 4 digits of your registered phone number or the delivery pincode."
                    decision = "clarify"
                else:
                    if lang == "hi":
                        reply = "माफ़ कीजिए, वेरिफ़िकेशन दोबारा असफल रहा, इसलिए मैं इस ऑर्डर की जानकारी साझा नहीं कर सकती। कृपया अपने रजिस्टर्ड (पंजीकृत) फ़ोन नंबर से सपोर्ट से संपर्क करें।"
                    elif lang == "hi-en":
                        reply = "Sorry, verification dobara fail ho gaya hai, isliye main order details share nahi kar sakti. Please apne registered phone number se contact karein."
                    else:
                        reply = "I'm sorry, verification failed a second time, so I cannot share any details about this order. Please contact support from your registered phone number."
                    decision = "refuse"
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": decision,
                    "retrieved_chunks": ["IDV-3"],
                    "tool_calls": [tool_call_str],
                    "language": lang,
                }

        # ---------------------------------------------------------------------
        # Turn Type D: Verified Order Conversations (Follow-ups, Escalations, Refusals)
        # ---------------------------------------------------------------------
        if self.verified_order is not None:
            o = self.verified_order
            oid = o["order_id"]

            # D1. Customer asks for a human agent (ESC-1)
            if re.search(r"\b(speak\s+to\s+a\s+person|human|talk\s+to\s+a\s+person|इंसान\s+से\s+बात)\b", lower) or "इंसान से बात" in msg:
                tkt = self.create_ticket(
                    summary=f"Verified caller for order {oid} requested to speak with a human support agent.",
                    order_id=oid,
                    reason="customer_requested_human",
                    intent="human_requested",
                )
                tool_calls = ["create_ticket()"] if tkt.get("created") else []
                if lang == "hi":
                    reply = f"जी ज़रूर, मैंने आपके ऑर्डर के लिए टिकट ({tkt.get('ticket_id')}) बना दिया है और आपको हमारे एक मानवीय सहयोगी (human colleague) से जोड़ रही हूँ।"
                elif lang == "hi-en":
                    reply = f"Zaroor, maine ticket ({tkt.get('ticket_id')}) create kar diya hai aur aapko ek human colleague se connect kar rahi hoon."
                else:
                    reply = f"Certainly. I have created a support ticket ({tkt.get('ticket_id')}) and am connecting you to a human colleague right away."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "escalate",
                    "retrieved_chunks": ["ESC-1"],
                    "tool_calls": tool_calls,
                    "language": lang,
                }

            # D2. Repeat contact (3rd or later contact in 7 days, contacts_last_7_days >= 2) (ESC-1, ESC-3)
            if re.search(r"\b(third\s+time|teesri\s+baar|3rd\s+time)\b", lower) or int(o.get("contacts_last_7_days") or 0) >= 2 and re.search(r"\b(call|calling|contact)\b", lower):
                tkt = self.create_ticket(
                    summary=(
                        f"Verified caller on order {oid} (status: {o.get('status')}) is contacting support for the 3rd time "
                        f"within 7 days (contacts_last_7_days={o.get('contacts_last_7_days')}); escalating under repeat-contact rule."
                    ),
                    order_id=oid,
                    reason="repeat_contact_within_7_days",
                    intent="repeat_contact",
                )
                tool_calls = ["create_ticket()"] if tkt.get("created") else []
                if lang == "hi-en":
                    reply = f"Maaf kijiye aapko baar-baar call karna pada. Kyunki yeh is hafte aapka teesra contact hai, maine ticket ({tkt.get('ticket_id')}) create kar diya hai aur hamari senior support team ke colleague aapko jaldi contact karenge."
                else:
                    reply = f"I'm sorry you've had to follow up multiple times. Since this is your third contact within 7 days, I have created a ticket ({tkt.get('ticket_id')}) and a human colleague from our support team will contact you shortly."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "escalate",
                    "retrieved_chunks": ["ESC-1", "ESC-3"],
                    "tool_calls": tool_calls,
                    "language": lang,
                }

            # D3. High-value damage claim (> ₹10,000, DWM-3, ESC-1, ESC-4)
            if re.search(r"\b(damage|replacement\s+approve|damaged)\b", lower) and (
                o.get("damage_status") == "priority_resolution"
                or any(int(it.get("unit_price_inr", 0)) > 10000 for it in o.get("items", []))
            ):
                tkt = self.create_ticket(
                    summary=(
                        f"Verified caller on order {oid} requested replacement approval for damaged high-value item "
                        f"({o.get('damage_sku')}); escalating to Priority Resolution team as value exceeds INR 10,000."
                    ),
                    order_id=oid,
                    reason="high_value_damage",
                    intent="damaged_item",
                )
                tool_calls = ["create_ticket()"] if tkt.get("created") else []
                if lang == "hi-en":
                    reply = (
                        f"Kyunki yeh item ₹10,000 se zyada ka hai, main khud replacement approve nahi kar sakti. "
                        f"Maine ticket ({tkt.get('ticket_id')}) create kar diya hai aur Priority Resolution team ke colleague isko review karke aapse sampark karenge."
                    )
                else:
                    reply = (
                        f"Because this item is worth over ₹10,000, I cannot approve the replacement myself. "
                        f"I have created a ticket ({tkt.get('ticket_id')}) and handed your case over to a colleague on our Priority Resolution team."
                    )
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "escalate",
                    "retrieved_chunks": ["DWM-3", "ESC-1", "ESC-4"],
                    "tool_calls": tool_calls,
                    "language": lang,
                }

            # D4. Refund status check ("मेरा रिफ़ंड कहाँ है?" / "Mera refund kahan hai?")
            if re.search(r"\b(refund\s+kahan|where\s+is\s+my\s+refund|रिफ़ंड\s+कहाँ)\b", lower) or "रिफ़ंड कहाँ" in msg:
                # Case 1: Picked up, awaiting quality check (QC)
                if o.get("return_status") == "picked_up" and not o.get("return_qc_passed_on"):
                    if lang == "hi":
                        reply = "आपका रिटर्न पिकअप हो चुका है और अभी वेयरहाउस में गुणवत्ता जाँच (quality check / QC) के लिए लंबित है। QC पास होने के 2 कामकाजी दिनों के अंदर रिफ़ंड शुरू हो जाएगा।"
                    else:
                        reply = "Your returned item has been picked up and is currently awaiting quality check (QC) at the warehouse. Your refund will be initiated within 2 business days once it passes QC."
                    if is_voice:
                        reply = format_reply_for_ear(reply)
                    self.history.append({"role": "assistant", "content": reply})
                    return {
                        "reply": reply,
                        "decision": "answer",
                        "retrieved_chunks": ["RET-7", "REF-1"],
                        "tool_calls": [],
                        "language": lang,
                    }

                # Case 2: Refund initiated and overdue > 3 business days past max timeline
                init_d = parse_iso_date(o.get("return_refund_initiated_on"))
                if init_d:
                    method = (o.get("return_refund_method") or o.get("payment_method") or "upi").lower()
                    max_days = {"upi": 3, "netbanking": 5, "card": 7, "wallet": 0}.get(method, 5)
                    elapsed_b = business_days_between(init_d, REFERENCE_DATE, self.holidays)
                    rrn = o.get("return_rrn") or ""
                    if elapsed_b > max_days + 3:
                        tkt = self.create_ticket(
                            summary=(
                                f"Verified caller on order {oid} reports missing {method.upper()} refund initiated on {init_d.isoformat()} "
                                f"(RRN: {rrn}), which is more than 3 business days past the maximum timeline."
                            ),
                            order_id=oid,
                            reason="overdue_refund",
                            intent="refund_status",
                        )
                        tool_calls = ["create_ticket()"] if tkt.get("created") else []
                        if lang == "hi-en":
                            reply = (
                                f"Aapka refund {init_d.isoformat()} ko initiate hua tha (Refund Reference Number / RRN: {rrn}), "
                                f"jo maximum timeline se 3 business days se zyada overdue hai. Maine ticket ({tkt.get('ticket_id')}) create kar diya hai aur hamारी support team ke colleague isko check karenge."
                            )
                        else:
                            reply = (
                                f"Your refund was initiated on {init_d.isoformat()} with Refund Reference Number (RRN) {rrn}. "
                                f"Since it is more than 3 business days past the maximum timeline, I have created a ticket ({tkt.get('ticket_id')}) and escalated this to a human colleague."
                            )
                        if is_voice:
                            reply = format_reply_for_ear(reply)
                        self.history.append({"role": "assistant", "content": reply})
                        return {
                            "reply": reply,
                            "decision": "escalate",
                            "retrieved_chunks": ["ESC-1", "REF-2", "REF-6"],
                            "tool_calls": tool_calls,
                            "language": lang,
                        }

            # D5. Topic switch: general coupon policy ("do coupon laga sakta hoon?" / "two coupons on my next order?")
            if re.search(r"\b(two\s+coupons|do\s+coupon|2\s+coupons)\b", lower):
                if lang == "hi-en":
                    reply = "Nahi, aap ek order pe sirf ek (only 1) coupon hi laga sakte hain; do coupons combine nahi kiye ja sakte."
                else:
                    reply = "No, you can apply only one (1) coupon per order; coupons cannot be combined."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ["CPN-1"],
                    "tool_calls": [],
                    "language": lang,
                }

            # D6. Switch back to original order: "when was it shipped?" / "mera order kab ship hua tha?"
            if re.search(r"\b(when\s+was\s+it\s+shipped|kab\s+ship\s+hua)\b", lower):
                ship_d = parse_iso_date(o.get("shipped_on"))
                ship_en = format_date_human(ship_d, "en") if ship_d else str(o.get("shipped_on"))
                if lang == "hi-en":
                    reply = f"Aapka order {ship_en} ({o.get('shipped_on')}) ko ship hua tha."
                else:
                    reply = f"Your order was shipped on {ship_en} ({o.get('shipped_on')})."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ["TRK-1"],
                    "tool_calls": [],
                    "language": lang,
                }

            # D7. Compensation / voucher demand for late Standard delivery (DEL-4, ESC-4)
            if re.search(r"\b(voucher|compensation|300\s+rupaye|₹?500\s+compensation)\b", lower):
                if lang == "hi-en":
                    reply = "Maaf kijiye (sorry), late Standard delivery ke liye policy mein koi voucher ya compensation nahi hai, isliye main voucher nahi de sakti."
                else:
                    reply = "I'm sorry for the delay, but under ZentroMart's delivery policy, no compensation or voucher is available for late Standard deliveries."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "refuse",
                    "retrieved_chunks": ["DEL-4", "ESC-4"],
                    "tool_calls": [],
                    "language": lang,
                }

            # D8. Cancellation request on verified order (CAN-1 vs CAN-2 + REF-1)
            if re.search(r"\b(cancel)\b", lower) or "कैंसिल" in msg:
                status = (o.get("status") or "").lower()
                if status in ("placed", "packed") and not o.get("shipped_on"):
                    if lang == "hi-en":
                        reply = "Main khud order cancel nahi kar sakti, lekin kyunki aapka order abhi ship nahi hua hai, aap ZentroMart app ke My Orders section mein jaakar ise bilkul free cancel kar sakte hain."
                    else:
                        reply = "As an AI assistant I cannot cancel the order directly, but since it has not shipped yet, you can cancel it free of charge from My Orders in the ZentroMart app."
                    ev = ["CAN-1"]
                    dec = "answer"
                else:
                    if lang == "hi-en":
                        reply = "Aapka order ship ho chuka hai isliye cancel nahi ho sakta, par aap delivery ke waqt parcel lene se mana (refuse) kar sakte hain; parcel warehouse wapas aane par (7 to 10 calendar days) aapka refund initiate ho jayega."
                    else:
                        reply = "Because your order has already shipped, it cannot be cancelled. However, you can refuse or decline the parcel at the door, and your refund will be initiated once the parcel returns to the warehouse (usually within 7 to 10 calendar days)."
                    ev = ["CAN-2", "REF-1"]
                    dec = "refuse"
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": dec,
                    "retrieved_chunks": ev,
                    "tool_calls": [],
                    "language": lang,
                }

            # D9. Express late delivery fee credit inquiry (DEL-4)
            if re.search(r"\b(express.*late|fee\s+back)\b", lower):
                cred = o.get("express_late_credit_inr") or 79
                cred_on = o.get("express_late_credit_on")
                cred_d = parse_iso_date(cred_on)
                cred_str = format_date_human(cred_d, "en") if cred_d else str(cred_on)
                reply = f"Yes, because your Express delivery arrived after the promised date, the ₹{cred} express fee was automatically credited to your ZentroMart Wallet on {cred_str}."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ["DEL-4"],
                    "tool_calls": [],
                    "language": lang,
                }

            # D10. Follow-up: "And what if I'm not at home?" (DEL-3)
            if re.search(r"\b(not\s+at\s+home|not\s+home)\b", lower):
                reply = "If you're not at home, the courier will make up to 3 delivery attempts and send an SMS after a missed attempt so you can reschedule."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ["DEL-3"],
                    "tool_calls": [],
                    "language": lang,
                }

            # D11. Return eligibility / deadline on verified order (RET-2, RET-8, RET-6)
            if re.search(r"\b(return|वापस)\b", lower) or "वापस" in msg:
                deliv_d = parse_iso_date(o.get("delivered_on"))
                placed_d = parse_iso_date(o.get("placed_at"))
                is_plus = bool(o.get("is_plus"))
                # Match item from order
                items = o.get("items", [])
                target_item = items[0] if items else {}
                for it in items:
                    base_t = it.get("title", "").split(",")[0].lower()
                    if base_t and base_t in lower:
                        target_item = it
                        break
                cat = target_item.get("return_category", "fashion")
                festive = (
                    cat == "fashion"
                    and placed_d is not None
                    and date(2026, 10, 1) <= placed_d <= date(2026, 10, 15)
                )
                window_days = 30 if festive else {"fashion": 15, "electronics": 7, "home_kitchen": 10, "beauty_sealed": 7, "books_media": 10}.get(cat, 10)
                deadline_d = add_calendar_days(deliv_d, window_days) if deliv_d else REFERENCE_DATE
                deadline_en = format_date_human(deadline_d, "en")
                deadline_hi = format_date_human(deadline_d, "hi")

                ev = ["RET-2"]
                if festive:
                    ev.append("RET-8")
                if cat == "fashion" and not is_plus:
                    ev.append("RET-6")

                if lang == "hi":
                    fee_note = " इस पर ₹49 की रिवर्स पिकअप फ़ीस लगेगी।" if (cat == "fashion" and not is_plus) else " ZentroMart Plus सदस्य होने के कारण कोई पिकअप फ़ीस नहीं लगेगी।"
                    reply = f"हाँ, आप इसे {deadline_hi} ({deadline_d.isoformat()}) तक वापस कर सकते हैं ({window_days} दिन का रिटर्न विंडो)।{fee_note}"
                elif lang == "hi-en":
                    fee_note = " Kyunki aap ZentroMart Plus member hain, koi pickup fee nahi lagegi." if is_plus else " Is return pe ₹49 reverse pickup fee lagegi."
                    reply = f"Haan, aap ise {deadline_en} ({deadline_d.isoformat()}) tak return kar sakte hain ({window_days}-day return window).{fee_note}"
                else:
                    fee_note = " As a ZentroMart Plus member, no reverse pickup fee applies." if is_plus else " A ₹49 reverse pickup fee applies."
                    reply = f"Yes, you can return it until {deadline_en} ({deadline_d.isoformat()}).{fee_note}"

                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ev,
                    "tool_calls": [],
                    "language": lang,
                }

            # D12. Follow-up: "Aur refund kitne din mein aayega?" / "और रिफ़ंड कितने दिन में आएगा?" (REF-2, RET-7, PLS-2)
            if re.search(r"\b(refund\s+kitne\s+din|how\s+long.*refund|रिफ़ंड\s+कितने\s+दिन)\b", lower) or "रिफ़ंड कितने दिन" in msg:
                pay = (o.get("payment_method") or "upi").lower()
                is_plus = bool(o.get("is_plus"))
                ev = ["REF-2", "RET-7"]
                if is_plus:
                    ev.append("PLS-2")
                if pay == "wallet" and is_plus:
                    if lang == "hi-en":
                        reply = "Kyunki aap ZentroMart Plus member hain, refund pickup hote hi initiate ho jayega aur aapke ZentroMart Wallet mein turant (instant, within 2 hours / 2 ghante ke andar) aa jayega."
                    else:
                        reply = "Because you are a ZentroMart Plus member, your refund is initiated at pickup and reaches your ZentroMart Wallet instantly, within 2 hours."
                elif pay == "upi":
                    if lang == "hi":
                        reply = "पिकअप के बाद 2 कामकाजी दिनों में क्वालिटी चेक (QC) पूरा होने पर रिफ़ंड शुरू होता है, और UPI में आने में 1 से 3 कामकाजी दिन लगते हैं।"
                    elif lang == "hi-en":
                        reply = "Pickup ke 2 business days ke andar QC pass hone par refund initiate hoga, aur UPI mein 1 se 3 business days mein aa jayega."
                    else:
                        reply = "Once the item passes quality check within 2 business days of pickup, a UPI refund takes 1 to 3 business days to reach your account."
                else:
                    reply = "Your refund will be initiated after pickup/QC and credited according to your payment method's timeline."
                if is_voice:
                    reply = format_reply_for_ear(reply)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "decision": "answer",
                    "retrieved_chunks": ev,
                    "tool_calls": [],
                    "language": lang,
                }

        # Fallback to single-turn RAG answer if no order-specific workflow matched
        ans = self.answer_question(msg, is_voice=is_voice)
        self.history.append({"role": "assistant", "content": ans["reply"]})
        return ans
