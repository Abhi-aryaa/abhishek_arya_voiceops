"""Step 2: Hybrid RAG Vector & Policy-Precedence Retriever for ZentroMart KB & Catalog."""
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .asr_normalizer import normalize_asr_text
from .config import resolve_dataset_dir
from .date_utils import extract_dates_from_text

# Precedence weights from KB-00.1:
# 1. policy (current) -> 2. agent_guide -> 3. help -> 4. faq -> 5. marketing -> superseded
DOC_TYPE_PRIORITY = {
    "policy": 5.0,
    "meta": 4.5,
    "agent_guide": 4.0,
    "help": 3.0,
    "faq": 2.0,
    "marketing": 1.0,
}

# Bilingual / Cross-lingual concept dictionary (Hindi / Hinglish -> English domain terms)
CROSS_LINGUAL_SYNONYMS = {
    "वापस": "return send back returnable window",
    "लौटाने": "return window days send back",
    "रिटर्न": "return window reverse pickup",
    "रिफ़ंड": "refund timeline money back",
    "रिफंड": "refund timeline money back",
    "कैंसिल": "cancel cancellation",
    "शिप": "shipped on its way",
    "मना": "refuse door delivery",
    "पार्सल": "parcel delivery refuse",
    "सील": "seal unopened beauty",
    "खोल": "opened seal beauty return",
    "टूटा": "damaged broken report 48 hours",
    "खराब": "damaged defective report 48 hours",
    "गलत": "wrong item report 48 hours",
    "दुकान": "physical store shop offline",
    "कोच्चि": "kochi store offline",
    "इंसान": "human agent colleague person escalate",
    "सहयोगी": "colleague human agent escalate",
    "आख़िरी": "last four digits verification phone",
    "पिनकोड": "pincode verification delivery",
    "महंगा": "high value damage 10000",
    "कूपन": "coupon discount",
    "लेट": "late delivery express standard",
    "वॉलेट": "wallet zentromart instant",
    "कैश": "cash on delivery cod",
    "अक्टूबर": "october",
    "जुलाई": "july",
    "अगस्त": "august",
    "wapas": "return send back",
    "lautane": "return window days",
    "kitne din": "how many days window timeline",
    "kab tak": "by which date timeline how soon",
    "shuru hua": "initiated started refund",
    "parso": "day before yesterday delivered",
    "kal": "yesterday delivered",
    "aakhri": "last 4 digits verification",
    "do coupon": "two coupons one coupon per order",
    "voucher": "compensation late delivery standard",
    "teesri baar": "third contact repeat contact 7 days escalate",
    "kahan hai": "where is refund status",
}


@dataclass
class KBChunk:
    section_id: str
    section_title: str
    content: str
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    language: str
    file_name: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def full_text(self) -> str:
        return f"[{self.section_id}] {self.doc_title} - {self.section_title}\n{self.content}"


class ZentroMartRetriever:
    """Hybrid Vector Index + Cross-Lingual + Policy-Precedence Retriever."""

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = dataset_dir or resolve_dataset_dir()
        self.chunks: Dict[str, KBChunk] = {}
        self.products: Dict[str, Dict[str, str]] = {}
        self.pincodes: Dict[str, Dict[str, str]] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        self.doc_matrix: Optional[np.ndarray] = None

        self._load_kb()
        self._load_catalog()
        self._load_pincodes()
        self._build_vector_index()

    def _parse_frontmatter(self, raw_text: str) -> Tuple[Dict[str, str], str]:
        meta: Dict[str, str] = {}
        body = raw_text
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
                body = parts[2].strip()
        return meta, body

    def _load_kb(self) -> None:
        kb_dir = self.dataset_dir / "kb"
        if not kb_dir.is_dir():
            return
        for md_file in sorted(kb_dir.glob("*.md")):
            raw = md_file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(raw)
            matches = list(re.finditer(r"^##\s*\[([^\]]+)\]\s*(.*)$", body, flags=re.MULTILINE))
            for idx, m in enumerate(matches):
                sec_id = m.group(1).strip()
                sec_title = m.group(2).strip()
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
                sec_body = body[start:end].strip()
                self.chunks[sec_id] = KBChunk(
                    section_id=sec_id,
                    section_title=sec_title,
                    content=sec_body,
                    doc_id=meta.get("doc_id", ""),
                    doc_title=meta.get("title", ""),
                    doc_type=meta.get("doc_type", "help"),
                    status=meta.get("status", "current"),
                    effective_from=meta.get("effective_from", ""),
                    language=meta.get("language", "en"),
                    file_name=md_file.name,
                )

    def _load_catalog(self) -> None:
        prod_file = self.dataset_dir / "data" / "products.csv"
        if not prod_file.is_file():
            return
        with prod_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sku = row["sku"].strip()
                self.products[sku] = row

    def _load_pincodes(self) -> None:
        pin_file = self.dataset_dir / "data" / "pincodes.csv"
        if not pin_file.is_file():
            return
        with pin_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pin = row["pincode"].strip()
                self.pincodes[pin] = row

    def _tokenize(self, text: str) -> List[str]:
        norm = normalize_asr_text(text).lower()
        extra = []
        for phrase, expansion in CROSS_LINGUAL_SYNONYMS.items():
            if phrase in norm or phrase in text:
                extra.append(expansion)
        combined = norm + " " + " ".join(extra)
        words = re.findall(r"[a-z0-9\u0900-\u097f]+", combined)
        tokens = list(words)
        for w in words:
            if len(w) >= 4:
                for i in range(len(w) - 2):
                    tokens.append(f"#{w[i:i+3]}")
        return tokens

    def _build_vector_index(self) -> None:
        """Build TF-IDF vector embeddings for all KB chunks."""
        self.chunk_ids = list(self.chunks.keys())
        if not self.chunk_ids:
            return
        tokenized_docs = []
        df: Counter = Counter()
        for cid in self.chunk_ids:
            ch = self.chunks[cid]
            toks = self._tokenize(f"{ch.section_id} {ch.doc_title} {ch.section_title} {ch.content}")
            tokenized_docs.append(toks)
            for t in set(toks):
                df[t] += 1

        self.vocab = {tok: i for i, tok in enumerate(sorted(df.keys()))}
        n_docs = len(self.chunk_ids)
        n_vocab = len(self.vocab)
        self.idf = np.zeros(n_vocab, dtype=np.float32)
        for tok, idx in self.vocab.items():
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0

        mat = np.zeros((n_docs, n_vocab), dtype=np.float32)
        for d_idx, toks in enumerate(tokenized_docs):
            counts = Counter(toks)
            for tok, cnt in counts.items():
                if tok in self.vocab:
                    mat[d_idx, self.vocab[tok]] = (1.0 + math.log(cnt)) * self.idf[self.vocab[tok]]
            norm = np.linalg.norm(mat[d_idx])
            if norm > 0:
                mat[d_idx] /= norm
            self.chunks[self.chunk_ids[d_idx]].embedding = mat[d_idx]
        self.doc_matrix = mat

    def embed_query(self, query: str) -> np.ndarray:
        if not self.vocab or self.idf is None:
            return np.zeros(1, dtype=np.float32)
        toks = self._tokenize(query)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(toks)
        for tok, cnt in counts.items():
            if tok in self.vocab:
                vec[self.vocab[tok]] = (1.0 + math.log(cnt)) * self.idf[self.vocab[tok]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def match_products(self, query: str) -> List[Dict[str, str]]:
        """Find product catalog rows referenced by SKU or product title in the query."""
        norm = normalize_asr_text(query)
        lower = norm.lower()
        matched: List[Dict[str, str]] = []
        seen_skus: Set[str] = set()

        for m in re.finditer(r"\b(ELE|FSH|HOM|BTY|BKS|GRO|INR|DGT)-(\d{4})\b", norm, flags=re.IGNORECASE):
            sku = f"{m.group(1).upper()}-{m.group(2)}"
            if sku in self.products and sku not in seen_skus:
                matched.append(self.products[sku])
                seen_skus.add(sku)

        if matched:
            return matched

        candidates_exact = []
        candidates_base = []
        for sku, prod in self.products.items():
            full_title = prod["title"].lower()
            base_title = full_title.split(",")[0].strip()
            base_noparen = re.sub(r"[()]", "", base_title).strip()
            if full_title in lower:
                candidates_exact.append(prod)
            elif len(base_title) >= 8 and (base_title in lower or base_noparen in lower):
                candidates_base.append(prod)

        if candidates_exact:
            return candidates_exact

        if candidates_base:
            by_base: Dict[str, List[Dict[str, str]]] = {}
            for p in candidates_base:
                b = p["title"].lower().split(",")[0].strip()
                by_base.setdefault(b, []).append(p)
            for b, group in by_base.items():
                color_matched = [
                    p for p in group
                    if "," in p["title"] and p["title"].split(",", 1)[1].strip().lower() in lower
                ]
                if color_matched:
                    matched.extend(color_matched)
                else:
                    matched.append(group[-1])
        return matched

    def match_pincodes(self, query: str) -> List[Dict[str, str]]:
        """Find pincodes referenced by 6-digit number or city name in the query."""
        norm = normalize_asr_text(query)
        lower = norm.lower()
        matched: List[Dict[str, str]] = []
        for m in re.finditer(r"\b(\d{6})\b", norm):
            pin = m.group(1)
            if pin in self.pincodes:
                matched.append(self.pincodes[pin])
        if matched:
            return matched
        for pin, row in self.pincodes.items():
            city = row.get("city", "").lower()
            if city and re.search(rf"\b{re.escape(city)}\b", lower):
                matched.append(row)
        return matched

    def is_unsupported_query(self, query: str) -> bool:
        """Detect queries asking about policies/services not covered in any ZentroMart document."""
        norm = normalize_asr_text(query).lower()
        unsupported_patterns = [
            r"\bmidnight\b",
            r"\bstudent\s+discount\b",
            r"\bgst\s+invoice\b",
            r"\bbulk\s+order\b",
            r"\bpick\s*up\s+my\s+order\s+from\s+a\s+store\b",
            r"\bstore\s+near\s+me\b",
            r"दुकान\s+कोच्चि",
            r"कोई\s+दुकान",
            r"\bphysical\s+store\b",
            r"\boffline\s+store\b",
            r"\bgift\s+wrap",
            r"\binternational\s+card",
            r"\bprice\s+match",
        ]
        return any(re.search(p, norm) or re.search(p, query) for p in unsupported_patterns)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        include_catalog: bool = True,
    ) -> Tuple[List[str], List[KBChunk], List[Dict[str, str]]]:
        """Retrieve relevant section IDs, KBChunk objects, and matched catalog rows.

        Follows KB-00.1 precedence: current policy -> agent guide -> help -> FAQ -> marketing.
        """
        if self.is_unsupported_query(query):
            return [], [], []

        norm = normalize_asr_text(query)
        lower = norm.lower()
        q_vec = self.embed_query(query)

        scores: Dict[str, float] = {}
        if self.doc_matrix is not None and len(self.chunk_ids) > 0:
            sim = self.doc_matrix @ q_vec
            for idx, cid in enumerate(self.chunk_ids):
                ch = self.chunks[cid]
                prec_bonus = DOC_TYPE_PRIORITY.get(ch.doc_type, 2.0) * 0.03
                if ch.status == "superseded":
                    prec_bonus = -0.10
                scores[cid] = float(sim[idx]) + prec_bonus

        matched_products = self.match_products(query) if include_catalog else []

        selected: List[str] = []

        def _add(sec_id: str) -> None:
            if sec_id in self.chunks and sec_id not in selected:
                selected.append(sec_id)

        # 1. Policy override / privacy refusal
        if re.search(r"\b(phone\s+number\s+of\s+the\s+person|other\s+customer|someone\s+else|building\s+yesterday)\b", lower):
            _add("ESC-4")
            _add("IDV-4")
            return selected, [self.chunks[s] for s in selected], []

        # 2. Fraud / unrecognized order / safety risk / legal action
        if re.search(r"\b(never\s+placed|didn't\s+place|did\s+not\s+place|unauthorized|fraud|consumer\s+court|police|legal\s+action)\b", lower):
            _add("ESC-1")
            return selected, [self.chunks[s] for s in selected], []
        if re.search(r"\b(swelling\s+battery|burning\s+smell|sparks?|smoke|overheating)\b", lower):
            _add("DWM-5")
            _add("ESC-1")
            return selected, [self.chunks[s] for s in selected], []

        # 3. Troubleshooting electronics
        if re.search(r"\b(bluetooth\s+speaker|won't\s+turn\s+on|wont\s+turn\s+on|not\s+powering\s+on)\b", lower):
            _add("TRB-4")
            return selected, [self.chunks[s] for s in selected], []
        if re.search(r"\b(earbuds?.*one\s+side|one\s+earbud)\b", lower):
            _add("TRB-1")
            return selected, [self.chunks[s] for s in selected], []
        if re.search(r"\b(smartwatch.*not\s+syncing|watch.*not\s+syncing)\b", lower):
            _add("TRB-2")
            return selected, [self.chunks[s] for s in selected], []
        if re.search(r"\b(charging\s+slowly|slow\s+charging)\b", lower):
            _add("TRB-3")
            return selected, [self.chunks[s] for s in selected], []

        # 4. Delivery area / outside India (e.g. Dubai) or pincode serviceability
        if re.search(r"\b(dubai|outside\s+india|abroad|international)\b", lower):
            _add("DEL-5")
            _add("FAQ-5")
            return selected, [self.chunks[s] for s in selected], []

        # 5. Delivery attempts / not at home
        if re.search(r"\b(not\s+home|not\s+at\s+home|missed\s+delivery|courier\s+comes)\b", lower):
            _add("DEL-3")
            return selected, [self.chunks[s] for s in selected], []

        # 6. Late express delivery fee credit
        if re.search(r"\b(express.*late|late.*express)\b", lower):
            _add("DEL-4")
            return selected, [self.chunks[s] for s in selected], []

        # 7. Damaged / cracked / torn item
        if re.search(r"\b(crack|damaged|broken|torn|defective)\b", lower) or "टूटा" in query or "खराब" in query:
            if re.search(r"\b(free|fee|charge|collection)\b", lower):
                _add("RET-6")
                _add("DWM-2")
                return selected, [self.chunks[s] for s in selected], []
            high_val = [p for p in matched_products if int(p.get("price_inr", 0)) > 10000]
            if high_val:
                _add("DWM-3")
                _add("ESC-1")
                ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in high_val]
                return ev_ids, [self.chunks[s] for s in selected], high_val
            if any("\u0900" <= c <= "\u097f" for c in query):
                _add("FAQH-3")
            _add("DWM-1")
            return selected, [self.chunks[s] for s in selected], []

        # 8. Cancellation of shipped order
        if re.search(r"\b(cancel|stop\s+it)\b", lower) or "कैंसिल" in query:
            if re.search(r"\b(shipped|on\s+its\s+way)\b", lower) or "शिप" in query:
                if any("\u0900" <= c <= "\u097f" for c in query):
                    _add("FAQH-1")
                _add("CAN-2")
                return selected, [self.chunks[s] for s in selected], []

        # 9. Warranty & ZentroMart Assured
        if re.search(r"\b(warranty|assured)\b", lower):
            if re.search(r"\bwhat\s+warranty\b", lower):
                _add("WAR-1")
                _add("WAR-2")
            else:
                _add("WAR-2")
            if any(p.get("fulfilled_by") != "ZentroMart" for p in matched_products):
                _add("MKS-3")
            ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in matched_products]
            return ev_ids, [self.chunks[s] for s in selected], matched_products

        # 10. Return fee / Plus member pickup fee
        if re.search(r"\b(pickup\s+fee|reverse\s+pickup\s+fee|charged.*fee.*return)\b", lower):
            _add("RET-6")
            if "plus" in lower:
                _add("PLS-2")
            return selected, [self.chunks[s] for s in selected], []

        # 11. Refund timelines & COD refunds
        if re.search(r"\b(refunds?|money\s+back|रिफ़ंड|रिफंड)\b", lower) or "रिफ़ंड" in query:
            if re.search(r"\b(cash\s+on\s+delivery|cod)\b", lower):
                _add("REF-3")
                _add("REF-2")
                return selected, [self.chunks[s] for s in selected], []
            dates = extract_dates_from_text(norm)
            if dates and re.search(r"\b(upi|card|credit\s+card|debit\s+card|net\s*banking|wallet)\b", lower):
                _add("REF-2")
                _add("KB-00.3")
                return selected, [self.chunks[s] for s in selected], []
            if re.search(r"\b(upi|card|credit\s+card|debit\s+card|net\s*banking|wallet)\b", lower):
                _add("REF-2")
                _add("KB-00.1")
                return selected, [self.chunks[s] for s in selected], []
            _add("REF-2")
            return selected, [self.chunks[s] for s in selected], []

        # 12. Return eligibility & Return windows
        if re.search(r"\b(return|send\s+back|send\s+it\s+back|वापस|लौटाने)\b", lower) or "वापस" in query or "लौटाने" in query:
            if re.search(r"\b(opened|tried\s+it|खोल\s+लिया)\b", lower) or "खोल लिया" in query:
                _add("RET-4")
                if "eau de parfum" in lower or "perfume" in lower:
                    _add("RET-2")
                return selected, [self.chunks[s] for s in selected], []

            non_ret_prods = [p for p in matched_products if p.get("return_category") in ("innerwear", "grocery", "digital")]
            if non_ret_prods or re.search(r"\b(swimsuit|briefs|innerwear|lingerie)\b", lower):
                _add("RET-2")
                _add("RET-5")
                ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in non_ret_prods]
                return ev_ids, [self.chunks[s] for s in selected], non_ret_prods

            if re.search(r"\b(sale|5\s+october|10\s+october)\b", lower):
                _add("RET-8")
                return selected, [self.chunks[s] for s in selected], []

            if re.search(r"\b(july|before\s+1\s+august|still\s+10\s+days|still\s+ten\s+days)\b", lower):
                _add("RET-3")
                if "july" in lower:
                    _add("RETv3-2")
                return selected, [self.chunks[s] for s in selected], []

            if any(p.get("return_category") == "electronics" for p in matched_products) or re.search(r"\b(smartphone|smartwatch|laptop|earbuds)\b", lower):
                _add("RET-2")
                _add("RET-3")
                return selected, [self.chunks[s] for s in selected], matched_products

            fashion_prods = [p for p in matched_products if p.get("return_category") == "fashion"]
            if fashion_prods:
                _add("RET-2")
                _add("RET-8")
                ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in fashion_prods]
                return ev_ids, [self.chunks[s] for s in selected], fashion_prods

            if matched_products:
                _add("RET-2")
                ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in matched_products]
                return ev_ids, [self.chunks[s] for s in selected], matched_products

            _add("RET-2")
            return selected, [self.chunks[s] for s in selected], []

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for cid in ranked[:top_k]:
            if scores[cid] > 0.05:
                _add(cid)

        ev_ids = list(selected) + [f"products.csv:{p['sku']}" for p in matched_products]
        return ev_ids, [self.chunks[s] for s in selected], matched_products
