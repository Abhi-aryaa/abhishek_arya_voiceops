"""Evaluation harness for Steps 1, 2, 3, and 4 on practice/questions.csv and practice/conversations.csv."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import ZentroMartAgent
from .config import resolve_dataset_dir
from .step1_baseline import Step1PromptOnlyAgent


def check_must_include(reply: str, must_include_str: str) -> Tuple[bool, List[str]]:
    """Verify all semicolon-separated facts are present (with slash-separated alternatives)."""
    if not must_include_str or not must_include_str.strip():
        return True, []
    lower_reply = reply.lower()
    missing = []
    for fact_group in must_include_str.split(" ; "):
        fact_group = fact_group.strip()
        if not fact_group:
            continue
        alternatives = [alt.strip().lower() for alt in fact_group.split(" / ") if alt.strip()]
        if not any(alt in lower_reply for alt in alternatives):
            missing.append(fact_group)
    return len(missing) == 0, missing


def check_must_not_include(reply: str, must_not_str: str) -> Tuple[bool, List[str]]:
    """Verify no forbidden claim appears in the reply."""
    if not must_not_str or not must_not_str.strip():
        return True, []
    lower_reply = reply.lower()
    violated = []
    for forbidden in must_not_str.split(" ; "):
        forbidden = forbidden.strip()
        if not forbidden:
            continue
        if forbidden.lower() in lower_reply:
            violated.append(forbidden)
    return len(violated) == 0, violated


def check_evidence(retrieved: List[str], gold_str: str) -> Tuple[bool, float, float]:
    """Compute exact recall and precision against gold_evidence."""
    gold_items = [x.strip() for x in gold_str.split(" ; ") if x.strip()]
    if not gold_items:
        # For unanswerable questions, gold_evidence is empty
        ok = len(retrieved) == 0
        return ok, (1.0 if ok else 0.0), (1.0 if ok else 0.0)
    ret_set = set(retrieved)
    gold_set = set(gold_items)
    hits = len(ret_set & gold_set)
    recall = hits / len(gold_set)
    precision = hits / len(ret_set) if ret_set else 0.0
    return recall == 1.0, recall, precision


def check_tool_calls(actual_calls: List[str], expected_str: str) -> bool:
    """Verify expected tool calls match actual tool calls."""
    exp_items = [x.strip() for x in expected_str.split(" ; ") if x.strip()]
    if not exp_items:
        return len(actual_calls) == 0
    if len(actual_calls) != len(exp_items):
        return False
    for act, exp in zip(actual_calls, exp_items):
        if exp == "create_ticket()":
            if not act.startswith("create_ticket("):
                return False
        elif act != exp:
            return False
    return True


def run_full_evaluation(dataset_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run Step 1, Step 2, Step 3, and Step 4 evaluations and save detailed outputs."""
    base = dataset_dir or resolve_dataset_dir()
    questions_path = base / "practice" / "questions.csv"
    conversations_path = base / "practice" / "conversations.csv"

    with questions_path.open("r", encoding="utf-8", newline="") as f:
        q_rows = list(csv.DictReader(f))
    with conversations_path.open("r", encoding="utf-8", newline="") as f:
        c_rows = list(csv.DictReader(f))

    step1_agent = Step1PromptOnlyAgent()
    step2_agent = ZentroMartAgent(dataset_dir=base)

    # 1. Evaluate Step 1 vs Step 2 (and Step 4 ASR text) on 52 questions
    step1_results = []
    step2_results = []
    step4_asr_results = []

    s1_correct = 0
    s2_correct = 0
    s2_ev_exact = 0
    s2_recall_sum = 0.0
    s4_correct = 0
    s4_ev_exact = 0

    intent_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "step1": 0, "step2": 0, "step4_asr": 0})

    for row in q_rows:
        qid = row["id"]
        intent = row["intent"]
        acceptable_decs = [d.strip() for d in row["acceptable_decisions"].split(" ; ") if d.strip()]
        intent_breakdown[intent]["total"] += 1

        # Step 1
        res1 = step1_agent.answer_question(row["question"])
        inc1_ok, inc1_miss = check_must_include(res1["reply"], row["must_include"])
        not1_ok, not1_viol = check_must_not_include(res1["reply"], row["must_not_include"])
        dec1_ok = res1["decision"] in acceptable_decs
        pass1 = inc1_ok and not1_ok and dec1_ok
        if pass1:
            s1_correct += 1
            intent_breakdown[intent]["step1"] += 1
        step1_results.append({
            "id": qid,
            "intent": intent,
            "question": row["question"],
            "reply": res1["reply"],
            "decision": res1["decision"],
            "passed": pass1,
            "missing_facts": inc1_miss,
            "forbidden_violations": not1_viol,
        })

        # Step 2 (Written question with RAG)
        res2 = step2_agent.answer_question(row["question"], is_voice=False)
        inc2_ok, inc2_miss = check_must_include(res2["reply"], row["must_include"])
        not2_ok, not2_viol = check_must_not_include(res2["reply"], row["must_not_include"])
        dec2_ok = res2["decision"] in acceptable_decs
        ev2_ok, rec2, prec2 = check_evidence(res2["retrieved_chunks"], row["gold_evidence"])
        pass2 = inc2_ok and not2_ok and dec2_ok
        if pass2:
            s2_correct += 1
            intent_breakdown[intent]["step2"] += 1
        if ev2_ok:
            s2_ev_exact += 1
        s2_recall_sum += rec2
        step2_results.append({
            "id": qid,
            "intent": intent,
            "language": row["language"],
            "question": row["question"],
            "reply": res2["reply"],
            "decision": res2["decision"],
            "expected_decision": row["expected_decision"],
            "retrieved_chunks": res2["retrieved_chunks"],
            "gold_evidence": [x.strip() for x in row["gold_evidence"].split(" ; ") if x.strip()],
            "passed_answer": pass2,
            "passed_retrieval": ev2_ok,
            "retrieval_recall": round(rec2, 3),
            "retrieval_precision": round(prec2, 3),
            "missing_facts": inc2_miss,
            "forbidden_violations": not2_viol,
        })

        # Step 4 (ASR transcript input + ear-formatted reply)
        res4 = step2_agent.answer_question(row["asr_text"], is_voice=True)
        inc4_ok, inc4_miss = check_must_include(res4["reply"], row["must_include"])
        not4_ok, not4_viol = check_must_not_include(res4["reply"], row["must_not_include"])
        dec4_ok = res4["decision"] in acceptable_decs
        ev4_ok, rec4, _ = check_evidence(res4["retrieved_chunks"], row["gold_evidence"])
        pass4 = inc4_ok and not4_ok and dec4_ok
        if pass4:
            s4_correct += 1
            intent_breakdown[intent]["step4_asr"] += 1
        if ev4_ok:
            s4_ev_exact += 1
        step4_asr_results.append({
            "id": qid,
            "intent": intent,
            "asr_text": row["asr_text"],
            "reply": res4["reply"],
            "decision": res4["decision"],
            "retrieved_chunks": res4["retrieved_chunks"],
            "passed_answer": pass4,
            "passed_retrieval": ev4_ok,
            "missing_facts": inc4_miss,
            "forbidden_violations": not4_viol,
        })

    # 2. Evaluate Step 3 Multi-Turn Conversations (23 conversations, 96 turns)
    conv_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in c_rows:
        conv_groups[row["conversation_id"]].append(row)

    step3_turn_results = []
    s3_turns_passed = 0
    s3_dec_passed = 0
    s3_tools_passed = 0
    s3_ev_passed = 0
    s3_convs_perfect = 0
    scenario_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {"turns": 0, "passed": 0})

    for cid, turns in sorted(conv_groups.items()):
        turns_sorted = sorted(turns, key=lambda r: int(r["turn"]))
        conv_agent = ZentroMartAgent(dataset_dir=base)
        all_turns_ok = True
        for r in turns_sorted:
            scen = r["scenario"]
            scenario_breakdown[scen]["turns"] += 1
            out = conv_agent.respond(r["customer_message"], is_voice=False)
            inc_ok, inc_miss = check_must_include(out["reply"], r["must_include"])
            not_ok, not_viol = check_must_not_include(out["reply"], r["must_not_include"])
            dec_ok = out["decision"] == r["expected_decision"]
            tools_ok = check_tool_calls(out["tool_calls"], r["expected_tool_calls"])
            ev_ok, rec_c, _ = check_evidence(out["retrieved_chunks"], r["gold_evidence"])

            turn_ok = inc_ok and not_ok and dec_ok and tools_ok and ev_ok
            if dec_ok:
                s3_dec_passed += 1
            if tools_ok:
                s3_tools_passed += 1
            if ev_ok:
                s3_ev_passed += 1
            if turn_ok:
                s3_turns_passed += 1
                scenario_breakdown[scen]["passed"] += 1
            else:
                all_turns_ok = False

            step3_turn_results.append({
                "conversation_id": cid,
                "turn": int(r["turn"]),
                "scenario": scen,
                "language": r["language"],
                "order_id": r["order_id"],
                "customer_message": r["customer_message"],
                "agent_reply": out["reply"],
                "decision": out["decision"],
                "expected_decision": r["expected_decision"],
                "tool_calls": out["tool_calls"],
                "expected_tool_calls": r["expected_tool_calls"],
                "retrieved_chunks": out["retrieved_chunks"],
                "gold_evidence": [x.strip() for x in r["gold_evidence"].split(" ; ") if x.strip()],
                "passed": turn_ok,
                "dec_ok": dec_ok,
                "tools_ok": tools_ok,
                "ev_ok": ev_ok,
                "missing_facts": inc_miss,
                "forbidden_violations": not_viol,
            })
        if all_turns_ok:
            s3_convs_perfect += 1

    total_q = len(q_rows)
    total_turns = len(c_rows)
    total_convs = len(conv_groups)

    summary = {
        "total_questions": total_q,
        "step1_prompt_only_correct": s1_correct,
        "step1_accuracy_pct": round(100.0 * s1_correct / total_q, 2),
        "step2_rag_correct": s2_correct,
        "step2_accuracy_pct": round(100.0 * s2_correct / total_q, 2),
        "step2_retrieval_exact_match": s2_ev_exact,
        "step2_retrieval_recall_pct": round(100.0 * s2_recall_sum / total_q, 2),
        "step4_asr_correct": s4_correct,
        "step4_asr_accuracy_pct": round(100.0 * s4_correct / total_q, 2),
        "step4_asr_retrieval_exact": s4_ev_exact,
        "total_conversations": total_convs,
        "total_conversation_turns": total_turns,
        "step3_turns_passed": s3_turns_passed,
        "step3_turn_accuracy_pct": round(100.0 * s3_turns_passed / total_turns, 2),
        "step3_decision_accuracy_pct": round(100.0 * s3_dec_passed / total_turns, 2),
        "step3_tool_call_accuracy_pct": round(100.0 * s3_tools_passed / total_turns, 2),
        "step3_evidence_accuracy_pct": round(100.0 * s3_ev_passed / total_turns, 2),
        "step3_conversations_perfect": s3_convs_perfect,
        "intent_breakdown": dict(intent_breakdown),
        "scenario_breakdown": dict(scenario_breakdown),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "step1_vs_step2_questions.json").write_text(
            json.dumps({"step1": step1_results, "step2": step2_results, "step4_asr": step4_asr_results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "step3_conversations_outputs.json").write_text(
            json.dumps(step3_turn_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "summary": summary,
        "step1": step1_results,
        "step2": step2_results,
        "step3": step3_turn_results,
        "step4_asr": step4_asr_results,
    }
