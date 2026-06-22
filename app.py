"""
AI Learning OS — Streamlit UI
Three pages: Chat | Queue | Monitoring
"""

import json
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from ai_learning_os.agent import agent
from ai_learning_os.config import get_config
from ai_learning_os.monitoring import (
    export_thumbsup_as_eval_candidates,
    load_logs,
    log_interaction,
    update_feedback,
)
from ai_learning_os.tools import browse_queue, get_progress

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Learning OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = get_config()
client = OpenAI()


def _show_ingest_error(error: str) -> None:
    """Display a user-friendly error for add-to-queue failures."""
    if any(k in error for k in ("rate-limiting", "429", "Too Many Requests", "IPBlocked", "RequestBlocked")):
        st.warning(
            "**YouTube is rate-limiting requests from your IP.** "
            "Try again in 10–15 minutes, or connect a VPN and retry. "
            "Your existing queue items are unaffected.",
            icon="🚫",
        )
    else:
        st.error(error)

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []          # display messages [{role, content, tool_calls?, log_id?}]
if "thread" not in st.session_state:
    st.session_state.thread = []            # full OpenAI message thread (includes tool results)
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "current_resource" not in st.session_state:
    st.session_state.current_resource = None  # {"title": ..., "url": ..., "topic": ...}
if "pending_learn" not in st.session_state:
    st.session_state.pending_learn = None     # title of resource to auto-load on Chat page

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Learning OS")
    st.caption("Personal knowledge management + learning companion")
    st.divider()
    _pages = ["Chat", "Queue", "Monitoring"]
    # Only sync radio state when navigation was triggered programmatically (not by user click)
    if st.session_state.get("_nav_pending"):
        st.session_state.nav_radio = st.session_state.page
        st.session_state._nav_pending = False
    page = st.radio("Navigate", _pages, key="nav_radio")
    st.session_state.page = page
    st.divider()

    progress = get_progress(None, cfg)
    st.metric("KB resources", progress["kb_size"])
    st.metric("Queue depth", progress["queue_depth"])
    st.metric("Concepts mastered", progress["concepts_count"])


# ══════════════════════════════════════════════════════════════════════════════
# Page: CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Chat":
    hcol1, hcol2 = st.columns([8, 1])
    hcol1.header("Chat with your Learning OS")
    if hcol2.button("🗑 Clear", help="Clear conversation history"):
        st.session_state.messages = []
        st.session_state.thread = []
        st.session_state.current_resource = None
        st.session_state.pending_learn = None
        st.rerun()

    # Auto-load a resource that was queued from the Queue page
    if st.session_state.pending_learn:
        title = st.session_state.pending_learn
        st.session_state.pending_learn = None
        load_prompt = f"Load '{title}' and walk me through the key concepts."
        with st.spinner(f"Loading '{title}'..."):
            result_learn = agent(load_prompt, cfg=cfg, client=client)
        log_id = log_interaction(
            logs_file=cfg.logs_file,
            session_id=st.session_state.session_id,
            user_message=load_prompt,
            tool_calls=result_learn.tool_calls,
            answer=result_learn.answer,
            latency_ms=result_learn.latency_ms,
        )
        st.session_state.messages = [
            {"role": "user", "content": load_prompt},
            {
                "role": "assistant",
                "content": result_learn.answer,
                "tool_calls": result_learn.tool_calls,
                "log_id": log_id,
                "feedback": None,
            },
        ]
        st.session_state.thread = result_learn.thread
        st.rerun()

    if st.session_state.current_resource:
        r = st.session_state.current_resource
        st.info(
            f"**Currently studying:** [{r['title']}]({r['url']}) — {r['topic']}  \n"
            f"Ask questions, explore concepts, or say **\"I'm done\"** to add it to your knowledge base.",
            icon="📖",
        )

    # Render prior messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Tool call activity (collapsed by default)
            if msg.get("tool_calls"):
                with st.expander(f"🔧 {len(msg['tool_calls'])} tool call(s)", expanded=False):
                    for tc in msg["tool_calls"]:
                        st.code(f"{tc['name']}({json.dumps(tc['args'], indent=2)})", language="json")

            # Feedback buttons on assistant messages
            if msg["role"] == "assistant" and msg.get("log_id"):
                col1, col2, col3 = st.columns([1, 1, 10])
                current_fb = msg.get("feedback")
                with col1:
                    if st.button("👍", key=f"up_{i}", help="Good response",
                                 type="primary" if current_fb == "up" else "secondary"):
                        update_feedback(cfg.logs_file, msg["log_id"], "up")
                        st.session_state.messages[i]["feedback"] = "up"
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{i}", help="Bad response",
                                 type="primary" if current_fb == "down" else "secondary"):
                        update_feedback(cfg.logs_file, msg["log_id"], "down")
                        st.session_state.messages[i]["feedback"] = "down"
                        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about your learning..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent — pass full thread so tool results (e.g. loaded transcripts) stay in context
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent(prompt, cfg=cfg, client=client, history=st.session_state.thread)

            st.markdown(result.answer)

            if result.tool_calls:
                with st.expander(f"🔧 {len(result.tool_calls)} tool call(s)", expanded=False):
                    for tc in result.tool_calls:
                        st.code(f"{tc['name']}({json.dumps(tc['args'], indent=2)})", language="json")

            # Log interaction
            log_id = log_interaction(
                logs_file=cfg.logs_file,
                session_id=st.session_state.session_id,
                user_message=prompt,
                tool_calls=result.tool_calls,
                answer=result.answer,
                latency_ms=result.latency_ms,
            )

            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_new", help="Good response"):
                    update_feedback(cfg.logs_file, log_id, "up")
                    st.session_state.messages[-1]["feedback"] = "up" if st.session_state.messages else None
            with col2:
                if st.button("👎", key=f"down_new", help="Bad response"):
                    update_feedback(cfg.logs_file, log_id, "down")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "tool_calls": result.tool_calls,
            "log_id": log_id,
            "feedback": None,
        })
        st.session_state.thread = result.thread  # persist full thread for next turn
        # Clear resource banner once the user has consolidated it to the KB
        if any(tc["name"] == "consolidate_to_kb" for tc in result.tool_calls):
            st.session_state.current_resource = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: QUEUE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Queue":
    st.header("📚 Learning Queue")
    st.caption("Resources staged for study — not yet in your knowledge base")

    topic_filter = st.text_input("Filter by topic", placeholder="e.g. LLMs, RAG, Agents")
    result = browse_queue(topic_filter or None, cfg)

    if result["count"] == 0:
        st.info(result.get("message", "Your queue is empty. Ask the chat to add a YouTube link or article."))
    else:
        st.metric("Resources in queue", result["count"])
        st.divider()
        for item in result["items"]:
            with st.expander(f"**{item['title']}** — {item['topic']} · {item['difficulty']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Source type", item["source_type"])
                col2.metric("Word count", f"{item['word_count']:,}")
                col3.metric("Added", item["added_at"][:10])

                if item.get("url"):
                    st.caption(f"🔗 {item['url']}")

                if item.get("preview"):
                    st.markdown("**Content preview:**")
                    st.text(item["preview"] + " …")

                if st.button(f"▶ Start learning this", key=f"learn_{item['title']}"):
                    # Clear prior conversation, set context, navigate — agent runs on Chat page
                    st.session_state.messages = []
                    st.session_state.thread = []
                    st.session_state.current_resource = {
                        "title": item["title"],
                        "url": item.get("url", ""),
                        "topic": item["topic"],
                    }
                    st.session_state.pending_learn = item["title"]
                    st.session_state.page = "Chat"
                    st.session_state._nav_pending = True
                    st.rerun()

    st.divider()
    st.subheader("Add to queue")
    url_tab, file_tab = st.tabs(["🔗 URL", "📄 File upload (.md / image)"])

    with url_tab:
        with st.form("add_url_form"):
            source_url = st.text_input(
                "YouTube URL or article URL",
                placeholder="https://www.youtube.com/watch?v=... or https://..."
            )
            col1, col2 = st.columns(2)
            topic_url = col1.text_input("Topic", placeholder="e.g. RAG", key="topic_url")
            difficulty_url = col2.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="diff_url")
            submitted_url = st.form_submit_button("Add URL to queue")

        if submitted_url and source_url:
            with st.spinner("Fetching content..."):
                from ai_learning_os.tools import add_to_queue as _add
                result_add = _add(source_url, topic_url or "general", difficulty_url, cfg, client)
            if result_add.get("error"):
                _show_ingest_error(result_add["error"])
            elif result_add.get("already_queued"):
                st.warning(result_add["message"])
            else:
                st.success(f"Added **{result_add['title']}** ({result_add['word_count']:,} words)")
                st.rerun()  # refresh queue list only on success

    with file_tab:
        st.caption("Upload a `.md` file (e.g. marker output from a PDF) or an image (`.png`, `.jpg`, `.webp`).")
        with st.form("add_file_form"):
            uploaded = st.file_uploader(
                "Choose a file",
                type=["md", "png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            topic_file = col1.text_input("Topic", placeholder="e.g. LLMs", key="topic_file")
            difficulty_file = col2.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="diff_file")
            submitted_file = st.form_submit_button("Add file to queue")

        if submitted_file and uploaded:
            # Save uploaded file to data/uploads/ so tools.py can read it by path
            uploads_dir = cfg.data_dir / "uploads"
            uploads_dir.mkdir(exist_ok=True)
            save_path = uploads_dir / uploaded.name
            save_path.write_bytes(uploaded.getvalue())

            with st.spinner(f"Processing {uploaded.name}..."):
                from ai_learning_os.tools import add_to_queue as _add
                result_add = _add(str(save_path), topic_file or "general", difficulty_file, cfg, client)
            if result_add.get("error"):
                _show_ingest_error(result_add["error"])
            elif result_add.get("already_queued"):
                st.warning(result_add["message"])
            else:
                st.success(f"Added **{result_add['title']}** ({result_add['word_count']:,} words)")
                st.rerun()  # refresh queue list only on success


# ══════════════════════════════════════════════════════════════════════════════
# Page: MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Monitoring":
    st.header("📊 Monitoring Dashboard")

    logs = load_logs(cfg.logs_file)

    if not logs:
        st.info("No interactions logged yet. Start chatting!")
    else:
        import pandas as pd

        df = pd.DataFrame(logs)

        # ── KPI row ────────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total interactions", len(df))
        col2.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")
        thumbs_up = (df["feedback"] == "up").sum()
        thumbs_down = (df["feedback"] == "down").sum()
        rated = thumbs_up + thumbs_down
        score = f"{thumbs_up/rated*100:.0f}%" if rated > 0 else "—"
        col3.metric("Feedback score", score, f"{rated} rated")
        col4.metric("Sessions", df["session_id"].nunique())

        st.divider()

        # ── Tool call frequency ────────────────────────────────────────────────
        st.subheader("Tool call frequency")
        tool_counts: dict[str, int] = {}
        for row in logs:
            for tc in row.get("tool_calls", []):
                tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1

        if tool_counts:
            tc_df = pd.DataFrame(
                list(tool_counts.items()), columns=["Tool", "Count"]
            ).sort_values("Count", ascending=True)
            st.bar_chart(tc_df.set_index("Tool"))
        else:
            st.caption("No tool calls yet.")

        st.divider()

        # ── Recent interactions ────────────────────────────────────────────────
        st.subheader("Recent interactions")
        display_df = df[["timestamp", "user_message", "latency_ms", "feedback"]].copy()
        display_df["user_message"] = display_df["user_message"].str[:80] + "..."
        display_df.columns = ["Timestamp", "User message", "Latency (ms)", "Feedback"]
        st.dataframe(display_df.tail(20).iloc[::-1], use_container_width=True)

        st.divider()

        # ── Export eval candidates ─────────────────────────────────────────────
        st.subheader("Export thumbs-up interactions as eval candidates")
        candidates = export_thumbsup_as_eval_candidates(cfg.logs_file)
        st.caption(f"{len(candidates)} thumbs-up interaction(s) available for export.")

        if st.button("Export to evals/ground_truth_candidates.json"):
            out = Path("evals/ground_truth_candidates.json")
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
            st.success(f"Exported {len(candidates)} candidates to {out}")

        if candidates:
            with st.expander("Preview candidates"):
                for c in candidates[:3]:
                    st.json(c)
