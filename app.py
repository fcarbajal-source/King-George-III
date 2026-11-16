import re
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 280
MIN_WORDS = 15
MAX_WORDS = 120
MAX_REMEMBERED_QUESTIONS = 6

SYSTEM_INSTRUCTION = """
You are King George V (born 1865, reigned 6 May 1910 – 20 January 1936), King of the United Kingdom and the British Dominions, and Emperor of India.

AUDIENCE & VOICE:
- Speak directly to U.S. 10th-grade high school students (ages 15–16).
- Sound like King George V himself: dignified, dutiful, polite, plainspoken like a former Royal Navy officer, with early 20th-century British royal cadence (e.g., "My dear young friend," "It was My duty," "Our Empire").
- Keep sentences clear and understandable for a 10th-grade American reader. Briefly clarify British terms like "Peers" (nobles in the House of Lords) or "Dominions" (self-governing nations like Canada and Australia) when used.

MANDATED HISTORICAL ANCHORS (BE FACT-DENSE):
Anchor answers in verified dates, people, letters, statutes, speeches, and legal precedents:
- Accession & Coronation: Succeeded Edward VII on 6 May 1910; crowned 22 June 1911; Delhi Durbar Dec 1911.
- Constitutional Crisis & Parliament Act 1911 (1 & 2 Geo. 5. c. 13): Correspondence with Prime Minister H. H. Asquith (1910–1911) and royal diaries (18 May 1910) on agreeing to create new Peers if needed so the elected House of Commons could curb the House of Lords' veto.
- World War I (1914–1918) & U.S. Alliance: Letters to/from cousins Kaiser Wilhelm II and Tsar Nicholas II; Royal Proclamation of 17 July 1917 renaming the royal house from Saxe-Coburg and Gotha to Windsor; 1917 Letters Patent limiting "Prince/Princess" titles; April 1918 welcome letter to American Expeditionary Forces (AEF) soldiers ("fighting in the Old World the great battle for human freedom"); U.S. war declarations and diplomatic/Congressional records (1917–1919).
- Post-War Governance & Empire: Belfast speech opening the Northern Ireland Parliament (June 1921); appointing Britain's first Labour Prime Minister Ramsay MacDonald (1924); 1926 General Strike; Statute of Westminster 1931 (22 & 23 Geo. 5. c. 4) granting legislative equality to the Dominions; first Royal Christmas Broadcast on 25 December 1932 ("Through one of the marvels of modern Science..."), written with Rudyard Kipling; Silver Jubilee (1935).
- Also cite relevant judicial or constitutional decisions when asked about royal/parliamentary authority (e.g., UK constitutional conventions recorded in Cabinet/Parliamentary papers and later court rulings referencing the 1911 Act or 1917 Letters Patent).

MANDATORY SOURCES & CITATIONS:
Ground every response in primary and secondary sources—especially letters to and from King George V, official government documents, speeches, parliamentary debates, U.S. Congressional documents, court/constitutional records, and legislation.
Every response MUST end with a short citation naming the document/letter and at least one of these approved digital archives:
1. Library of Congress Digital Collections
2. HathiTrust Digital Library
3. The Royal Archives (Royal Collection Trust)
4. The National Archives (UK) – Discovery
5. UK Parliamentary Papers (ProQuest) & Hansard
6. The London Gazette
7. British Library Digitised Manuscripts & Sound Archive

STRICT LENGTH & ACCURACY RULES:
- Every reply MUST be between 15 words minimum and 120 words maximum (target 65–105 words), INCLUDING the final "Source:" citation.
- Pack in specific years, names, and primary-source facts. Do not invent facts or sources. If a question goes beyond 1936, state politely in character that your earthly reign ended in January 1936, while still citing an archive.
""".strip()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def enforce_word_limits(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    """Guarantees response stays strictly between 15 and 120 words."""
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) > max_words:
        # Preserve a trailing Source: line if possible
        source_match = re.search(r"(Source:\s*.+)$", cleaned)
        source_tail = source_match.group(1) if source_match else ""
        source_words = source_tail.split() if source_tail else []

        if source_tail and len(source_words) < max_words - 15:
            body_text = cleaned[: source_match.start()].strip()
            budget = max_words - len(source_words)
            trimmed_body = " ".join(body_text.split()[:budget])
            # Trim back to last sentence punctuation if present
            last_punct = max(trimmed_body.rfind("."), trimmed_body.rfind("!"), trimmed_body.rfind("?"))
            if last_punct > 20:
                trimmed_body = trimmed_body[: last_punct + 1]
            cleaned = f"{trimmed_body} {source_tail}".strip()
        else:
            trimmed = " ".join(words[:max_words])
            last_punct = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
            cleaned = trimmed[: last_punct + 1] if last_punct > 20 else trimmed + "..."

    elif len(words) < min_words:
        cleaned = (
            f"{cleaned} Duty to Crown and Parliament guided every step of My reign. "
            "Source: The Royal Archives (Royal Collection Trust)."
        ).strip()

    return cleaned


def build_limited_history(messages: list[dict], new_prompt: str) -> list[types.Content]:
    """
    Keeps ONLY the previous 6 user questions (and their 6 model answers)
    plus the current user question.
    """
    prior_turns = messages[-(MAX_REMEMBERED_QUESTIONS * 2) :]
    contents = []
    for msg in prior_turns:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt)],
        )
    )
    return contents


def ask_king_george_v(client: genai.Client, history_contents: list[types.Content]) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=history_contents,
        config=config,
    )
    raw_text = (response.text or "").strip()
    return enforce_word_limits(raw_text)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask King George V (1910–1936)",
    page_icon="👑",
    layout="centered",
)

st.title("👑 Speak with King George V")
st.caption(
    "An interactive primary-source history chatbot for U.S. 10th-grade students "
    f"• Powered by `{MODEL_ID}`"
)

# Load hidden API key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing `GEMINI_API_KEY`. Add it to `.streamlit/secrets.toml` locally "
        "or in Streamlit Community Cloud → App Settings → Secrets."
    )
    st.stop()

client = get_genai_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls & archive reference
with st.sidebar:
    st.header("Historical Guidelines")
    remembered_q_count = min(
        MAX_REMEMBERED_QUESTIONS,
        sum(1 for m in st.session_state.messages if m["role"] == "user"),
    )
    st.metric("Active Memory (Max 6 Questions)", f"{remembered_q_count} / {MAX_REMEMBERED_QUESTIONS}")
    st.write("**Reading Level:** U.S. 10th Grade")
    st.write("**Response Length:** 15–120 words")
    st.write(f"**Temperature:** {TEMPERATURE} | **Max Tokens:** {MAX_OUTPUT_TOKENS}")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try Asking:")
    sample_questions = [
        "Why did you change your family name to Windsor in 1917?",
        "What did your April 1918 letter say to American soldiers?",
        "How did you handle the Parliament Act of 1911 crisis?",
        "Why did you start the 1932 Christmas radio broadcast?",
        "What changed for Canada and Australia in the 1931 Statute of Westminster?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q

    st.divider()
    st.subheader("Referenced Digital Archives")
    st.markdown(
        "- Library of Congress Digital Collections\n"
        "- HathiTrust Digital Library\n"
        "- The Royal Archives (Royal Collection Trust)\n"
        "- The National Archives (UK) – Discovery\n"
        "- UK Parliamentary Papers & Hansard\n"
        "- The London Gazette\n"
        "- British Library Digitised Manuscripts & Sound Archive"
    )

# Render chat history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="👑"):
        st.write(
            "Good day, my young American friend. I am George the Fifth, King from 1910 to 1936. "
            "Ask Me of World War One, Our letters to your brave soldiers, or Parliament's laws, "
            "and I shall answer faithfully from official records. "
            "Source: The Royal Archives (Royal Collection Trust)."
        )

for msg in st.session_state.messages:
    avatar = "👑" if msg["role"] == "assistant" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Accept chat input or sidebar button prompt
prompt = st.chat_input("Ask King George V a historical question...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    with st.chat_message("user", avatar="🎓"):
        st.write(prompt)

    # Build history with ONLY the previous 6 questions + current prompt
    history_contents = build_limited_history(st.session_state.messages, prompt)

    with st.chat_message("assistant", avatar="👑"):
        with st.spinner("Consulting the Royal Archives..."):
            try:
                reply = ask_king_george_v(client, history_contents)
            except Exception as exc:
                st.error(f"Could not reach Gemini API: {exc}")
                st.stop()
        st.write(reply)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})
