"""ENSET-ChatBot : interface Streamlit d'un RAG local (Ollama + Chroma).

Lancer :  streamlit run app.py
"""
import streamlit as st

from rag import config
from rag.ingestion import index_pdf
from rag.pipeline import run_advanced, run_classic

st.set_page_config(page_title="ENSET-ChatBot", page_icon="📄", layout="centered")

MODE_ADVANCED = "RAG amélioré"
MODE_CLASSIC = "RAG classique"
MODE_COMPARE = "Comparaison classique vs amélioré"

# ---------------------- BARRE LATÉRALE ----------------------
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

st.sidebar.markdown("---")
st.sidebar.subheader("Paramètres")
model_name = st.sidebar.text_input("Modèle Ollama", config.DEFAULT_MODEL)
mode = st.sidebar.radio("Mode", [MODE_ADVANCED, MODE_CLASSIC, MODE_COMPARE])
chunk_size = st.sidebar.slider("Taille des chunks", 300, 2000, config.CHUNK_SIZE, 100)
top_k = st.sidebar.slider("Chunks envoyés au LLM (top_k)", 2, 8, config.TOP_K)
with st.sidebar.expander("Réglages du RAG amélioré"):
    k_search = st.slider("Chunks cherchés par requête (k)", 2, 8, config.K_SEARCH)
    n_variants = st.slider("Reformulations", 1, 4, config.N_VARIANTS)
    max_candidates = st.slider(
        "Candidats max à noter (plus = plus lent)", 4, 15, config.MAX_CANDIDATES
    )

st.title("ENSET-ChatBot : Local PDF-RAG with LangChain, Ollama, and Chroma")


# ---------------------- RESSOURCES (chargées une seule fois) ----------------------
@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@st.cache_resource
def get_llm(name: str):
    from langchain_community.llms import Ollama

    # temperature=0 : réponses stables et plus fidèles au document
    return Ollama(
        model=name,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=config.NUM_CTX,
    )


# ---------------------- ÉTAT DE SESSION ----------------------
for key, default in {
    "db": None,
    "messages": [],
    "processed_key": None,
    "n_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------- AFFICHAGE ----------------------
def render_result(res):
    st.markdown(f"**{res.title}**")
    st.markdown(res.answer)
    if res.sources:
        st.markdown("**Sources :**\n" + "\n".join(f"- {s}" for s in res.sources))
    st.caption(f"⏱ Temps total : {res.elapsed:.1f} s")
    with st.expander("Détails"):
        st.markdown(res.details)


def render_results(results):
    if len(results) == 1:
        render_result(results[0])
        return
    for col, res in zip(st.columns(len(results)), results):
        with col:
            render_result(res)


def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_results(msg["results"])


# ---------------------- INDEXATION DU PDF (une seule fois par fichier) ----------------------
if uploaded_file is not None:
    current_key = (uploaded_file.name, uploaded_file.size, chunk_size)
    if st.session_state.processed_key != current_key:
        try:
            with st.spinner("Processing PDF..."):
                new_db, n_chunks = index_pdf(
                    uploaded_file.getvalue(), uploaded_file.name,
                    get_embeddings(), chunk_size,
                )
        except Exception as exc:
            st.error(f"Impossible de traiter ce PDF : {exc}")
        else:
            if st.session_state.db is not None:
                try:
                    st.session_state.db.delete_collection()
                except Exception:
                    pass
            st.session_state.db = new_db
            st.session_state.n_chunks = n_chunks
            st.session_state.processed_key = current_key
            st.session_state.messages = []
            st.success(f"PDF traité : {n_chunks} chunks. Pose tes questions !")

# ---------------------- HISTORIQUE + QUESTION ----------------------
for message in st.session_state.messages:
    render_message(message)

query = st.chat_input("Ask a question about the PDF:")

if query:
    if st.session_state.db is None:
        st.warning("Upload d'abord un PDF dans la barre latérale.")
    else:
        user_msg = {"role": "user", "text": query}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)

        llm = get_llm(model_name)
        db = st.session_state.db
        try:
            with st.chat_message("assistant"):
                with st.spinner("Generating answer... (le RAG amélioré est plus lent)"):
                    results = []
                    if mode in (MODE_CLASSIC, MODE_COMPARE):
                        results.append(run_classic(query, db, llm, top_k))
                    if mode in (MODE_ADVANCED, MODE_COMPARE):
                        results.append(
                            run_advanced(
                                query, db, llm, top_k, k_search, n_variants, max_candidates
                            )
                        )
                render_results(results)
        except Exception as exc:
            st.session_state.messages.pop()  # on retire la question restée sans réponse
            st.error(
                f"Erreur pendant la génération : {exc}\n\n"
                f"Vérifie qu'Ollama tourne (`ollama list`) et que le modèle "
                f"« {model_name} » est installé (`ollama pull {model_name}`)."
            )
        else:
            st.session_state.messages.append({"role": "assistant", "results": results})
