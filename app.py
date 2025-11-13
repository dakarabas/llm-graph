import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile
import pathlib
import json

st.set_page_config(page_title="RAG + Agents + Prompting + Fine-tuning (+ skills)", page_icon="🕸️", layout="wide")
st.title("🕸️ RAG pipeline + agenti + prompting + fine-tuning aj.")

# --- Boční panel ---
st.sidebar.image("qest-logo-new.png", use_container_width=True)
physics = st.sidebar.checkbox("Fyzika (táhni uzly myší)", False)
st.sidebar.caption("Zapni/vypni, jestli mají uzly po puštění „dojíždět“.")

# --- Uzly ---
nodes = [
    # RAG pipeline
    "Chunking", "Embeddings", "Indexing", "VectorDB", "Ranking", "RAG", "LLM", "OpenAI",
    # Framework a orchestrátor
    "LangChain", "LangGraph",
    # Agenti a související uzly
    "Planner", "Executor", "Evaluator", "Memory", "ToolUse",
    # Ladění a prompting
    "Fine-tuning", "Prompt engineering", "Chain-of-Thought", "Zero-Shot", "Few-Shot",
    # === Nové skills z tvého seznamu ===
    "Prompt templates", "Sampling (temperature/top-p)", "Guardrails",
    "Human-in-the-loop", "LangSmith (Eval/Tracing)", "Model routing", "Caching",
]

# --- Vztahy (směrované hrany) ---
edges = [
    # RAG pipeline
    ("Chunking", "Embeddings"),
    ("Embeddings", "Indexing"),
    ("Indexing", "VectorDB"),
    ("VectorDB", "RAG"),
    ("Ranking", "RAG"),
    ("RAG", "LLM"),
    ("LLM", "OpenAI"),

    # Orchestrace
    ("LangChain", "LangGraph"),
    ("LangChain", "RAG"),
    ("LangChain", "LLM"),

    # Graf řízení
    ("LangGraph", "Planner"),
    ("LangGraph", "Executor"),
    ("LangGraph", "Evaluator"),
    ("LangGraph", "Memory"),

    # Agenti používají nástroje/RAG
    ("Planner", "ToolUse"),
    ("Planner", "RAG"),
    ("Executor", "ToolUse"),
    ("Executor", "RAG"),
    ("Evaluator", "LangGraph"),
    ("Memory", "RAG"),
    ("Memory", "LangGraph"),

    # Fine-tuning & prompting
    ("Fine-tuning", "LLM"),
    ("Prompt engineering", "Chain-of-Thought"),
    ("Prompt engineering", "Zero-Shot"),
    ("Prompt engineering", "Few-Shot"),
    ("Prompt engineering", "RAG"),
    ("Prompt engineering", "LLM"),

    # === Nové skills: napojení ===
    ("Prompt engineering", "Prompt templates"),
    ("Prompt engineering", "Sampling (temperature/top-p)"),
    ("Prompt templates", "LLM"),
    ("Sampling (temperature/top-p)", "LLM"),

    ("Guardrails", "LLM"),            # filtry na vstupu/výstupu
    ("Guardrails", "Evaluator"),      # validace/eval pravidla

    ("Human-in-the-loop", "Evaluator"),

    ("LangSmith (Eval/Tracing)", "Evaluator"),
    ("LangSmith (Eval/Tracing)", "LangChain"),
    ("LangSmith (Eval/Tracing)", "LangGraph"),

    ("LangGraph", "Model routing"),
    ("Model routing", "LLM"),
    ("Model routing", "OpenAI"),

    ("Caching", "RAG"),
    ("Caching", "LLM"),
]

# --- Tooltipy ---
titles = {
    "Chunking": "Dělení dokumentů na pasáže pro lepší retrieval.",
    "Embeddings": "Vektorové reprezentace textu pro podobnostní vyhledávání.",
    "Indexing": "Index nad embeddingy (např. HNSW/IVF).",
    "VectorDB": "Úložiště vektorů + nearest-neighbor search.",
    "Ranking": "Re-ranking pasáží (např. cross-encoder).",
    "RAG": "Retrieval-Augmented Generation: dohledá kontext a předá ho LLM.",
    "LLM": "Velký jazykový model – generace odpovědí.",
    "OpenAI": "Příklad poskytovatele LLM/API.",
    "LangChain": "Framework pro chains/agents, paměť a nástroje.",
    "LangGraph": "Stavový graf/orchestrátor: řízení toku, větvení, retry.",
    "Planner": "Plánuje kroky (plan-and-execute).",
    "Executor": "Provádí kroky/nástroje dle plánu.",
    "Evaluator": "Hodnotí kvalitu/validuje (guardrails, evaly).",
    "Memory": "Paměť (dlouhodobá/konverzační/vektorová).",
    "ToolUse": "Volání nástrojů/API (search, DB, kód...).",
    "Fine-tuning": "Učení/ladění modelu na vlastních datech.",
    "Prompt engineering": "Tvorba promptů a šablon pro řízení LLM.",
    "Chain-of-Thought": "Technika, která vede k postupnému uvažování.",
    "Zero-Shot": "Bez příkladů – model generuje rovnou.",
    "Few-Shot": "Pár příkladů v promptu pro lepší přesnost.",
    # Nové
    "Prompt templates": "Šablony promptů (Jinja/YAML/parametrizace).",
    "Sampling (temperature/top-p)": "Parametry kreativity a rozmanitosti.",
    "Guardrails": "Bezpečnost/validace výstupu (policy, filtry, schémata).",
    "Human-in-the-loop": "Člověk schvaluje/koriguje kroky či odpovědi.",
    "LangSmith (Eval/Tracing)": "Tracing, evaluace a experimenty s LLM pipeline.",
    "Model routing": "Dynamická volba modelu podle úlohy/nákladů.",
    "Caching": "Ukládání výsledků pro nižší latenci a cenu.",
}

# --- Barvy (skupiny) ---
palette = {
    # RAG pipeline
    "Chunking": "#e6e6ff", "Embeddings": "#e6e6ff", "Indexing": "#e6e6ff",
    "VectorDB": "#e6e6ff", "Ranking": "#e6e6ff", "RAG": "#b3ffb3",
    # LLM/Provider
    "LLM": "#ffd699", "OpenAI": "#ffd699",
    # Framework / orchestrátor
    "LangChain": "#99c2ff", "LangGraph": "#99c2ff",
    # Agenti + paměť + nástroje
    "Planner": "#ffe6f2", "Executor": "#ffe6f2", "Evaluator": "#ffe6f2",
    "Memory": "#e8d1ff", "ToolUse": "#d9d9d9",
    # Fine-tuning & Prompting
    "Fine-tuning": "#fff0b3", "Prompt engineering": "#fff0b3",
    "Chain-of-Thought": "#fff0b3", "Zero-Shot": "#fff0b3", "Few-Shot": "#fff0b3",
    # Nové skills
    "Prompt templates": "#fff0b3",
    "Sampling (temperature/top-p)": "#fff0b3",
    "Guardrails": "#ffdfdf",
    "Human-in-the-loop": "#ffdfdf",
    "LangSmith (Eval/Tracing)": "#c2f0f0",
    "Model routing": "#c2f0f0",
    "Caching": "#c2f0f0",
}

# --- Postav graf (směrovaný) ---
G = nx.DiGraph()
for n in nodes:
    G.add_node(n, label=n, title=titles.get(n, f"Uzol {n}"))
for src, dst in edges:
    G.add_edge(src, dst)

# --- PyVis síť ---
net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222", directed=True)
net.barnes_hut()
net.from_nx(G)

# --- Vzhled uzlů ---
for node in net.nodes:
    name = node["label"]
    node["shape"] = "dot"
    node["size"] = 22
    node["borderWidth"] = 2
    node["color"] = {"border": "#222", "background": palette.get(name, "#d9d9d9")}

# --- Nastavení (čistý JSON) ---
options = {
    "physics": {"enabled": bool(physics), "stabilization": {"iterations": 240}},
    "nodes": {"font": {"size": 18}},
    "edges": {"smooth": False, "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}}},
    "interaction": {"hover": True}
}
net.set_options(json.dumps(options))

# --- Render ---
tmp_dir = tempfile.gettempdir()
html_path = str(pathlib.Path(tmp_dir) / "rag_agents_prompting_skills_graph.html")
net.save_graph(html_path)
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=920, scrolling=False)