"""Retrieval-augmented explanations: a small language model words the reason under one recommendation.

The decision model has already ranked the paper. Here we retrieve the evidence behind it: [1] the paper,
[2] the saved paper closest to it, then its graph facts. The model reads only [1] and says in one sentence what
the paper does; the app states how it links to the saved paper. Earlier rounds gave the model more: it then
miscounted co-citations, reversed who cites whom, and described the saved paper as the recommended one.
The model's sentence is checked before anyone sees it."""
import os
import re
import threading
import time
from pathlib import Path

SYSTEM = ("Say in one sentence what the paper does, for a reader deciding whether to read it. "
          "Use only its abstract. End the sentence with [1].")
EXAMPLE_EVIDENCE = ("[1] Recommended: Neural Collaborative Filtering (2017). We present a neural network architecture "
                    "to model latent features of users and items, replacing the inner product of matrix factorization.")
EXAMPLE_ANSWER = "It models users and items with a neural network instead of the inner product of matrix factorization [1]."
DEFAULT_MODEL = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
# Words any reason may use without the evidence containing them.
COMMON = set("""about above after again also alongside among another approach based because before being both build
builds building cite cited cites citing close closely could does each extend extends extending from further have
help helps including into introduces itself like more most much other papers paper propose proposes provides read
reader reading reads recommended related relevant same saved shares similar some such than that their them then
there these they this those through together using uses very what when where which while with within work works
would your you""".split())


def describe(paper, words=45):
    abstract = " ".join((paper.get("abstract") or "").split()[:words])
    return f"{paper['title']} ({paper.get('year') or 'n.d.'}). {abstract}".strip()


def passages(engine, seeds, pid, vectors=None):
    """The numbered evidence for one recommendation and the sentence that states its graph facts.
    vectors are SPECTER rows in engine order, used to pick the saved paper closest to the recommendation."""
    witnesses = engine.co_citation_witnesses(seeds).get(pid, [])
    graph = engine.graph

    def closeness(seed):  # a direct citation, then shared citing papers, then semantic similarity
        cosine = float(vectors[engine.pos[seed]] @ vectors[engine.pos[pid]]) if vectors is not None else 0.0
        return (graph.has_edge(seed, pid) or graph.has_edge(pid, seed),
                sum(w["seed_id"] == seed for w in witnesses), cosine)

    saved = max(sorted(seeds), key=closeness)
    evidence = [f"Recommended: {describe(engine.by_id[pid])}", f"Saved by the reader: {describe(engine.by_id[saved])}"]
    facts = [f"Closest saved paper: {engine.by_id[saved]['title']} [2]"]

    def add(line, said):
        evidence.append(line)
        facts.append(f"{said} [{len(evidence)}]")
    witness = next((w for w in witnesses if w["seed_id"] == saved), None)
    if witness:
        title = engine.by_id[witness["paper_id"]]["title"]
        add(f"{title} cites both [1] and [2].", f"{title} cites it together with your saved paper")
    if graph.has_edge(saved, pid):
        add("[2] cites [1].", "your saved paper cites it")
    if graph.has_edge(pid, saved):
        add("[1] cites [2].", "it cites your saved paper")
    if len(witnesses) > 1:
        add(f"{len(witnesses)} papers cite [1] together with a saved paper.",
            f"{len(witnesses)} papers cite it together with a saved paper")
    return evidence, "; ".join(facts) + "."


def prompt(evidence):
    """Only [1], the recommended paper, goes to the model; the app states everything else."""
    text = f"[1] {evidence[0]}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": EXAMPLE_EVIDENCE},
            {"role": "assistant", "content": EXAMPLE_ANSWER}, {"role": "user", "content": text}]


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def words(text):
    """Lower-case words of four letters or more, cut to six letters so that support, supports and supported match."""
    return [w[:6] for w in re.findall(r"[a-z]{4,}", text.lower())]


def check(sentence, source, title):
    """Whether the model's sentence about one paper may be shown: it cites [1] and nothing else, every number in
    it appears in the source, and it says at least three things beyond the title (else it only echoes it).
    novel_words is the share of its content words absent from the source, a rough proxy for invented content
    that we report but do not gate on."""
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
    numbers = re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", " ", sentence))
    common = {w[:6] for w in COMMON}
    content = [w for w in words(sentence) if w not in common]
    known, named = set(words(source)), set(words(title))
    result = dict(valid_citations=bool(cited) and set(cited) == {1},
                  cites_recommended=1 in cited,
                  beyond_title=len({w for w in content if w in known} - named) >= 3,
                  grounded_numbers=all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", source) for n in numbers),
                  novel_words=sum(w not in known for w in content) / len(content) if content else 0.0)
    result["shown"] = result["valid_citations"] and result["grounded_numbers"] and result["beyond_title"]
    return result


class Explainer:
    """Wraps a chat function (messages -> text). load() builds one from a GGUF file with llama.cpp."""

    def __init__(self, chat, name):
        self.chat, self.name, self.lock = chat, name, threading.Lock()

    @classmethod
    def load(cls, path, threads=8, max_tokens=60):
        from llama_cpp import Llama  # optional dependency: uv sync --extra slm
        # n_threads_batch defaults to every logical CPU; on 8 cores with hyper-threading, 16 prompt-reading
        # threads contend with each other and with the server, so both phases use the same count.
        llm = Llama(str(path), n_ctx=2048, n_threads=threads, n_threads_batch=threads, verbose=False)

        def chat(messages):
            reply = llm.create_chat_completion(messages, max_tokens=max_tokens, temperature=0)
            return reply["choices"][0]["message"]["content"]
        return cls(chat, Path(path).name)

    def explain(self, evidence, fact, title):
        """The model's sentence followed by the graph facts; only the facts when the sentence fails the check."""
        started = time.perf_counter()
        with self.lock:  # one llama.cpp context serves one request at a time
            sentence = first_sentence(self.chat(prompt(evidence)))
        checks = check(sentence, evidence[0], title)
        summary = sentence if checks["shown"] else None
        return dict(text=f"{summary} {fact}" if summary else fact, summary=summary, draft=sentence, fact=fact, checks=checks, evidence=evidence, model=self.name,
                    seconds=time.perf_counter() - started)


def default_model_path():
    """The GGUF file named by RESEARCHLENS_SLM, else the default model in the Hugging Face cache, else None."""
    if os.environ.get("RESEARCHLENS_SLM"):
        return Path(os.environ["RESEARCHLENS_SLM"])
    repo, name = DEFAULT_MODEL
    cache = Path.home() / f".cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots"
    found = sorted(cache.glob(f"*/{name}"))
    return found[0] if found else None
