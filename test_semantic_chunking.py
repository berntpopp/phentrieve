#!/usr/bin/env python3
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pysbd  # For sentence segmentation
import re  # For regular expression splitting
import spacy  # For dependency parsing

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s [%(funcName)s]: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "FremyCompany/BioLORD-2023-M"

# --- spaCy Model Loading ---
NLP_MODELS = {}


def get_spacy_model(lang_code: str):
    if lang_code not in NLP_MODELS:
        model_name_spacy = ""
        if lang_code == "en":
            model_name_spacy = "en_core_web_sm"
        elif lang_code == "de":
            model_name_spacy = "de_core_news_sm"
        if model_name_spacy:
            try:
                NLP_MODELS[lang_code] = spacy.load(model_name_spacy)
                logger.info(
                    f"Loaded spaCy model '{model_name_spacy}' for language '{lang_code}'."
                )
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name_spacy}' for lang '{lang_code}' not found. "
                    f"Download with: python -m spacy download {model_name_spacy}. "
                    f"Dependency parsing will be skipped for '{lang_code}'."
                )
                NLP_MODELS[lang_code] = None
        else:
            logger.warning(
                f"No spaCy model configured for lang '{lang_code}'. Dependency parsing will be skipped."
            )
            NLP_MODELS[lang_code] = None
    return NLP_MODELS.get(lang_code)


# --- Text Pre-processing ---
def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_into_paragraphs(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    normalized_text = normalize_line_endings(text)
    paragraphs = re.split(r"\n\s*\n+", normalized_text)
    return [p.strip() for p in paragraphs if p.strip()]


def clean_internal_newlines_and_extra_spaces(text_chunk: str) -> str:
    if not text_chunk:
        return ""
    cleaned = re.sub(r"\s*\n\s*", " ", text_chunk)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


# --- Sentence Segmentation ---
def segment_into_sentences(text: str, lang: str = "en") -> list[str]:
    if not text.strip():
        return []
    try:
        segmenter = pysbd.Segmenter(language=lang, clean=False)
        sentences = segmenter.segment(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(
            f"pysbd error for lang '{lang}' on text '{text[:50]}...': {e}. Fallback splitting."
        )
        processed_lines = []
        for line in text.split("\n"):
            stripped_line = line.strip()
            if stripped_line:
                if not stripped_line.endswith((".", "?", "!")):
                    processed_lines.append(stripped_line + ".")
                else:
                    processed_lines.append(stripped_line)
        text_for_fallback_splitting = " ".join(processed_lines)
        sentences_raw = re.findall(r"[^.?!]+(?:[.?!]|$)", text_for_fallback_splitting)
        return [s.strip() for s in sentences_raw if s.strip()]


# --- Negation and Normality Detection ---
NEGATION_CUES = {
    "en": [
        "no ",
        "not ",
        "n't",
        "denies ",
        "denied ",
        "without ",
        "negative for ",
        "rules out ",
        "ruled out ",
        "absence of ",
        "lack of ",
        "free of ",
        "never had ",
        "cannot be identified ",
    ],
    "de": [
        "kein ",
        "keine ",
        "keinen ",
        "keiner ",
        "keines ",
        "nicht ",
        "ohne ",
        "Abwesenheit von ",
        "Fehlen von ",
        "Mangel an ",
        "negativ für ",
        "schließt aus ",
        "ausgeschlossen ",
        "frei von ",
        "niemals gehabt ",
        "kann nicht identifiziert werden ",
    ],
}
NORMALITY_CUES = {
    "en": [
        "normal",
        "unremarkable",
        "within normal limits",
        "wnl",
        "clear ",
        "no evidence of ",
        "nev",
        "resolved",
        "absent",
        " unremarkable",
        "no acute process",
    ],
    "de": [
        "normal",
        "unauffällig",
        "o.B.",
        "oB",
        "ohne Befund",
        "im Normbereich",
        "reizlos",
        "kein Anhalt für ",
        "nicht vorhanden",
        "aufgehoben",
        "physiologisch",
    ],
}
KEYWORD_WINDOW = 7


def detect_negation_normality_keyword(
    chunk: str, lang: str = "en"
) -> tuple[bool, bool, list[str], list[str]]:
    is_negated_flag, is_normal_flag = False, False
    negated_scopes, normal_scopes = [], []
    lower_chunk = chunk.lower()

    def is_cue_match(text_lower, cue_lower, index):
        start_ok = (index == 0) or (not text_lower[index - 1].isalnum())
        end_ok = (index + len(cue_lower) == len(text_lower)) or (
            not text_lower[index + len(cue_lower)].isalnum()
        )
        return start_ok and end_ok

    for cue_list, flag_var_name, scopes_list, type_str in [
        (NEGATION_CUES, "is_negated_flag", negated_scopes, "Negation"),
        (NORMALITY_CUES, "is_normal_flag", normal_scopes, "Normality"),
    ]:
        if type_str == "Normality" and is_negated_flag:
            continue

        cue_found_for_type = (
            False  # To break from outer loop once a cue of this type is found
        )
        for cue_orig in cue_list.get(lang, []):
            cue_lower = cue_orig.lower().strip()
            idx = 0
            while idx < len(lower_chunk):
                found_at = lower_chunk.find(cue_lower, idx)
                if found_at == -1:
                    break
                if is_cue_match(lower_chunk, cue_lower, found_at):
                    if flag_var_name == "is_negated_flag":
                        is_negated_flag = True
                    else:
                        is_normal_flag = True

                    scope_start_char = max(0, found_at - KEYWORD_WINDOW * 6)
                    scope_end_char = min(
                        len(chunk), found_at + len(cue_lower) + KEYWORD_WINDOW * 10
                    )
                    scope_text = chunk[scope_start_char:scope_end_char]
                    scopes_list.append(
                        f"'{chunk[found_at:found_at+len(cue_lower)]}' in context: '...{scope_text}...'"
                    )
                    cue_found_for_type = True
                    break  # Break from inner while loop (finding this specific cue)
                idx = found_at + 1  # Continue searching for this specific cue
            if cue_found_for_type:
                break  # Break from outer for cue in cue_list loop

    if is_negated_flag and is_normal_flag:
        is_normal_flag = False
    return is_negated_flag, is_normal_flag, negated_scopes, normal_scopes


def detect_negation_normality_dependency(
    chunk: str, lang: str = "en"
) -> tuple[bool, bool, list[str], list[str]]:
    nlp = get_spacy_model(lang)
    if not nlp:
        return False, False, [], []
    doc = nlp(chunk)
    is_negated_flag, is_normal_flag = False, False
    negated_concepts, normal_concepts = [], []
    handled_token_indices = set()

    for cue_list, target_concepts_list, flag_setter_name in [
        (NEGATION_CUES.get(lang, []), negated_concepts, "is_negated_flag"),
        (NORMALITY_CUES.get(lang, []), normal_concepts, "is_normal_flag"),
    ]:
        for cue_phrase in cue_list:
            cue_phrase_stripped = cue_phrase.strip()
            if " " not in cue_phrase_stripped:
                continue
            for match in re.finditer(
                r"\b" + re.escape(cue_phrase_stripped) + r"\b", chunk, re.IGNORECASE
            ):
                start_char, end_char = match.span()
                for token_idx_in_doc in range(len(doc)):  # Check all tokens in doc
                    token = doc[token_idx_in_doc]
                    if token.idx >= start_char and token.idx < end_char:
                        handled_token_indices.add(token.i)
                concept_after_cue = ""
                if end_char < len(chunk):
                    remaining_text = chunk[end_char:]
                    concept_match = re.match(r"\s*([\w\s\-]+)", remaining_text)
                    if concept_match:
                        concept_after_cue = concept_match.group(1).strip()[:50]
                target_concepts_list.append(
                    f"'{cue_phrase_stripped}' -> '{concept_after_cue}...'"
                )
                if flag_setter_name == "is_negated_flag":
                    is_negated_flag = True
                else:
                    is_normal_flag = True

    for token in doc:
        if token.i in handled_token_indices:
            continue
        token_lemma_lower = token.lemma_.lower()

        if token.dep_ == "neg" or (
            token_lemma_lower in NEGATION_CUES.get(lang, [])
            and not any(
                " " in c.strip()
                for c in NEGATION_CUES.get(lang, [])
                if c.lower().strip() == token_lemma_lower
            )
        ):  # Ensure it's a single-word cue
            is_negated_flag = True
            head = token.head
            neg_target_text = head.lemma_
            obj_children = [
                child.lemma_
                for child in head.children
                if child.dep_ in ("dobj", "obj", "acomp", "attr")
            ]
            if obj_children:
                neg_target_text += " " + " ".join(obj_children)
            negated_concepts.append(f"'{token.text}' negates '{neg_target_text}' (dep)")
            handled_token_indices.add(token.i)
            handled_token_indices.add(head.i)
            for child in head.children:
                if child.dep_ in ("dobj", "obj", "acomp", "attr"):
                    handled_token_indices.add(child.i)

        if (
            token_lemma_lower in NORMALITY_CUES.get(lang, [])
            and token.i not in handled_token_indices
        ):
            if not any(
                " " in c.strip()
                for c in NORMALITY_CUES.get(lang, [])
                if c.lower().strip() == token_lemma_lower
            ):  # Single word cue
                is_normal_flag = True
                norm_target_text = ""
                if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                    norm_target_text = f"{token.text} {token.head.lemma_}"
                elif token.dep_ in ("attr", "acomp") and token.head.lemma_ == "be":
                    subjects = [
                        child.lemma_
                        for child in token.head.children
                        if "subj" in child.dep_
                    ]
                    norm_target_text = f"{' '.join(subjects) if subjects else 'Something'} is {token.lemma_}"
                else:
                    norm_target_text = f"{token.lemma_} (general context)"
                normal_concepts.append(norm_target_text)
                handled_token_indices.add(token.i)
                if token.head:
                    handled_token_indices.add(token.head.i)

    if is_negated_flag and is_normal_flag:
        is_normal_flag = False
    return (
        is_negated_flag,
        is_normal_flag,
        list(set(negated_concepts)),
        list(set(normal_concepts)),
    )


# --- Semantic Chunker ---
def semantic_chunker(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 1,
    max_chunk_sentences: int = 10,
    language: str = "en",
) -> list[str]:
    if not text.strip():
        return []
    sentences = segment_into_sentences(text, lang=language)
    if not sentences:
        return [text.strip()] if text.strip() else []
    logger.debug(
        f"SemChunker: Segmented into {len(sentences)} sentences for '{language}'. Input text: '{text[:100]}...'"
    )

    if len(sentences) <= min_chunk_sentences:
        logger.debug(
            f"SemChunker: Text has {len(sentences)} sentence(s) <= min ({min_chunk_sentences}). Single chunk."
        )
        return [" ".join(sentences)] if sentences else []

    try:
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=32)
    except Exception as e:
        logger.error(f"SemChunker: Error encoding: {e}")
        return [" ".join(s for s in sentences if s)] if sentences else []
    if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
        logger.warning(
            "SemChunker: Embeddings are invalid. Returning original sentences as chunks."
        )
        return sentences

    similarities = []
    if len(embeddings) > 1:
        for i in range(len(embeddings) - 1):
            emb1, emb2 = embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1)
            similarities.append(cosine_similarity(emb1, emb2)[0, 0])
            logger.debug(f"SemChunker: Sim S{i+1}-S{i+2}: {similarities[-1]:.4f}")
    elif sentences:
        logger.debug("SemChunker: Only one sentence after segmentation, returning it.")
        return [" ".join(sentences)]
    else:
        return []

    chunks, current_chunk_sentences = [], []
    for i, sentence in enumerate(sentences):
        current_chunk_sentences.append(sentence)
        is_last_sentence_in_doc = i == len(sentences) - 1

        split_now = False
        if is_last_sentence_in_doc:
            split_now = True
            logger.debug(
                f"SemChunker Split Trigger: End of document at S{i+1} ({len(current_chunk_sentences)} sents in chunk)"
            )
        elif len(current_chunk_sentences) >= max_chunk_sentences:
            split_now = True
            logger.debug(
                f"SemChunker Split Trigger: Max chunk size {max_chunk_sentences} reached with S{i+1}"
            )
        elif i < len(similarities):
            if (
                similarities[i] < similarity_threshold
                and len(current_chunk_sentences) >= min_chunk_sentences
            ):
                split_now = True
                logger.debug(
                    f"SemChunker Split Trigger: Similarity drop after S{i+1} (Sim to S{i+2}: {similarities[i]:.4f} < {similarity_threshold}, current chunk size {len(current_chunk_sentences)})"
                )

        if split_now:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        logger.debug(
            f"SemChunker: Appending final leftover chunk of {len(current_chunk_sentences)} sents."
        )

    chunks = [c for c in chunks if c.strip()]
    if len(chunks) > 1:
        last_chunk_sents_list = segment_into_sentences(chunks[-1], lang=language)
        if 0 < len(last_chunk_sents_list) < min_chunk_sentences:
            logger.debug(
                f"SemChunker: Last chunk too small ({len(last_chunk_sents_list)} sents vs min {min_chunk_sentences}), merging."
            )
            last_content = chunks.pop()
            if chunks:
                chunks[-1] = (chunks[-1] + " " + last_content).strip()
            else:
                chunks.append(last_content)
    return chunks


# --- Fine-grained Splitter (Revised) ---
def fine_grained_split(text_chunk: str) -> list[str]:
    if not text_chunk.strip():
        return []

    # Temporarily replace decimal points to avoid splitting numbers like "2.5"
    text_chunk_marked = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL_POINT>\2", text_chunk)

    SPLIT_MARKER = "<PHRASE_SPLIT_Phentrieve>"

    # Stage 1: Insert SPLIT_MARKER after sentence enders (.?!)
    # Ensure punctuation is kept with the left part.
    # Handle cases where sentence ender is followed by optional whitespace then potentially a list marker or end of string.
    processed_text = re.sub(
        r"([.?!])(\s*(?:\n\s*(?:\d+\.|\*|\-|\•|\([a-zA-Z0-9]+\))\s*)?|$)",
        rf"\1{SPLIT_MARKER}",
        text_chunk_marked,
    )

    # Stage 2: Insert SPLIT_MARKER after commas and semicolons (common clause separators)
    # Avoid splitting on colons for now, as they often introduce related descriptors.
    processed_text = re.sub(r"([,;])(\s+|$)", rf"\1{SPLIT_MARKER}", processed_text)

    sub_chunks_raw = processed_text.split(SPLIT_MARKER)

    final_sub_chunks = []
    for sub_p_raw in sub_chunks_raw:
        if sub_p_raw is None:
            continue
        cleaned_sub_p = sub_p_raw.replace("<DECIMAL_POINT>", ".").strip()
        if cleaned_sub_p:
            final_sub_chunks.append(cleaned_sub_p)

    if not final_sub_chunks and text_chunk.strip():
        return [text_chunk.replace("<DECIMAL_POINT>", ".").strip()]

    return final_sub_chunks


def main():
    logger.info(f"Loading sentence model: {MODEL_NAME}")
    sbert_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    logger.info("Sentence model loaded.")

    german_text_with_paragraphs = (
        "Patientenanamnese:\n"
        "Der Patient ist kleinwüchsig und zeigt eine deutliche Entwicklungsverzögerung, insbesondere im sprachlichen Bereich.\n"
        "Motorische Fähigkeiten wie Laufen und Greifen sind altersentsprechend. Kein Tremor.\n\n"
        "Befunde:\n"
        "Feinmotorik ist ungeschickt. Zusätzlich wurden faziale Dysmorphien, darunter eine Hypertelorismus und eine fliehende Stirn, beobachtet.\n"
        "Keine kardialen Probleme festgestellt; Herz ist unauffällig. Blutdruck normal. Lunge o.B.\n\n"
        "Weitere Untersuchungen:\n"
        "Nierenfunktion normal. Ultraschall ohne Befund. Kein Anhalt für Zystennieren oder maligne Erkrankung.\n"
        "Labor: Leukozyten normal, CRP nicht erhöht."
    )

    english_text_with_paragraphs_and_lists = (
        "Presenting Complaint:\n"
        "The patient exhibits significant global developmental delay. Speech is particularly affected, showing no spontaneous utterances.\n"
        "Previous medical history is negative for major illnesses. Vision is not impaired.\n\n"
        "Physical Examination:\n"
        "1. Head: Normocephalic, no microcephaly noted.\n"
        "2. Eyes: PERRLA, fundi normal. Ruled out strabismus. No nystagmus.\n"
        "3. Chest: Lungs clear to auscultation. No wheezes or rales. Thorax is unremarkable.\n"
        "   Heart sounds are regular, S1 and S2 normal, no murmurs appreciated.\n\n"
        "Impression & Plan:\n"
        "Developmental delay, cause undetermined. Genetic testing was uninformative.\n"
        "Plan: Referral to neurology. Ruled out metabolic disorders. Patient denies any pain. Follow-up in 3 months."
    )

    german_short_sections = (
        "Allgemeinzustand: Reduziert.\n"
        "Ernährung: Mangelhaft.\n\n"
        "Kopf: Mikrozephalie.\n"
        "Augen: Kein Strabismus. Nystagmus nicht vorhanden.\n"
        "Ohren: Normal geformt. Gehör ist gut.\n\n"
        "Herz: Rhythmisch, keine pathologischen Geräusche. Frequenz normal.\n"
        "Lunge: Vesikulär, keine Rasselgeräusche. Keine Dyspnoe.\n"
        "Abdomen: Weich, keine Hepatosplenomegalie. Keine Druckschmerzhaftigkeit."
    )

    single_long_german_sentence_with_negations = (
        "Bei der Untersuchung zeigte sich kein Fieber, keine Tachykardie, aber eine leichte Dyspnoe bei Belastung, "
        "jedoch ohne Zyanose; die neurologische Untersuchung war unauffällig bis auf einen nicht vorhandenen Babinski-Reflex rechts, "
        "während Reflexe sonst seitengleich und mittellebhaft auslösbar waren und keine Ataxie bestand."
    )

    # New example to specifically test list-like structures and colons
    german_list_and_colon_example = (
        "Klinische Merkmale:\n"
        "- Faziale Auffälligkeiten: Epikanthus medialis, tief ansetzende Ohren, breiter Nasenrücken.\n"
        "- Skelett: Brachydaktylie Typ E; Skoliose, mäßig ausgeprägt.\n"
        "- Neurologie: Allgemeine Muskelhypotonie, keine Krampfanfälle in der Vorgeschichte.\n"
        "Zusammenfassung: Komplexes Syndrom mit multiplen Anomalien."
    )

    texts_to_analyze = {
        "German with Paragraphs & Neg/Norm": (german_text_with_paragraphs, "de"),
        "English with Paragraphs, Lists & Neg/Norm": (
            english_text_with_paragraphs_and_lists,
            "en",
        ),
        "German Short Sections & Neg/Norm": (german_short_sections, "de"),
        "Single Long German Sentence w/ Negations": (
            single_long_german_sentence_with_negations,
            "de",
        ),
        "German List and Colon Example": (german_list_and_colon_example, "de"),
    }

    similarity_threshold = 0.38  # Adjusted slightly, tune as needed
    min_sem_sentences = 1
    max_sem_sentences = 3  # Keep low to test paragraph + semantic interaction

    for name, (text, lang) in texts_to_analyze.items():
        print(f"\n\n{'='*70}")
        print(f"--- ANALYZING: {name} (Language: {lang}) ---")
        print(f"{'='*70}")
        print(f"Original Text:\n{text}\n")

        paragraphs = split_into_paragraphs(text)
        print(f"\n>>> Found {len(paragraphs)} PARAGRAPH chunks:")
        overall_final_sub_chunks_count = 0

        for p_idx, para_text_raw in enumerate(paragraphs):
            para_text = clean_internal_newlines_and_extra_spaces(para_text_raw)
            print(f"\n  Paragraph {p_idx+1} (Cleaned):\n  '{para_text}'")

            if not para_text:
                print(f"    >>> Paragraph {p_idx+1} is empty after cleaning. Skipping.")
                continue

            semantic_chunks = semantic_chunker(
                para_text,
                sbert_model,
                similarity_threshold,
                min_sem_sentences,
                max_sem_sentences,
                language=lang,
            )
            print(
                f"    >>> Found {len(semantic_chunks)} SEMANTIC sub-chunks in Paragraph {p_idx+1}:"
            )

            for i, sem_chunk_raw in enumerate(semantic_chunks):
                sem_chunk = clean_internal_newlines_and_extra_spaces(sem_chunk_raw)
                print(
                    f"\n      Semantic Sub-chunk {p_idx+1}.{i+1} (Cleaned): '{sem_chunk}'"
                )
                if not sem_chunk:
                    continue

                fine_grained_sub_chunks = fine_grained_split(sem_chunk)
                print(
                    f"        >>> Fine-grained parts from Semantic Sub-chunk {p_idx+1}.{i+1}: {len(fine_grained_sub_chunks)}"
                )

                for j, sub_chunk_raw_fg in enumerate(fine_grained_sub_chunks):
                    sub_chunk = clean_internal_newlines_and_extra_spaces(
                        sub_chunk_raw_fg
                    )
                    if not sub_chunk.strip():
                        continue
                    overall_final_sub_chunks_count += 1
                    print(f"          Part {p_idx+1}.{i+1}.{j+1}: '{sub_chunk}'")

                    (is_neg_key, is_norm_key, neg_scopes_key, norm_scopes_key) = (
                        detect_negation_normality_keyword(sub_chunk, lang)
                    )
                    print(
                        f"            Keyword Neg: {is_neg_key} {neg_scopes_key if neg_scopes_key else ''}"
                    )
                    print(
                        f"            Keyword Norm: {is_norm_key} {norm_scopes_key if norm_scopes_key else ''}"
                    )

                    nlp_model = get_spacy_model(lang)
                    final_status = "AFFIRMED/UNCERTAIN"
                    if nlp_model:
                        (
                            is_neg_dep,
                            is_norm_dep,
                            neg_concepts_dep,
                            norm_concepts_dep,
                        ) = detect_negation_normality_dependency(sub_chunk, lang)
                        print(
                            f"            Dependency Neg: {is_neg_dep} {neg_concepts_dep if neg_concepts_dep else ''}"
                        )
                        print(
                            f"            Dependency Norm: {is_norm_dep} {norm_concepts_dep if norm_concepts_dep else ''}"
                        )

                        if is_neg_dep:
                            final_status = "NEGATED"
                        elif is_norm_dep:
                            final_status = "NORMAL"
                        elif is_neg_key:
                            final_status = "NEGATED"
                        elif is_norm_key:
                            final_status = "NORMAL"
                    else:
                        if is_neg_key:
                            final_status = "NEGATED"
                        elif is_norm_key:
                            final_status = "NORMAL"

                    print(f"            ==> Combined Status: {final_status}")
                if fine_grained_sub_chunks:
                    print("        -------------------------")
            if semantic_chunks:
                print("    =========================")

        print(
            f"\n>>> TOTAL FINAL SUB-CHUNKS for '{name}': {overall_final_sub_chunks_count}"
        )


if __name__ == "__main__":
    main()
