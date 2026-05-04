import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SIM_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in ['EmotionROI', 'ArtPhoto', 'EmoSet']:
        score = semantic_accuracy(dataset, results, model)
    else:
        raise ValueError(f"Unsupported dataset for eval: {dataset}")
    return score


def semantic_accuracy(dataset, results, model, thresholds=(0.7, 0.8, 0.9)):
    if dataset in ['EmotionROI', 'ArtPhoto', 'EmoSet']:
        prefix = "the emotion of this image is"
    else:
        prefix = ''

    acc = {th: [] for th in thresholds}

    for result in results:
        raw_pred = result['prediction'].strip().lower()
        raw_answer = str(result['answer']).strip().lower()

        if model == 'idefics-9b-instruct':
            pred_final = ""

            first_line = raw_pred.split('\n')[0].strip()
            if first_line.startswith(prefix):
                pred_final = first_line[len(prefix):].strip()

            if not pred_final:
                lines = raw_pred.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(prefix):
                        pred_final = line[len(prefix):].strip()
                        break

            if pred_final:
                answer_final = raw_answer[len(prefix):].strip() \
                    if raw_answer.startswith(prefix) \
                    else raw_answer

                embeddings = SIM_MODEL.encode([pred_final, answer_final])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                for th in thresholds:
                    acc[th].append(1 if similarity >= th else 0)
                continue

        raw_pred = raw_pred.replace('\n', ' ')

        valid_contents = []
        sentences = [s.strip() for s in raw_pred.split('.') if s.strip()]

        for sent in sentences:
            clean_sent = sent.split('!')[0].split('?')[0].strip()
            if clean_sent.startswith(prefix):
                content = clean_sent[len(prefix):].strip()
                if content:
                    valid_contents.append(content)

        unique_contents = list(set(valid_contents))
        if len(unique_contents) > 1:
            for th in thresholds:
                acc[th].append(0)
            continue

        pred_final = unique_contents[0] if valid_contents else ""

        if not valid_contents:
            if raw_pred.startswith(prefix):
                pred_final = raw_pred[len(prefix):].strip()
            else:
                pred_final = raw_pred

        answer_final = raw_answer[len(prefix):].strip() \
            if raw_answer.startswith(prefix) \
            else raw_answer

        embeddings = SIM_MODEL.encode([pred_final, answer_final])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        for th in thresholds:
            acc[th].append(1 if similarity >= th else 0)

    return {th: np.mean(acc[th]) for th in thresholds}
