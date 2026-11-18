"""Weakly supervised review polarity; ratings are proxies, not sentiment gold."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LABELS=['critical','mixed','positive']
def rating_label(rating):
    return 'critical' if rating<=4 else 'positive' if rating>=7 else 'mixed'

def deduplicate(reviews):
    seen=set(); result=[]
    for r in reviews:
        key=(r['paper_id'],re.sub(r'\s+',' ',r['text']).strip().lower())
        if key in seen: continue
        seen.add(key); result.append(r)
    return result

def train_model(reviews):
    train=[r for r in deduplicate(reviews) if r['split']=='train']
    if len(set(rating_label(r['rating']) for r in train))<2: return None
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=40000),
                        LogisticRegression(C=1.0,class_weight='balanced',max_iter=1000,random_state=42))
    texts=[r['text'] for r in train]; labels=[rating_label(r['rating']) for r in train]
    try:
        model.fit(texts,labels)
    except ValueError as exc:
        if not any(t in str(exc) for t in ('empty vocabulary','After pruning','max_df corresponds')): raise
        model.set_params(tfidfvectorizer__min_df=1)
        try: model.fit(texts,labels)
        except ValueError as fallback:
            if 'empty vocabulary' in str(fallback): return None
            raise
    return model

def predict(model,text):
    if model is None: return None
    probs=model.predict_proba([text])[0]
    return {'label':str(model.classes_[probs.argmax()]),'probabilities':{str(k):float(v) for k,v in zip(model.classes_,probs)},
            'method':'TF-IDF + logistic regression; trained on PeerRead train split; target is rating-derived polarity.'}
