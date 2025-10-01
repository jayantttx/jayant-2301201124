import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import re

df = pd.read_csv(r"C:\Users\singh\OneDrive\Desktop\Assignments\NLP\Comment-Sentiment.csv")

if 'Comment' not in df.columns or 'Sentiment' not in df.columns:
    raise ValueError("CSV must contain 'Comment' and 'Sentiment' columns.")
df = df.dropna(subset=['Comment', 'Sentiment'])

label_encoder = LabelEncoder()
df['Sentiment_num'] = label_encoder.fit_transform(df['Sentiment'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r'\b\w+\b', text)

def tokenize_text(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('#', '')
    tokens = safe_word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['Comment_proc'] = df['Comment'].apply(tokenize_text)

comments_train, comments_test, sentiments_train, sentiments_test = train_test_split(
    df['Comment_proc'], df['Sentiment_num'], test_size=0.2, random_state=42, stratify=df['Sentiment_num']
)

tfidf = TfidfVectorizer(stop_words=None, max_features=1000)
features_train = tfidf.fit_transform(comments_train)
features_test = tfidf.transform(comments_test)

classifier = MultinomialNB()
classifier.fit(features_train, sentiments_train)

predictions = classifier.predict(features_test)
print("Classification Report:")
print(classification_report(sentiments_test, predictions, target_names=label_encoder.classes_))

cm = confusion_matrix(sentiments_test, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(sentiments_test, predictions)
print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

total_comments = len(df)
processed_nonempty = df['Comment_proc'].astype(str).str.strip().ne('').sum()
empty_count = total_comments - processed_nonempty
print(f"Comments processed (non-empty after tokenization): {processed_nonempty}/{total_comments} "
      f"({processed_nonempty/total_comments*100:.2f}%)")
if empty_count:
    print(f"Comments empty after tokenization: {empty_count}. Showing up to 10 examples:")
    print(df.loc[df['Comment_proc'].astype(str).str.strip() == '', 'Comment'].head(10).tolist())

analyzed_in_test = comments_test.astype(str).str.strip().ne('').sum()
missing_in_test = len(comments_test) - analyzed_in_test
print(f"Test set size: {len(comments_test)}. Non-empty (analyzed) in test: {analyzed_in_test}. Missing in test: {missing_in_test}")

if len(predictions) != len(comments_test):
    print(f"Warning: predictions ({len(predictions)}) != test samples ({len(comments_test)})")
else:
    print("All test samples have corresponding predictions.")

cm = confusion_matrix(sentiments_test, predictions, labels=sorted(np.unique(sentiments_test)))
true_counts = cm.sum(axis=1)
diag = cm.diagonal()
with np.errstate(divide='ignore', invalid='ignore'):
    per_class_acc = np.where(true_counts > 0, diag / true_counts, np.nan)

class_indices = sorted(np.unique(sentiments_test))
class_names = label_encoder.inverse_transform(class_indices)
print("\nPer-class accuracy:")
for name, acc_pc in zip(class_names, per_class_acc):
    acc_str = f"{acc_pc:.4f}" if not np.isnan(acc_pc) else "n/a"
    print(f"  {name}: {acc_str}")

plt.figure(figsize=(6,3))
vals = [processed_nonempty, empty_count]
labels = ['Processed', 'Empty after tokenization']
colors = ['tab:green', 'tab:red']
plt.bar(labels, vals, color=colors)
plt.ylabel('Count')
plt.title('Comments processed vs empty after tokenization')
for i, v in enumerate(vals):
    plt.text(i, v + max(vals)*0.01, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()

total_comments = len(df)
processed_nonempty = df['Comment_proc'].astype(str).str.strip().ne('').sum()
all_comments_processed = (processed_nonempty == total_comments)

test_nonempty_count = comments_test.astype(str).str.strip().ne('').sum()
all_test_analyzed = (test_nonempty_count == len(comments_test))

predictions_match = (len(predictions) == len(comments_test))

print("\n=== Analysis Coverage Summary ===")
print(f"Total comments in dataset: {total_comments}")
print(f"Non-empty after tokenization: {processed_nonempty} ({processed_nonempty/total_comments*100:.2f}%)")
print(f"All comments processed (non-empty)? {'YES' if all_comments_processed else 'NO'}")
print(f"Test set size: {len(comments_test)}")
print(f"Non-empty in test set: {test_nonempty_count} ({test_nonempty_count/len(comments_test)*100:.2f}%)")
print(f"All test samples analyzed (non-empty)? {'YES' if all_test_analyzed else 'NO'}")
print(f"Predictions count matches test samples? {'YES' if predictions_match else 'NO'}")

if not all_comments_processed:
    print("\nExamples of comments empty after tokenization (up to 10):")
    empties = df.loc[df['Comment_proc'].astype(str).str.strip() == '', 'Comment'].head(10)
    for i, ex in enumerate(empties, 1):
        print(f"{i}. {ex!r}")

if not all_test_analyzed:
    print("\nExamples of test comments empty after tokenization (up to 10):")
    empties_test = comments_test[comments_test.astype(str).str.strip() == ''].head(10)
    for i, ex in enumerate(empties_test, 1):
        print(f"{i}. {ex!r}")

if not predictions_match:
    print("\nWarning: number of predictions does not match number of test samples.")

features_all = tfidf.transform(df['Comment_proc'])
preds_all = classifier.predict(features_all)

acc_all = accuracy_score(df['Sentiment_num'], preds_all)
print(f"\nOverall accuracy on ALL comments: {acc_all:.4f} ({acc_all*100:.2f}%)")

print("\nClassification report (ALL comments):")
print(classification_report(df['Sentiment_num'], preds_all, target_names=label_encoder.classes_))

cm_all = confusion_matrix(df['Sentiment_num'], preds_all)
plt.figure(figsize=(8,6))
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — ALL Comments')
plt.tight_layout()
plt.show()

total_all = len(df)
processed_all = df['Comment_proc'].astype(str).str.strip().ne('').sum()
mis_idx = np.where(preds_all != df['Sentiment_num'])[0]
print(f"\nTotal comments: {total_all}")
print(f"Processed (non-empty after tokenization): {processed_all} ({processed_all/total_all*100:.2f}%)")
print(f"Misclassified: {len(mis_idx)} ({len(mis_idx)/total_all*100:.2f}%)")

if len(mis_idx) > 0:
    print("\nExamples of misclassified comments (up to 10):")
    for idx in mis_idx[:10]:
        true_label = label_encoder.inverse_transform([int(df.at[df.index[idx],'Sentiment_num'])])[0]
        pred_label = label_encoder.inverse_transform([int(preds_all[idx])])[0]
        print(f"\nComment (orig): {df.at[df.index[idx],'Comment']!r}")
        print(f"Processed: {df.at[df.index[idx],'Comment_proc']!r}")
        print(f"True: {true_label}  --> Predicted: {pred_label}")

try:
    import emoji
    def demojize_text(s):
        return emoji.demojize(s, language='en')
except Exception:
    def demojize_text(s):
        return s

def mark_negation(text):
    return re.sub(r'\b(not|no|never)\s+([a-zA-Z]+)', lambda m: f"{m.group(1)}_{m.group(2)}", text)

df['Comment_proc'] = df['Comment'].astype(str).apply(lambda t: mark_negation(demojize_text(t)))
df['Comment_proc'] = df['Comment_proc'].apply(tokenize_text)

from sklearn.utils import resample

df_bal = []
max_count = df['Sentiment_num'].value_counts().max()
for lbl, group in df.groupby('Sentiment_num'):
    if len(group) < max_count:
        up = resample(group, replace=True, n_samples=max_count, random_state=42)
        df_bal.append(up)
    else:
        df_bal.append(group)
df_balanced = pd.concat(df_bal).sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after upsampling:")
print(df_balanced['Sentiment_num'].value_counts())

comments_train, comments_test, sentiments_train, sentiments_test = train_test_split(
    df_balanced['Comment_proc'], df_balanced['Sentiment_num'],
    test_size=0.2, random_state=42, stratify=df_balanced['Sentiment_num']
)

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=5000, sublinear_tf=True)
tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=2000, sublinear_tf=True)

Xw_train = tfidf_word.fit_transform(comments_train)
Xc_train = tfidf_char.fit_transform(comments_train)
X_train = hstack([Xw_train, Xc_train])

Xw_test = tfidf_word.transform(comments_test)
Xc_test = tfidf_char.transform(comments_test)
X_test = hstack([Xw_test, Xc_test])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

nb = MultinomialNB()
nb.fit(X_train, sentiments_train)
pred_nb = nb.predict(X_test)
print("NB accuracy (balanced training):", accuracy_score(sentiments_test, pred_nb))

lr = LogisticRegression(solver='saga', max_iter=3000, random_state=42, class_weight='balanced')
param_grid = {'C': [0.01, 0.1, 1.0, 5.0, 10.0]}
grid = GridSearchCV(lr, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=0)
grid.fit(X_train, sentiments_train)
best_lr = grid.best_estimator_
pred_lr = best_lr.predict(X_test)
print("Best LR params:", grid.best_params_)
print("LR accuracy (balanced training):", accuracy_score(sentiments_test, pred_lr))

if accuracy_score(sentiments_test, pred_lr) >= accuracy_score(sentiments_test, pred_nb):
    classifier = best_lr
    predictions = pred_lr
    print("Selected model: LogisticRegression")
else:
    classifier = nb
    predictions = pred_nb
    print("Selected model: MultinomialNB")

print("\nClassification Report (selected model):")
print(classification_report(sentiments_test, predictions, target_names=label_encoder.classes_))

cm = confusion_matrix(sentiments_test, predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (balanced + enhanced features)')
plt.show()

mis_idx = np.where(predictions != np.array(sentiments_test))[0]
if len(mis_idx) > 0:
    print(f"Total misclassified in test: {len(mis_idx)}. Sample misclassified examples:")
    for i in mis_idx[:10]:
        orig = comments_test.iloc[i]
        true = label_encoder.inverse_transform([sentiments_test.iloc[i]])[0]
        pred = label_encoder.inverse_transform([predictions[i]])[0]
        print(f"- True:{true} Pred:{pred}  -->  {orig[:120]!r}")
else:
    print("No misclassifications on test set.")

import joblib
joblib.dump({'tfidf_word': tfidf_word, 'tfidf_char': tfidf_char, 'model': classifier, 'label_encoder': label_encoder}, 'model_enhanced.joblib')
