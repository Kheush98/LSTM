
# 📩 Application de Détection de Spam avec LSTM - Rapport Explicatif Complet

Ce document explique étape par étape la conception, l'entraînement et le fonctionnement d'une **application de détection de spam dans des messages texte (SMS)** à l'aide d'un **réseau de neurones LSTM (Long Short-Term Memory)**. Il est conçu pour être compris même par ceux qui débutent dans le deep learning.

**Streamlit** est un framework open-source en Python qui permet de créer des applications web interactives particulièrement adapté pour visualiser des modèles de machine learning et interagir avec eux de manière simple et efficace.

---

## 🧠 Partie 1 : Comprendre le LSTM

### 🔁 Qu’est-ce qu’un LSTM ?

Un **LSTM** est un type particulier de réseau de neurones appelé **Réseau de Neurones Récurrent (RNN)**. Contrairement aux réseaux classiques qui traitent les données comme indépendantes, les RNN (et donc les LSTM) sont conçus pour **gérer des séquences de données** — comme des textes ou des séries temporelles.

### 🧱 Structure d’un LSTM

Un LSTM contient plusieurs **portes** internes qui contrôlent comment l'information est conservée ou oubliée au fil de la séquence. Ces portes permettent :

- **d’oublier** ce qui n’est plus utile (`forget gate`)
- **de mémoriser** les nouvelles informations importantes (`input gate`)
- **de produire** une sortie à chaque étape (`output gate`)

Cela en fait un excellent outil pour **analyser des textes**, où l’ordre des mots est essentiel.

---

## 🎯 Objectif du projet

> Développer un modèle de machine learning capable de **prédire si un SMS est un spam ou non**, en utilisant une architecture LSTM.

---

## 🧱 Partie 2 : Architecture de l’application

### 📦 Dataset utilisé

Nous utilisons le célèbre **SMS Spam Collection Dataset**. Il contient :

- 5 572 messages SMS
- Deux catégories :
  - `ham` : message normal
  - `spam` : message frauduleux ou commercial

---

## ⚙️ Étapes de traitement et d’entraînement

### 1. Chargement et nettoyage des données

```python
df = pd.read_csv(".../sms.tsv", sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
```

- On lit les messages et on convertit les labels (`ham`, `spam`) en valeurs numériques (`0`, `1`).

---

### 2. Prétraitement du texte

```python
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['message'])
X = tokenizer.texts_to_sequences(df['message'])
X = pad_sequences(X, maxlen=100)
```

- Chaque mot est transformé en un **indice numérique**.
- On tronque ou complète les messages pour qu’ils aient tous la **même longueur** (`100` mots maximum).

---

### 3. Séparation des données

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

- **80 % des données** servent à entraîner le modèle.
- **20 % des données** servent à le tester.

---

### 4. Architecture du modèle LSTM

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
```

#### Détails :
- `Embedding` : transforme chaque mot en un **vecteur dense de 128 dimensions**.
- `LSTM` : mémorise le contexte du message.
- `Dense` : produit une probabilité entre 0 et 1 → plus proche de 1 = spam.

---

### 5. Entraînement

```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(...)])
```

- Le modèle apprend **en observant des exemples**.
- On utilise `EarlyStopping` pour arrêter automatiquement si le modèle cesse de s’améliorer.

---

### 6. Sauvegarde du modèle

```python
model.save("spam_lstm_model.h5")
```

- Une fois entraîné, le modèle est enregistré pour **ne pas avoir à le réentraîner** à chaque lancement.

---

## 🖥️ Partie 3 : Interface avec Streamlit

```python
user_input = st.text_area("Message texte")
```

- L’utilisateur saisit un message.
- Celui-ci est converti, puis analysé :

```python
prediction = model.predict(padded_sequence)
```

- Le résultat est affiché :
  - `Spam 📛` si score > 0.5
  - `Non-Spam ✅` sinon

---

## ✅ Exemples de messages à tester

### 🟢 Non-Spam
- Hey, are we still on for dinner tonight?
- I’m at the library, call me when you’re done.
- Happy birthday, bro! 🎉
- Don’t forget our meeting at 3pm.

### 🔴 Spam
- You’ve won $1,000! Claim now: click.me/gift
- Congratulations! You have been selected for a free cruise.
- Urgent! Confirm your reward: www.fakeoffer.com
- This is not a scam. Act fast and reply YES now.

---

## 🧪 Résultats attendus

- Le modèle atteint environ **96 à 98 % de précision**.
- Il détecte bien les schémas typiques du spam.
- Il peut faire des erreurs sur des messages **trop courts** ou **mal orthographiés**.

---

## 🎓 Conclusion

Cette application démontre comment une **architecture LSTM simple** peut résoudre un problème de classification de texte, ici la détection de SMS indésirables. Grâce à Streamlit, l'outil est accessible même à des utilisateurs non techniques, avec un modèle réutilisable, stable, et relativement performant.

Ce projet est idéal pour **comprendre concrètement le fonctionnement des LSTM**, tout en découvrant un cas d'usage réel du NLP (traitement automatique du langage naturel).
