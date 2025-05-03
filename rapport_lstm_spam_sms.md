
# 📩 Application LSTM pour la détection de SMS Spam

Ce rapport détaille la conception, l'entraînement, le déploiement et le test d’une application basée sur un modèle **LSTM** pour **classifier les messages texte (SMS)** comme **Spam** ou **Non-Spam (Ham)**. L’application est implémentée avec **Streamlit** pour une utilisation interactive.

---

## 🎯 Objectif du projet

Détecter automatiquement les **messages indésirables** (spam) dans des SMS à l’aide d’un réseau **LSTM (Long Short-Term Memory)** entraîné sur le dataset **SMS Spam Collection**. Le modèle apprend à distinguer les messages légitimes de ceux qui sont suspects ou frauduleux.

---

## 🧱 Architecture de l'application

- 📦 Dataset : `SMS Spam Collection Dataset` (UCI).
- 🔠 Traitement : tokenisation + padding.
- 🧠 Modèle : `Embedding` → `LSTM` → `Dense`.
- 🧪 Entraînement avec EarlyStopping.
- 💾 Sauvegarde : `spam_lstm_model.h5`.
- 🖥️ Interface utilisateur avec Streamlit.

---

## 1️⃣ Chargement et préparation des données

```python
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', header=None, names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
```

- La colonne `label` contient les classes (`ham`, `spam`).
- La colonne `message` contient le contenu du SMS.

---

## 2️⃣ Prétraitement des données

```python
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['message'])

X = tokenizer.texts_to_sequences(df['message'])
X = pad_sequences(X, maxlen=100)
y = df['label'].values
```

- **Tokenisation** : chaque mot est remplacé par son indice.
- **Padding** : toutes les séquences sont mises à la même longueur (`maxlen=100`).
- **X** : séquences de mots ; **y** : labels binaires.

---

## 3️⃣ Création du modèle LSTM

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
```

- `Embedding` : encode les mots sous forme vectorielle.
- `LSTM` : capture les dépendances dans la séquence.
- `Dense` : sortie binaire (sigmoïde).

---

## 4️⃣ Compilation et entraînement

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(patience=2, restore_best_weights=True)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])
```

- **EarlyStopping** arrête l'entraînement si la validation ne s'améliore pas.
- Le modèle est **entraîné une seule fois**, puis sauvegardé avec `model.save()`.

---

## 5️⃣ Interface Streamlit

```python
st.text_area("Message texte", "Congratulations! You've won a free ticket to Bahamas.")
```

- L'utilisateur saisit un SMS.
- Le message est encodé, passé au modèle, et la prédiction est affichée :

```python
label = "Spam" if pred > 0.5 else "Non-Spam"
```

---

## ✅ Exemple de sortie

```
Message : "You have won a $1000 gift card. Click here to claim!"
Résultat : Spam 📛
Score de confiance : 0.9923
```

---

## 🧪 Résultats

- **Accuracy sur jeu test** : entre 96 et 98 % selon le split.
- Bonne détection des modèles classiques de spam.
- Erreurs possibles sur les messages ambigus ou très courts.

---

## 💬 Exemples de messages pour les tests

### ✅ Non-Spam

- `Hey, are we still on for dinner tonight at 8?`
- `Don’t forget to pick up the groceries on your way back.`
- `Can you call me when you finish your meeting?`
- `Happy birthday! Hope you have a great day 🎉`
- `Let’s meet at the library to study for the exam.`

### 📛 Spam

- `You have won a $1,000 Walmart gift card! Click here to claim now!`
- `Congratulations! Your number was selected for a free cruise to the Bahamas.`
- `URGENT! You have 1 hour to confirm your prize: www.fakeurl.com/win`
- `Get cheap meds now without a prescription! Visit www.scamrx.com`
- `This is not a scam! Click now to receive your cash bonus.`

---

## 🎓 Conclusion

Ce projet montre comment une architecture **LSTM simple** peut être efficacement utilisée pour la détection de **spam dans les SMS**. Grâce à Streamlit, l’utilisateur peut interagir en temps réel avec le modèle. Ce cas d’usage pédagogique démontre les forces et limites de LSTM dans une tâche NLP classique, tout en respectant l’objectif de l’exercice.

