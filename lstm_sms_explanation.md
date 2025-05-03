
# Explication ligne par ligne du script `sms_lstm_train_and_app.py`

Ce script utilise un modèle LSTM pour détecter si un message SMS est un spam ou non. Il intègre une interface interactive grâce à Streamlit.

---

## Paramètres globaux

```python
MAX_WORDS = 10000
MAX_LEN = 100
MODEL_PATH = "spam_lstm_model.h5"
```
- `MAX_WORDS` : nombre maximal de mots uniques utilisés dans la tokenisation.
- `MAX_LEN` : longueur maximale des séquences (les séquences plus courtes sont remplies, les plus longues sont tronquées).
- `MODEL_PATH` : chemin où le modèle LSTM sera enregistré ou chargé.

---

## Chargement et préparation des données

```python
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                     sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df
```

- Cette fonction télécharge un dataset de SMS contenant deux colonnes : l’étiquette (`ham` ou `spam`) et le message.
- Elle transforme les étiquettes en valeurs numériques : `ham` → 0, `spam` → 1.
- La fonction est mise en cache grâce à `@st.cache_data` pour éviter de recharger à chaque exécution.

```python
df = load_and_prepare_data()
```
- Appelle la fonction pour charger les données.

---

## Tokenisation et préparation des données d'entraînement

```python
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['message'])
```
- Crée un tokenizer limité aux `MAX_WORDS` mots les plus fréquents, puis l’entraîne sur les messages.

```python
X = tokenizer.texts_to_sequences(df['message'])
X = pad_sequences(X, maxlen=MAX_LEN)
```
- Convertit les messages en séquences d’entiers.
- Applique un padding à chaque séquence pour qu’elles aient toutes `MAX_LEN` éléments.

```python
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Sépare les données en un ensemble d'entraînement (80%) et un ensemble de test (20%).

---

## Chargement ou entraînement du modèle

```python
@st.cache_resource
def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
```
- Vérifie si un modèle a déjà été entraîné et sauvegardé ; si oui, il le charge.

```python
    else:
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])
```
- Sinon, crée un modèle séquentiel avec :
  - Une couche `Embedding` pour transformer les entiers en vecteurs de taille 128.
  - Une couche LSTM avec 64 unités, dropout de 20% pour les entrées et récurrentes.
  - Une couche de sortie `Dense` à 1 neurone avec activation sigmoïde pour la classification binaire.

```python
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- Compile le modèle avec l’optimiseur `adam` et la perte binaire croisée.

```python
        early_stop = EarlyStopping(patience=2, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])
```
- Utilise `EarlyStopping` pour arrêter l’entraînement si la performance de validation stagne plus de 2 époques.
- Entraîne le modèle sur 10 époques avec 32 exemples par batch, en réservant 20% pour la validation.

```python
        model.save(MODEL_PATH)
```
- Sauvegarde le modèle entraîné.

```python
    return model
model = get_or_train_model()
```
- Retourne le modèle (chargé ou entraîné).

---

## Interface utilisateur avec Streamlit

```python
st.title("Détection de SMS Spam avec LSTM")
st.markdown("Entrez un message texte pour vérifier s’il s’agit de spam ou non :")
```
- Affiche le titre de l’application et une brève description.

```python
user_input = st.text_area("Message texte", "Congratulations! You've won a free ticket to Bahamas. Reply now!")
```
- Affiche une zone de texte pour que l’utilisateur saisisse un message.

---

## Prédiction

```python
def predict_sms(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]
    label = "Spam 📛" if pred > 0.5 else "Non-Spam ✅"
    return label, pred
```
- Convertit le texte utilisateur en séquence, applique le padding, et utilise le modèle pour prédire la probabilité que ce soit un spam.
- Retourne une étiquette lisible et le score.

```python
if st.button("Analyser"):
    label, score = predict_sms(user_input)
    st.markdown(f"### Résultat : `{label}`")
    st.write(f"Score de confiance : **{score:.4f}**")
```
- Quand l’utilisateur clique sur le bouton "Analyser", affiche l’étiquette et le score de confiance.

---

## Résumé

Ce script propose une application simple et interactive pour la détection de SMS spam, combinant un modèle LSTM et une interface utilisateur avec Streamlit.
