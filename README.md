# K-Nearest Neighbours Classifier

## Descrizione dell'Assegnamento

L'assegnamento consiste in tre compiti principali:

- **Compito 1:** Ottenere un set di dati
- **Compito 2:** Costruire un classificatore kNN
- **Compito 3:** Testare il classificatore kNN

### Compito 1: Ottenere un set di dati

Per questo assegnamento, utilizziamo il set di dati MNIST, un benchmark standard per i compiti di machine learning. I dati rappresentano 70.000 cifre scritte a mano in immagini in scala di grigi 28x28, già suddivise in un set di addestramento di 60.000 immagini e un set di test di 10.000 immagini, con 784 attributi e 10 classi.

Poiché i due set sono molto grandi, vengono selezionati due sottoinsiemi casuali all'interno della funzione principale [600 per il set di addestramento e 100 per il set di test]. Utilizzando la libreria di machine learning Keras, parte di TensorFlow, è possibile caricare i dati con le seguenti istruzioni:

```python
from tensorflow.keras.datasets import mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
```
## Compito 2: Costruzione di un Classificatore kNN

Nel secondo compito, abbiamo implementato un classificatore k-Nearest Neighbour (kNN). Il processo è stato suddiviso nelle seguenti fasi:

1. **Verifica delle Variabili di Input**
   - Abbiamo creato due funzioni per garantire la correttezza delle variabili di input:
     - `CheckDataset()`
     - `CheckKvalue()`

2. **Classificazione dei Dati**
   - La classificazione del set di dati è stata eseguita secondo la teoria kNN, come descritto nelle formule 2, 3 e 4.
   - Abbiamo utilizzato una funzione personalizzata, `KNNclassifier()`, per effettuare la classificazione.
   - Dopo aver applicato la funzione `reshape` per gestire i dati in 2 dimensioni e inizializzato le liste necessarie, abbiamo calcolato una lista di predizioni, memorizzandola in un dizionario.
   - I dati sono stati ordinati in ordine crescente in base alla distanza euclidea e abbiamo ottenuto il conteggio della classe dominante nella lista dei risultati.

3. **Calcolo delle Metriche di Performance**
   - Abbiamo calcolato l'accuratezza e il tasso di errore confrontando i valori nella lista delle predizioni con `testY`.
   - La funzione restituisce i seguenti risultati:
     - Accuratezza
     - Tasso di errore
     - Lista delle predizioni

## Compito 3: Testare il Classificatore kNN

Nel terzo compito, abbiamo testato il classificatore kNN utilizzando diversi valori di `k`. I passi principali sono stati i seguenti:

1. **Test dei Valori di `k`**
   - I valori di `k` utilizzati per il test sono:
     ```css
     K1 = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
     ```
   - Abbiamo eseguito un ciclo `for` per passare ciascuno di questi valori a `KNNclassifier()`.
   - I risultati ottenuti sono stati tracciati in un grafico a subplot che mostra l'accuratezza e il tasso di errore per ciascun valore di `k`.

2. **Creazione della Matrice di Confusione**
   - Abbiamo generato una matrice di confusione utilizzando le funzioni `ConfusionMatrix()` e `ClassificationQualityIndexes()`.
   - La matrice è stata impostata su una dimensione 10x10 (dove 10 rappresenta il numero di cifre) e abbiamo calcolato gli indici di qualità con le seguenti definizioni:
     - **Vero Positivo (TP)**: Quando il valore effettivo è positivo e la previsione è anch'essa positiva.
     - **Vero Negativo (TN)**: Quando il valore effettivo è negativo e la previsione è anch'essa negativa.
     - **Falso Positivo (FP)**: Quando il valore effettivo è negativo ma la previsione è positiva.
     - **Falso Negativo (FN)**: Quando il valore effettivo è positivo ma la previsione è negativa.

3. **Riduzione dei Valori di `k`**
   - Per ottenere una rappresentazione più chiara delle prestazioni, abbiamo ridotto il set di valori di `k` a:
     ```css
     K2 = [1, 3, 5, 10, 25, 50]
     ```
   - Abbiamo poi confrontato l'accuratezza del classificatore per le diverse classi (cifre), verificando ogni cifra rispetto alle altre nove, e ripetendo i passaggi sopra descritti.

Questa procedura ci ha permesso di valutare l'efficacia del classificatore kNN con diversi parametri e di ottenere una comprensione più dettagliata delle sue prestazioni.
