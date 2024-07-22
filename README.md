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
### Compito 2: Costruire un classificatore kNN
Nel secondo compito, dobbiamo implementare un classificatore k-Nearest Neighbour. Per verificare la correttezza delle variabili di input, abbiamo creato alcune funzioni: 'CheckDataset()' e CheckKvalue().

Successivamente, abbiamo classificato il set di dati secondo la teoria kNN, descritta con le formule 2, 3 e 4, e restituito la classificazione ottenuta. La classificazione è stata calcolata utilizzando una funzione creata da noi: KNNclassifier(). Dopo aver applicato la funzione reshape per lavorare in 2 dimensioni e inizializzato le liste, abbiamo calcolato una lista di predizioni e l'abbiamo memorizzata in un dizionario. Quindi abbiamo ordinato in ordine crescente per distanza euclidea e ottenuto il conteggio della classe massima nella lista dei risultati.

Infine, abbiamo calcolato l'accuratezza e il tasso di errore verificando se i valori all'interno della lista delle predizioni erano uguali a testY. La funzione restituisce: accuratezza, tasso di errore e lista delle predizioni.
