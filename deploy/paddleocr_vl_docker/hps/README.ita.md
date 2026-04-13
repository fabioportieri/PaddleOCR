# Implementazione come servizio ad alte prestazioni di PaddleOCR-VL-1.5

[English](README_en.md)

Questa directory fornisce una soluzione di implementazione come servizio ad alte prestazioni di PaddleOCR-VL-1.5 che supporta l'elaborazione di richieste simultanee.

> Attualmente questa soluzione supporta solo le GPU NVIDIA; il supporto per altri dispositivi di inferenza è ancora in fase di sviluppo.

## Architettura

```
Client → Gateway FastAPI → Server Triton → Server vLLM
```

| Componente      | Descrizione                                                                                                                                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gateway FastAPI | Punto di accesso unificato, semplificazione delle chiamate client, controllo della concorrenza                                                                                                                           |
| Server Triton   | Modello di rilevamento del layout (PP-DocLayoutV3) e logica di concatenamento della linea di produzione, responsabile della gestione dei modelli, dell'elaborazione batch dinamica e della pianificazione dell'inferenza |
| Server vLLM     | VLM (PaddleOCR-VL-1.5), inferenza in batch continuo                                                                                                                                                                      |

**Modelli Triton:**

| Modello             | Dispositivo                           | Descrizione                                                                                                          |
| ------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `layout-parsing`    | Dispositivo di inferenza (ad es. GPU) | Inferenza di analisi del layout                                                                                      |
| `restructure-pages` | CPU                                   | Post-elaborazione dei risultati multipagina (unione di tabelle su più pagine, riassegnazione dei livelli dei titoli) |

## Requisiti di sistema

- CPU x64
- GPU NVIDIA, Compute Capability >= 8.0 e < 12.0
- Driver NVIDIA con supporto CUDA 12.6
- Docker >= 19.03
- Docker Compose >= 2.0

## Guida rapida

1. Clonare il codice sorgente di PaddleOCR e passare alla directory corrente:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps
```

2. Preparare i file necessari:

```bash
bash prepare.sh
```

3. Avviare il servizio:

```bash
docker compose up
```

Il comando sopra riportato avvierà in sequenza 3 container:

| Servizio                    | Descrizione                                | Porta          |
| --------------------------- | ------------------------------------------ | -------------- |
| `paddleocr-vl-api`          | Gateway FastAPI (punto di accesso esterno) | 8080           |
| `paddleocr-vl-tritonserver` | Server di inferenza Triton                 | 8000 (interno) |
| `paddleocr-vlm-server`      | Servizio di inferenza VLM basato su vLLM   | 8080 (interno) |

> Al primo avvio, l'immagine verrà scaricata e compilata automaticamente; questa operazione richiede un po' di tempo. A partire dal secondo avvio, verrà utilizzata direttamente l'immagine locale, con una velocità di avvio maggiore.

## Istruzioni di configurazione

### Variabili d'ambiente

Copiare `.env.example` in `.env` e modificarlo secondo necessità.

```bash
cp .env.example .env
```

Oltre alle impostazioni tramite il file `.env`, è possibile impostare direttamente le variabili d'ambiente, ad esempio:

```bash
export HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
```

| Variabile                                   | Valore predefinito               | Descrizione                                                                                           |
| ------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`     | 16                               | Numero massimo di richieste simultanee per operazioni di inferenza (analisi della pagina)             |
| `HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS` | 64                               | Numero massimo di richieste simultanee per operazioni non di inferenza (riorganizzazione multipagina) |
| `HPS_INFERENCE_TIMEOUT`                     | 600                              | Tempo di timeout della richiesta (in secondi)                                                         |
| `HPS_HEALTH_CHECK_TIMEOUT`                  | 5                                | Tempo di timeout del controllo di integrità (in secondi)                                              |
| `HPS_VLM_URL`                               | http://paddleocr-vlm-server:8080 | Indirizzo del server VLM (utilizzato per il controllo di integrità)                                   |
| `HPS_LOG_LEVEL`                             | INFO                             | Livello di log (DEBUG, INFO, WARNING, ERROR)                                                          |
| `HPS_FILTER_HEALTH_ACCESS_LOG`              | true                             | Se filtrare i log di accesso dei controlli di integrità                                               |
| `UVICORN_WORKERS`                           | 4                                | Numero di processi worker del gateway                                                                 |
| `DEVICE_ID`                                 | 0                                | ID del dispositivo di inferenza utilizzato                                                            |

### Modifica della configurazione della linea di produzione

Per modificare le impostazioni relative alla pipeline (come il percorso del modello, la dimensione dei batch, i dispositivi di distribuzione, ecc.), consultare la sezione dedicata alla modifica delle impostazioni della pipeline nel [Tutorial su PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md).

## Utilizzo dell'API

### Analisi dei documenti

Si prega di fare riferimento al capitolo relativo alle chiamate client nella [Guida all'uso di PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md).

### Controllo di integrità

```bash
# Controllo di sopravvivenza
curl http://localhost:8080/health

# Controllo di prontezza (verifica che i servizi Triton e VLM siano pronti a elaborare le richieste)
curl http://localhost:8080/health/ready
```

## Ottimizzazione delle prestazioni

### Impostazioni di concorrenza

Il gateway gestisce in modo indipendente la concorrenza per le operazioni di inferenza e quelle non di inferenza:

- **`HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`** (impostazione predefinita 16): controlla il numero di operazioni di inferenza in concorrenza, come `layout-parsing` (analisi del layout)
  - Troppo basso (4): l'utilizzo del dispositivo di inferenza è insufficiente e le richieste vengono messe in coda inutilmente
  - Troppo alto (64): può causare un sovraccarico di Triton, con conseguente OOM o timeout
  - Il valore predefinito 16 consente di avere una coda di richieste sufficiente per formare il lotto successivo durante l'elaborazione del lotto corrente
  - Se le risorse del dispositivo di inferenza sono limitate, si consiglia di ridurre opportunamente questo valore
- **`HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS`** (impostazione predefinita 64): controlla il numero di operazioni non di inferenza in concorrenza, come `restructure-pages` (ristrutturazione multipagina)
  - Le operazioni non di inferenza non occupano risorse dei dispositivi di inferenza, quindi è possibile impostare un numero di operazioni in concorrenza più elevato
  - È possibile regolarlo in base al numero di core della CPU e alla disponibilità di memoria

**Esempio di configurazione ad alto throughput:**

````bash
# .env
```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=32
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=128
UVICORN_WORKERS=8
````

**Esempio di configurazione a bassa latenza:**

```bash
# .env
HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=8
HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=32
HPS_INFERENCE_TIMEOUT=300
UVICORN_WORKERS=2
```

### Numero di processi worker

Ogni worker Uvicorn è un processo indipendente con un proprio ciclo di eventi:

- **1 worker**: semplice, ma limitato a un singolo processo
- **4 worker**: adatto alla maggior parte dei casi d'uso
- **8+ worker**: adatto a scenari con elevata concorrenza e grandi quantità di piccole richieste

### Batch dinamico di Triton

Triton esegue automaticamente il batching delle richieste per migliorare l'utilizzo delle risorse di inferenza. La dimensione massima del batch è controllata dal parametro `max_batch_size` nel file di configurazione del modello (impostazione predefinita: 8). Il file di configurazione si trova in `config.pbtxt` nella directory del repository del modello (ad esempio, `model_repo/layout-parsing/config.pbtxt`).

### Numero di istanze di Triton

Il numero di istanze di inferenza parallela per ogni modello Triton è configurato tramite `instance_group` nel file `config.pbtxt` (impostazione predefinita: 1). Aumentare il numero di istanze migliora la capacità di elaborazione parallela, ma richiede più risorse del dispositivo.

```
# model_repo/layout-parsing/config.pbtxt
instance_group [
  {
      count: 1       # Numero di istanze; aumentarlo migliora il parallelismo
      kind: KIND_GPU
      gpus: [ 0 ]
  }
]
```

Esiste un compromesso tra il numero di istanze e l'elaborazione dinamica dei batch:

- **Singola istanza (`count: 1`)**: l'elaborazione dinamica dei batch unisce più richieste in un unico batch da eseguire in parallelo, ma le richieste dello stesso batch devono attendere il completamento di quella più lenta prima di poter essere restituite insieme, il che può causare un aumento della latenza per alcune richieste. Inoltre, una singola istanza può elaborare solo un batch alla volta; quando il batch corrente non è completato, le richieste successive devono attendere in coda. Adatto a scenari con memoria video limitata o con tempi di elaborazione delle richieste relativamente uniformi
- **Più istanze (`count: 2+`)**: più istanze possono elaborare contemporaneamente batch diversi, consentendo di gestire un numero maggiore di richieste, ridurre i tempi di attesa in coda e migliorare la latenza delle singole richieste. Tuttavia, è importante notare che i batch all'interno della stessa istanza seguono comunque il comportamento dell'elaborazione dinamica in batch (le richieste all'interno del batch iniziano e terminano insieme). Ogni istanza aggiuntiva occuperà una porzione aggiuntiva di memoria video del modello di rilevamento delle pagine, aumentando al contempo il carico sul servizio di inferenza VLM e l'utilizzo di memoria e CPU; è necessario configurare il numero di istanze in base alle risorse disponibili sul dispositivo di inferenza.

I modelli non di inferenza (come `restructure-pages`) vengono eseguiti sulla CPU; è possibile aumentare il numero di istanze in base al numero di core della CPU.

## Risoluzione dei problemi

### Impossibile avviare il servizio

Controllare i log di ciascun servizio per individuare il problema:

```bash
docker compose logs paddleocr-vl-api
docker compose logs paddleocr-vl-tritonserver
docker compose logs paddleocr-vlm-server
```

Tra le cause più comuni figurano porte occupate, dispositivi di inferenza non disponibili o errori nel download delle immagini.

### Errori di timeout

- Aumentare `HPS_INFERENCE_TIMEOUT` (per documenti complessi)
- Se il dispositivo di inferenza è sovraccarico, ridurre `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`

### Memoria/VRAM insufficiente

- Ridurre `HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`
- Assicurarsi che su ogni dispositivo di inferenza sia in esecuzione un solo servizio
- Controllare `shm_size` nel file compose.yaml (impostazione predefinita: 4 GB)
