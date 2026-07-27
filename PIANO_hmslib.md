# hmslib — piano di progetto

Mini libreria per analisi statistica e algoritmi di fault detection & diagnosis
su liquid rocket engines. Uso previsto: PC Windows **senza accesso a internet**,
librerie già presenti, versione Python ignota (presumibilmente recente).

---

## 1. Vincoli accertati

| # | vincolo | conseguenza sul design |
|---|---|---|
| 1 | PC offline, nessuna installazione possibile | pacchetto = cartella da copiare, `import hmslib` senza `pip install` |
| 2 | solo librerie certe: numpy, scipy, pandas, scikit-learn, matplotlib, torch | zero dipendenze opzionali (no seaborn/pyyaml/pyarrow/tqdm/statsmodels) |
| 3 | versione Python ignota | codice compatibile 3.9+, wrapper per le API instabili, `check_env()` |
| 4 | dati: 1 file nominale + 1 file failure per ogni punto di lavoro (OP) | discovery + `manifest.json` correggibile a mano |
| 5 | nomi file e nomi colonne ignoti a priori | inferenza schema *proposta e stampata*, mai applicata in silenzio |
| 6 | nomenclatura sensori stabile fra OP, ma qualche sensore può essere disattivato | lista sensori per-OP; `common_sensors()` solo per confronti cross-OP |
| 7 | tutto steady state, nessun asse temporale | niente feature dinamiche, niente latenza di detection |
| 8 | OP noto a test-time, punti discreti | un modello per OP (`ModelBank`); estensione parametrica futura non preclusa |
| 9 | max ~30 000 righe per file | SVM kernel praticabile senza approssimazioni |
| 10 | intensità del guasto: **scalare**, continua, variabile dentro il loop MC | formalismo POD (vedi §5) |
| 11 | normalizzazione dell'intensità ignota | flag `intensity_mode = auto \| absolute \| percent` + check automatico |
| 12 | segno dell'intensità: solo interpretativo | ordinamento e POD su `|intensity|`, valore con segno conservato per i plot |
| 13 | obiettivo finale: open set, unknown = classi di guasto assenti dal training | rigetto a soglia, protocollo leave-one-class-out |
| 14 | architettura a **due stadi** (detection → diagnosi) | lo stadio 2 vede solo i campioni promossi dallo stadio 1 |
| 15 | tutto offline, nessun vincolo real-time; nessuna metrica imposta da Avio | libertà nella scelta delle metriche |
| 16 | possibilità di aggiungere rumore sintetico | `preprocess.add_noise(level, mode)` |
| 17 | sensor selection / OpenMax / metodi del filone `osr`: fuori scopo | — |

### Trappole di compatibilità da evitare nel codice

| trappola | sostituto |
|---|---|
| `np.NaN`, `np.float_` (rimossi in numpy 2) | `np.nan`, `float` |
| `np.trapezoid` vs `np.trapz` | wrapper in `compat.py` |
| `df.append` (rimosso in pandas 2) | `pd.concat` |
| `sparse_output=`, `force_all_finite=` (rinominati in sklearn) | non usarli |
| `plt.cm.get_cmap` (rimosso in matplotlib 3.9) | `matplotlib.colormaps[...]` con fallback |
| `torch.load(weights_only=...)` (default cambiato) | try/except esplicito |
| annotazioni `X \| None` a runtime | `from __future__ import annotations` + `Optional` |

`hmslib.check_env()` stampa tutte le versioni rilevate e segnala le incompatibilità
prima che si manifestino come tracebacks oscuri.

---

## 2. Osservazioni sui dati di esempio (`esempio/data/`)

Misurate, non ipotizzate:

- nominale 3000 × 26, failure 24 061 × 27 con label `Failure`, 28 classi (467–1600 campioni)
- **covarianza del nominale esattamente singolare**: `HPOTP_w` ≡ `PBOBP_w` (r = 1.000000)
- cond(Σ) ≈ 2.6e19; 16 autovalori su 26 sotto 1e−3 → rango utile ~10 su 26
- altre coppie quasi collineari: `Man_OX_Temp`/`LPOTP_T_turb_inlet` (0.9995),
  `HPOTP_p_pump_out`/`PBOBP_p_pump_out` (0.985)
- marginali nominali quasi gaussiane (|skew| max 0.14) → soglia χ² difendibile,
  **a patto** di trattare il rango effettivo

Sono i dati SSME, non quelli DM2 che arriveranno, ma la patologia (sensori ridondanti
o derivati l'uno dall'altro) è strutturale nei modelli di ciclo motore e si ripresenterà.

---

## 3. Struttura del pacchetto

```
hmslib/
  compat.py       check_env, wrapper per le API instabili
  config.py       dataclass di configurazione, stile figure, seed globale
  schema.py       inferenza e override dei ruoli colonna
  io.py           scan_folder, manifest, Dataset, OperatingPointData, split
  quality.py      NaN/inf, costanti, duplicati, collinearità, rango -> QualityReport
  eda.py          statistiche descrittive, correlazioni, VIF, normalità, PCA
  viz.py          istogrammi, heatmap, ellissi sigma, PCA 2D, trend_vs_intensity, contributi
  preprocess.py   scaler standard/robusto, dedup collineari, rumore sintetico
  detect/
    base.py         contratto comune fit/score/threshold_/predict/save/load
    mahalanobis.py  §4
    pca_spe.py      Hotelling T2 + Q/SPE
    ocsvm.py        One-Class SVM / SVDD
  classify/
    base.py         contratto open-set comune
    openset.py      SVM / RF / kNN con rigetto a soglia calibrata
  nn/
    backend.py      import torch centralizzato, device, seed
    autoencoder.py  AE, denoising AE, residui per sensore
    classifier.py   MLP + soglia sul softmax
    training.py     early stopping, save/load, curve di loss
  analysis.py     detectability, curve POD, intensità minima rilevabile, separabilità
  evaluate.py     ROC/PR, TPR@FPR, confusion, metriche open-set, bootstrap CI
  report.py       quicklook, detection_report, openset_report -> PDF multipagina
  synth.py        generatore sintetico per test e prove senza dati veri
templates/        notebook: 00_quicklook, 01_mahalanobis, 02_detectability,
                  03_openset, 04_nn
tests/            pytest su dati sintetici a proprietà note
```

Niente CLI: libreria + notebook template.

---

## 4. Mahalanobis robusto

Catena esplicita, con diagnostica sempre esposta:

1. **Pre-check strutturale** prima di stimare Σ: colonne costanti, colonne
   duplicate o quasi (soglia su |r|), rango numerico.
2. **Scaling**: standard oppure robusto (mediana/MAD), selezionabile.
3. **Stimatore di Σ**: `empirical | ledoit_wolf | oas | mcd | diagonal | pca_truncated`.
   Mai implicito.
4. **Inversione: mai `np.linalg.inv`.** Cholesky di Σ + λI con `solve_triangular`,
   d² = ‖L⁻¹(x−μ)‖². Fallback su eigendecomposizione con troncamento
   (floor = λ_max·rcond, oppure k componenti a varianza spiegata fissata).
5. **`diagnostics_`**: cond(Σ), rango effettivo p_eff, λ usato, spettro,
   rapporto n/p con warning se n < 10p.
6. **Soglia**, tre modi confrontabili:
   - χ²(p_eff, 1−α)
   - Hotelling T² con correzione F per μ e Σ stimati su campione finito
   - percentile empirico **out-of-sample** (default: split nominale train/calib)
7. **`contributions(X)`**: decomposizione per sensore della distanza →
   isolamento del guasto, non solo detection.

Modelli separati per OP, orchestrati da `ModelBank`.

---

## 5. Detectability e curve POD — modulo centrale

L'intensità varia con continuità dentro il loop MC, quindi a parità di intensità
i campioni restano dispersi. La detection è una variabile aleatoria e va descritta
come tale.

**Protocollo**

1. Soglia del rivelatore fissata **sul solo nominale**, a un FPR dichiarato
   (default 1e−3). Tutte le curve POD sono valide solo a FPR fissato: confrontare
   rivelatori a soglie diverse non ha senso.
2. Per ogni classe di guasto: `detected = score > threshold`, in funzione di |intensità|.
3. **POD empirica**: binning per quantili di intensità, frazione di detection per bin,
   intervallo di Wilson per bin.
4. **POD parametrica**: modello log-odds lineare in log(intensità) — la formulazione
   classica MIL-HDBK-1823 — da cui si ricavano `i50` e `i90`.
5. **Limite di confidenza**: bootstrap sulle righe → `i90/95`, l'intensità a cui la
   detection è al 90% con confidenza 95%.
6. **Tabella riassuntiva** per OP: una riga per classe di guasto, colonne
   `i50 | i90 | i90/95 | POD@intensità_max`. È il risultato principale.

**Complementi**

- `viz.trend_vs_intensity`: valore del sensore (grezzo, oppure in unità di σ del
  nominale dello stesso OP — la versione confrontabile fra sensori) contro intensità,
  con nuvola MC in sottofondo, mediana per bin e banda interquartile.
- **Ranking di sensibilità dei sensori**: pendenza di z rispetto all'intensità, o |z|
  a un percentile fissato di intensità → quale sensore reagisce per primo, per classe.
- Split train/test **stratificato per classe × bin di intensità**, altrimenti un set
  raccoglie le intensità alte e le metriche risultano gonfiate.

---

## 6. Architettura a due stadi e open set

```
x, OP noto
   │
   ├─ stadio 1: detector dell'OP (Mahalanobis / OCSVM / AE), soglia a FPR fissato
   │     └─ sotto soglia → "NOMINAL"  (fine)
   │
   └─ stadio 2: classificatore open-set, addestrato solo su campioni rilevabili
         ├─ score ≥ soglia di rigetto → nome della classe
         └─ score <  soglia di rigetto → "UNKNOWN"
```

Punto non negoziabile: lo **stadio 2 va addestrato solo sui campioni che lo stadio 1
promuove**. A intensità bassa un guasto noto e uno ignoto sono entrambi indistinguibili
dal nominale, quindi includerli renderebbe le metriche open-set prive di significato —
misurerebbero la sovrapposizione al nominale, non la capacità di separare noto da ignoto.
La soglia di intensità da cui addestrare esce dall'analisi POD del §5, quindi i due
moduli sono accoppiati per costruzione.

**Valutazione**: leave-one-class-out ciclico su tutte le classi (lo split known/unknown
fissato resta come caso particolare). Metriche: accuratezza sulle sole note, recall
sugli ignoti, AUROC noto-vs-ignoto sullo score di confidenza, curva accuratezza contro
tasso di rigetto, intervalli di confidenza per bootstrap.

---

## 7. Fasi

| fase | contenuto | peso | stato |
|---|---|---|---|
| F0 | compat, schema, io/manifest, quality, `quicklook` | medio | **fatto** |
| F1 | viz, eda, `trend_vs_intensity` | leggero | **fatto** |
| F2 | Mahalanobis robusto: soglie, contributi, test | medio-alto | **fatto** |
| F3 | `analysis`: detectability e curve POD | medio | **fatto** |
| F4 | OCSVM, classificatore open-set, `evaluate` | medio | da fare |
| F5 | nn: autoencoder detector, MLP classifier | medio | da fare |
| F6 | report PDF, notebook template, `synth`, test | medio | **fatto** |

Stato al 2026-07-27: F0, F1, F2, F3 e F6 implementati, 120 test verdi, libreria
in [hmslib/](hmslib/), notebook in [templates/](templates/), test in
[tests/](tests/).

Il modulo `eda` previsto in F1 non è stato scritto come modulo separato: le
statistiche descrittive e la PCA vivono in `quality` e `viz`, dove servivano.

Validazione di F3: per una nuvola nominale gaussiana e un guasto che sposta la
media, `d²` segue una chi-quadro non centrale, quindi la POD vera è calcolabile
in forma chiusa. L'`i90` stimato dal fit cade entro il 3% di quello analitico,
e `i90_95` sta sopra `i90` di circa il 4%.

F0–F2 costituisce già un pacchetto autonomamente utile. I test girano su `synth`,
quindi la libreria si valida su questo PC, senza i dati definitivi.

---

## 8. Decisioni prese (default, modificabili)

- nome pacchetto `hmslib`; API e docstring in inglese
- matplotlib puro, figure vettoriali (PDF) per i report
- configurazione in JSON e dataclass Python, niente YAML
- persistenza modelli con `joblib` (dipendenza di scikit-learn, quindi certa) più
  un JSON di metadata affiancato
- `pytest` sì; il codice va in un repository git già esistente (vedi §9)
- i dati restano nella cartella di lavoro senza restrizioni particolari

---

## 9. Punto di ripresa

Sezione da leggere per prima se si riprende il lavoro dopo una pausa.

### Dove sta cosa

| | |
|---|---|
| [hmslib/](hmslib/) | 13 moduli, ~3100 righe, nessuna dipendenza esterna |
| [hmslib/README.md](hmslib/README.md) | documentazione operativa della libreria |
| [tests/](tests/) | 120 test, girano solo su `synth`, nessun dato esterno |
| [templates/](templates/) | `01_quicklook_and_detection`, `02_detectability` |

Vincolo di layout: `hmslib/`, `tests/` e `templates/` devono restare cartelle
sorelle. `tests/conftest.py` inserisce in `sys.path` la cartella padre di
`tests/`; i notebook fanno `sys.path.insert(0, os.path.abspath(".."))`.
Spostando `hmslib` sotto `src/` o dentro un sottopacchetto, quelle due righe
vanno cambiate.

Da tenere fuori dal repository: `__pycache__/`, `reports/`, `models/`, `data/`,
i CSV in `esempio/data/` (5.6 MB, il codice non li usa), e `manifest.json` —
`scan_folder` ci scrive `root` come percorso **assoluto**. Se si vuole
committare il manifest per non rifare le correzioni manuali, mettere a mano
`"root": "."`: `load_manifest` risolve i percorsi relativi rispetto alla
posizione del file.

### Cosa è stato validato

Sui dati SSME in `esempio/data/`, con soglia mirata allo 0.1% di falsi allarmi:

| catena | sensori | componenti | FPR reale | rilevati |
|---|---|---|---|---|
| default (drop + Ledoit-Wolf + Cholesky) | 24/26 | 24 | 0.20% | 100% |
| ingenua (tutti + covarianza empirica) | 26/26 | 26 | 4.73% | 100% |
| troncata (eigen, 99% varianza) | 26/26 | 7 | 0.13% | 99.7% |

Le due cause: `HPOTP_w` ≡ `PBOBP_w` (r = 1.000000) e `Man_OX_Temp` ~
`LPOTP_T_turb_inlet` (r = 0.9995).

### Limiti noti

| | |
|---|---|
| soglia empirica a α = 10⁻³ | con 3000 righe nominali poggia su ~1 punto; l'avviso compare sempre, decidere se alzare α o usare χ² |
| dati SSME senza colonna intensità | la POD lì non è calcolabile; sui dati DM2 serve che ci sia |
| `var_explained=0.99` | sui dati SSME tiene 7 direzioni su 26: aggressivo, usarlo sapendo cosa si scarta |
| quote dei contributi > 1 | normale, i contributi di sensori correlati che deviano in modo inatteso sono negativi e la somma resta esattamente `d²` |
| un Mahalanobis per OP | assume nuvola unimodale; se un OP risulta multimodale va spezzato |
| `i90` oltre lo sweep campionato | segnalato come estrapolazione in `res.notes`, non fidarsene |
| `cov='mcd'` | richiede n > p e va usato con `scaler='robust'` |

Fuori scopo per accordo: transitori, sensor selection, guasti combinati,
metodi del filone `osr` (OpenMax e simili).

### Procedura quando arrivano i dati veri

1. copiare `hmslib/`, `tests/`, `templates/` sul PC offline;
2. `python -m pytest tests -q` — se i 120 test passano, l'ambiente regge la
   libreria; se qualcosa fallisce si sa cosa e perché prima di toccare i dati;
3. `hm.check_env()`;
4. `scan_folder` → aprire il manifest: correggere i nomi degli OP (se i nomi
   file non condividono token diventa `"OP"`), verificare label e intensità;
5. `quicklook` → guardare subito rango effettivo e coppie duplicate: lì si
   decide se la catena default basta;
6. scegliere α e confrontare le tre soglie **prima** di produrre qualunque
   numero;
7. `detection_report` con `at_alpha` fissato.

### Se si riprende da F4

L'aggancio è nell'ultima cella di `templates/02_detectability.ipynb`: la
colonna `train above` della tabella POD è la soglia di intensità sotto la quale
i campioni sono indistinguibili dal nominale, e su cui lo stadio 2 non va
addestrato. È l'unico vincolo che F4 eredita da F3; il resto del progetto di F4
è nel §6.
