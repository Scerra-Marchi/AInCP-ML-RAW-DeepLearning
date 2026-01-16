# TODO

## QUESTIONS
- [ ] Serie temporali monodimensionali si o no? (tenere preprocessing?)
- [ ] Quali modelli provare? 
    - RNN
    - LSTM
    - XGBOOST
    - TRANSFORMERS
- [ ] E non ce la vogliamo mettere un'altra rete neurale al posto del regressore? (da serie temporale anzichè percentuale)
- [ ] Che preprocessing fare ai dati RAW e che windowsize usare? (anna, sofia)
- [ ] E I DATI? (path di sotto)
- [ ] controllare il parallelismo (quel -1 di njobs)
- [x] controllare quella storia del numero che non torna di anna e i tre grafici della correlazione finale
- [ ] sbarazzarsi anche della grid search su windowsize?
- [ ] xk dentro predict_samples non usiamo create_windows?
- [ ] in predict_samples esploderà la memoria perché ci salviamo la serie temporale
- [ ] Explainability del regressore (Grosso valore aggiunto!)
    - Vedere se regressore da più peso ai classificatori che avevamo scelto per primi
- [ ] Explainable graph dei punti dei classificatori (heatmap graph)
- [ ] Cercare pattern per ulteriore explainability del grafico dei punti
- [ ] Passare al regressore la serie temporale dei punti (assieme agli orari) e poi passare l'intera serie al regressore
- [ ] Andare sui dati RAW in ogni caso

## TO IMPROVE
- [ ] create_windows.py rimuovere loop se possibile, e poi read_csv problema I/O?
- [ ] 

## TO ADD
- [ ] 
- [ ] 

## DONE!
- controllare quella storia del numero che non torna di anna e i tre grafici della correlazione finale
-

---

Contattare:
s.pucci26@studenti.unipi.it
- Reti neurali (LSTM? CNN 1D)
- Serie multidimensionali cambiando classificatore
- No AI sui dati raw, ma usando una nuova formula (timeseries -> timeseries su cui fare AI)
- Provare anche altre window_size, e strategie di downsampling differenti (usando librerie) (80hz -> 30hz circa)
- Regressore invariato (qualcosa di diverso)
- Ulteriori modalità di trasformare da bidimensionali a monodimensionali?
Controllare cosa non ha iniziato e fare e il resto

Dati RAW di 80 soggetti sulla macchina remota:
- 60 soggetti -> scerramarchi@redcap:/data03/AInCPClinicalData/Dati_Actigraph_RAW/data
- 80 soggetti -> incoming

Sessioni:
27 Febbraio 2026
- 15 febbraio aver finito
- complicato
10 Aprile 2026

Conferenza
- Articolo pipeline -> Possiamo portare explainability (?)