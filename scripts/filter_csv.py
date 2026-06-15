import argparse
import os
import pandas as pd


def estrai_colonne_csv(file_input, file_output):
    colonne_target = ["gti", "period_end", "period"]

    if not os.path.exists(file_input):
        print(f"Errore: Il file di input '{file_input}' non esiste.")
        return

    try:
        # Lettura del CSV originale
        # Se i tuoi file usano il punto e virgola, aggiungi sep=';'
        df = pd.read_csv(file_input)

        # Intersezione tra le colonne richieste e quelle esistenti nel file
        colonne_presenti = [col for col in colonne_target if col in df.columns]

        if not colonne_presenti:
            print(
                f"Errore: Nessuna delle colonne cercate {colonne_target} è presente nel file."
            )
            return

        # Estrazione delle colonne
        df_estratto = df[colonne_presenti]

        # Salvataggio nel nuovo file CSV
        df_estratto.to_csv(file_output, index=False)

        print(f"✔ Elaborazione completata con successo!")
        print(f"   Colonne estratte: {colonne_presenti}")
        print(f"   Nuovo file creato: '{file_output}'")

    except Exception as e:
        print(f"❌ Si è verificato un errore durante l'elaborazione: {e}")


if __name__ == "__main__":
    # Configurazione del parser CLI
    parser = argparse.ArgumentParser(
        description="Estrae solo le colonne 'ghi', 'period_end' e 'period' da un file CSV di origine."
    )

    # Argomenti posizionali obbligatori
    parser.add_argument(
        "input", type=str, help="Percorso del file CSV di origine (input)"
    )
    parser.add_argument(
        "output", type=str, help="Percorso del nuovo file CSV da creare (output)"
    )

    # Parsing degli argomenti lanciati dall'utente
    args = parser.parse_args()

    # Esecuzione della funzione con i parametri passati da CLI
    estrai_colonne_csv(args.input, args.output)