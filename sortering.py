def sorter_fil(inndata_fil, utdata_fil):
    # Åpne inndata-filen og les alle linjer
    with open(inndata_fil, 'r', encoding='utf-8') as f:
        linjer = f.readlines()

    # Sorter linjene alfabetisk
    # strip() fjernes ikke permanent her, kun for sorteringsnøkkel
    linjer_sortert = sorted(linjer, key=lambda l: l.strip().lower())

    # Skriv de sorterte linjene til utdata-filen
    with open(utdata_fil, 'w', encoding='utf-8') as f:
        f.writelines(linjer_sortert)


if __name__ == "__main__":
    # Sett inn filnavnene her
    inndata = "Ordliste.txt"
    utdata = "output_sortert.txt"

    sorter_fil(inndata, utdata)
    print(f"Ferdig! Sortert innhold er skrevet til {utdata}")
