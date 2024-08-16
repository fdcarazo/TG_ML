#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
import sys
import os 
def reemplazar_caracteres_en_archivo(nombre_archivo):
    # Definir diccionario de reemplazo
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    try:
        # with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        with open(nombre_archivo, 'r') as archivo:
            contenido = archivo.read()
        # Realizar el reemplazo
        for caracter, reemplazo in reemplazos.items():
            contenido = contenido.replace(caracter, reemplazo)
        # Sobrescribir el archivo con los cambios
        with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
            archivo.write(contenido)
        print(f"Se han reemplazado las vocales acentuadas y la letra 'ñ' en el archivo {nombre_archivo}.")
    except FileNotFoundError:
        print(f"El archivo {nombre_archivo} no fue encontrado.")
    except IOError as e:
        print(f"Error de E/S al trabajar con el archivo {nombre_archivo}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python reemplazar_vocales.py <nombre_archivo>")
    else:
        archivo_fuente = os.getcwd()+'/'+str(sys.argv[1])
        reemplazar_caracteres_en_archivo(archivo_fuente)
