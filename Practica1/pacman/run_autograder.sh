#!/bin/bash

# Archivo de salida
output_file="autograder_results.txt"

# Limpiar el archivo antes de empezar
> "$output_file"

# Bucle para ejecutar las preguntas del 1 al 6
for n in {1..6}; do
    echo "==============================" >> "$output_file"
    echo "Ejecutando: python3 autograder.py -q q$n" >> "$output_file"
    echo "==============================" >> "$output_file"
    
    # Ejecutar y redirigir tanto stdout como stderr
    python3 autograder.py -q q$n >> "$output_file" 2>&1

    echo -e "\n\n" >> "$output_file"
done

echo "Ejecución completada. Resultados guardados en $output_file"
