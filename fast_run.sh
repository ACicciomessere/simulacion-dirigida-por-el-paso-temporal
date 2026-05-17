#!/bin/bash

# Bucle que va desde 42 hasta 48 (7 ejecuciones en total)
for seed in {42..48}
do
    echo "Iniciando batch_run.sh con seed: $seed"
    ./batch_run.sh "$seed"
done

echo "¡Todas las ejecuciones han finalizado!"