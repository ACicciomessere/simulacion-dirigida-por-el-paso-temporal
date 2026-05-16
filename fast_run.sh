#!/bin/bash

# Bucle que va desde 44 hasta 47 (4 ejecuciones en total)
for seed in {48..49}
do
    echo "Iniciando batch_run.sh con seed: $seed"
    ./batch_run.sh "$seed"
done

echo "¡Todas las ejecuciones han finalizado!"