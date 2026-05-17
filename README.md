# System 1 (compile + simulate + plot)
./run_system1.sh

# System 2 – single run
./run_system2.sh -N 300 -seeds 42,43,44

# System 2 – timing sweep (task 1.1)
./run_system2.sh -Nlist 100,200,300,500,600,800,1000

# System 2 - K variations (task 1.4)
## NN siendo el número una seed
./batch_run.sh NN 
## Cada run tarda un poco más de 30 min
## Cuando tenes los runs suficientes
python3 final_run.py
## Ahora los podes animar
python3 visualizer/analyze_system2.py \
    --base output/system2 \
    --k    100,1000,10000 \
    --Nlist 100,200,300,500,600,800,1000 \
    --out  output/system2/plots

# Animate a specific run
python3 visualizer/animate_particles.py \
    --states output/system2/NXXX_k1000/states.txt \
    --out anim.mp4