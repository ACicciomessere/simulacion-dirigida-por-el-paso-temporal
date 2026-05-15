# System 1 (compile + simulate + plot)
./run_system1.sh

# System 2 – single run
./run_system2.sh -N 300 -seeds 42,43,44

# System 2 – timing sweep (task 1.1)
./run_system2.sh -Nlist 100,200,300,500,700,1000

# System 2 - K variations (task 1.4)
./run_system2.sh -Nlist 100,200,300,500,700,800,1000 -k 100 -seeds 42
./run_system2.sh -Nlist 100,200,300,500,700,800,1000 -k 1000 -seeds 42
./run_system2.sh -Nlist 100,200,300,500,700,800,1000 -k 10000 -seeds 42
./run_system2.sh -Nlist 100,200,300,500,700,800,1000 -k 100000 -seeds 42

python3 visualizer/analyze_system2.py \
    --base output/system2 \
    --k    100,1000,10000,100000 \
    --Nlist 100,200,300,500,700,800,1000 \
    --out  output/system2/plots

# Animate a specific run
python3 visualizer/animate_particles.py \
    --states output/system2/NXXX_k1000/seedXX/states.txt \
    --out anim.mp4