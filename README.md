# System 1 (compile + simulate + plot)
./run_system1.sh

# System 2 – single run
./run_system2.sh -N 300 -seeds 42,43,44

# System 2 – timing sweep (task 1.1)
./run_system2.sh -Nlist 100,200,300,500,700,1000

# Animate a specific run
python3 visualizer/animate_particles.py \
    --states output/system2/N300_k1000/seed42/states.txt \
    --out anim.mp4