package ar.edu.itba.sds.system2;

import java.io.*;
import java.util.*;

/**
 * System 2: Molecular Dynamics in a circular enclosure with a central fixed obstacle.
 *
 * Domain : circle of radius R = L/2 = 40 m
 * Obstacle: fixed circle at origin, radius r0 = 1 m
 * Particles: radius r = 1 m, mass m = 1 kg, |v0| = 1 m/s, random direction
 * Force: elastic, F_ij = k * xi * e_ij  (xi = overlap, e_ij = unit vec from j to i)
 *
 * Integrator: Velocity-Verlet (forces are position-only → symplectic)
 *
 * States:
 *   fresh (isUsed=false): initial state; restored when particle hits outer wall
 *   used  (isUsed=true) : set when a fresh particle first touches the obstacle
 *
 * Cfc(t): cumulative count of fresh→used state changes (first dt of each contact episode).
 */
public class MDSimulation {

    // ── Fixed domain parameters ──────────────────────────────────────────────
    public static final double R_DOMAIN  = 40.0;  // outer wall radius
    public static final double R_OBSTACLE = 1.0;  // central obstacle radius
    public static final double R_PARTICLE = 1.0;  // particle radius
    public static final double MASS       = 1.0;  // kg
    public static final double V0         = 1.0;  // m/s

    // ── Simulation state ─────────────────────────────────────────────────────
    private final int N;
    private final double k;
    private final double dt;
    private final double dt2;
    private final double tf;
    private final long seed;

    private List<Particle> particles;

    // Contact tracking (to detect first dt of each obstacle-contact episode)
    private final boolean[] wasInContactObs;

    // Cfc vector: stored at max resolution (every dt)
    private final List<double[]> cfcSeries;  // [time, cfc]
    private int cfc;

    // Energy series (sampled every dt2)
    private final List<double[]> energySeries;  // [time, E]

    public MDSimulation(int N, double k, double dt, double dt2, double tf, long seed) {
        this.N    = N;
        this.k    = k;
        this.dt   = dt;
        this.dt2  = dt2;
        this.tf   = tf;
        this.seed = seed;

        wasInContactObs = new boolean[N];
        cfcSeries       = new ArrayList<>();
        energySeries    = new ArrayList<>();
        cfc = 0;

        particles = initParticles(new Random(seed));
    }

    // ── Initialization ───────────────────────────────────────────────────────

    private List<Particle> initParticles(Random rng) {
        List<Particle> list = new ArrayList<>(N);
        double maxR = R_DOMAIN   - R_PARTICLE;
        double minR = R_OBSTACLE + R_PARTICLE;
        int placed = 0, attempts = 0, maxAttempts = N * 50000;

        while (placed < N && attempts < maxAttempts) {
            // Uniform area distribution in annular ring
            double r = Math.sqrt(rng.nextDouble() * (maxR * maxR - minR * minR) + minR * minR);
            double ang = rng.nextDouble() * 2 * Math.PI;
            double x = r * Math.cos(ang);
            double y = r * Math.sin(ang);

            boolean ok = true;
            for (Particle p : list) {
                double dx = x - p.x, dy = y - p.y;
                if (Math.sqrt(dx * dx + dy * dy) < 2.0 * R_PARTICLE) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                double va = rng.nextDouble() * 2 * Math.PI;
                list.add(new Particle(placed, x, y, V0 * Math.cos(va), V0 * Math.sin(va), R_PARTICLE, MASS));
                placed++;
            }
            attempts++;
        }
        if (placed < N)
            System.err.printf("Warning: placed only %d/%d particles (seed=%d)%n", placed, N, seed);
        return list;
    }

    // ── Force computation (O(N^2) + boundary) ───────────────────────────────

    private void computeForces() {
        int n = particles.size();
        double[] fx = new double[n];
        double[] fy = new double[n];

        // Particle–particle
        for (int i = 0; i < n; i++) {
            Particle pi = particles.get(i);
            for (int j = i + 1; j < n; j++) {
                Particle pj = particles.get(j);
                double dx = pi.x - pj.x, dy = pi.y - pj.y;
                double dist = Math.sqrt(dx * dx + dy * dy);
                double xi = pi.radius + pj.radius - dist;
                if (xi > 0) {
                    double fn = k * xi / dist;
                    fx[i] += fn * dx;  fy[i] += fn * dy;
                    fx[j] -= fn * dx;  fy[j] -= fn * dy;
                }
            }

            // Fixed obstacle at origin (repulsive, pushes particle outward)
            double dist = pi.distFromOrigin();
            double xiObs = R_OBSTACLE + pi.radius - dist;
            if (xiObs > 0 && dist > 1e-12) {
                double fn = k * xiObs / dist;
                fx[i] += fn * pi.x;
                fy[i] += fn * pi.y;
            }

            // Outer wall (repulsive, pushes particle inward)
            double xiWall = pi.radius + dist - R_DOMAIN;
            if (xiWall > 0 && dist > 1e-12) {
                double fn = k * xiWall / dist;
                fx[i] -= fn * pi.x;
                fy[i] -= fn * pi.y;
            }
        }

        for (int i = 0; i < n; i++) {
            Particle p = particles.get(i);
            p.ax = fx[i] / p.mass;
            p.ay = fy[i] / p.mass;
        }
    }

    // ── State transitions & Cfc tracking ────────────────────────────────────

    private void updateStates(double t) {
        int n = particles.size();
        for (int i = 0; i < n; i++) {
            Particle p = particles.get(i);
            double dist = p.distFromOrigin();

            boolean inObs  = (R_OBSTACLE + p.radius - dist) > 0;
            boolean inWall = (p.radius + dist - R_DOMAIN)   > 0;

            // Fresh particle first touches obstacle → count in Cfc
            if (inObs && !wasInContactObs[i] && !p.isUsed) {
                cfc++;
                cfcSeries.add(new double[]{t, cfc});
            }

            // State transitions
            if (inObs)  p.isUsed = true;
            if (inWall) p.isUsed = false;

            wasInContactObs[i] = inObs;
        }
    }

    // ── Total energy (kinetic + elastic potential) ───────────────────────────

    public double totalEnergy() {
        int n = particles.size();
        double E = 0;
        for (int i = 0; i < n; i++) {
            Particle pi = particles.get(i);
            E += 0.5 * pi.mass * pi.speedSq();

            double dist = pi.distFromOrigin();
            double xiObs  = R_OBSTACLE + pi.radius - dist;
            double xiWall = pi.radius + dist - R_DOMAIN;
            if (xiObs  > 0) E += 0.5 * k * xiObs  * xiObs;
            if (xiWall > 0) E += 0.5 * k * xiWall * xiWall;

            for (int j = i + 1; j < n; j++) {
                Particle pj = particles.get(j);
                double dx = pi.x - pj.x, dy = pi.y - pj.y;
                double d  = Math.sqrt(dx * dx + dy * dy);
                double xi = pi.radius + pj.radius - d;
                if (xi > 0) E += 0.5 * k * xi * xi;
            }
        }
        return E;
    }

    // ── Run simulation ───────────────────────────────────────────────────────

    public long run(String outDir) throws IOException {
        new File(outDir).mkdirs();
        int steps       = (int) Math.round(tf / dt);
        int outputEvery = Math.max(1, (int) Math.round(dt2 / dt));

        computeForces();   // seed accelerations for velocity-Verlet

        long startMs = System.currentTimeMillis();

        try (PrintWriter statesPw = new PrintWriter(new BufferedWriter(
                new FileWriter(outDir + "/states.txt")))) {

            writeFrame(statesPw, 0.0);
            energySeries.add(new double[]{0.0, totalEnergy()});

            for (int step = 1; step <= steps; step++) {
                double t = step * dt;

                // ─ Velocity-Verlet step 1: half-kick + full drift ─
                for (Particle p : particles) {
                    p.vx += 0.5 * p.ax * dt;
                    p.vy += 0.5 * p.ay * dt;
                    p.x  += p.vx * dt;
                    p.y  += p.vy * dt;
                }

                // ─ Recompute forces ─
                computeForces();

                // ─ Half-kick with new accelerations ─
                for (Particle p : particles) {
                    p.vx += 0.5 * p.ax * dt;
                    p.vy += 0.5 * p.ay * dt;
                }

                // ─ State transitions (every dt for Cfc accuracy) ─
                updateStates(t);

                // ─ Output at dt2 intervals ─
                if (step % outputEvery == 0) {
                    writeFrame(statesPw, t);
                    energySeries.add(new double[]{t, totalEnergy()});
                }
            }
        }

        long elapsed = System.currentTimeMillis() - startMs;

        writeCfc(outDir + "/cfc.txt");
        writeEnergy(outDir + "/energy.txt");
        writeInfo(outDir + "/info.txt", elapsed);

        return elapsed;
    }

    // ── Output helpers ───────────────────────────────────────────────────────

    private void writeFrame(PrintWriter pw, double t) {
        pw.println(particles.size());
        pw.printf(Locale.US, "time=%.6f%n", t);
        for (Particle p : particles) {
            pw.printf(Locale.US, "%.6f %.6f %.6f %.6f %.6f %d%n",
                    p.x, p.y, p.vx, p.vy, p.radius, p.isUsed ? 1 : 0);
        }
    }

    private void writeCfc(String path) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("time,cfc");
            for (double[] e : cfcSeries)
                pw.printf(Locale.US, "%.6f,%d%n", e[0], (int) e[1]);
        }
    }

    private void writeEnergy(String path) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("time,energy");
            for (double[] e : energySeries)
                pw.printf(Locale.US, "%.6f,%.8e%n", e[0], e[1]);
        }
    }

    private void writeInfo(String path, long elapsedMs) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.printf("N=%d%n", particles.size());
            pw.printf("k=%.1f%n", k);
            pw.printf("dt=%.2e%n", dt);
            pw.printf("dt2=%.4f%n", dt2);
            pw.printf("tf=%.1f%n", tf);
            pw.printf("seed=%d%n", seed);
            pw.printf("elapsed_ms=%d%n", elapsedMs);
        }
    }

    public List<Particle> getParticles() { return particles; }
    public List<double[]> getCfcSeries()  { return cfcSeries; }
}
