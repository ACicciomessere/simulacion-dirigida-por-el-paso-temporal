package ar.edu.itba.sds.system1;

import java.io.*;
import java.util.*;

/**
 * System 1: Damped Harmonic Oscillator
 * Compares Euler, Verlet original, Beeman, and Gear PC order-5 integrators.
 *
 * Parameters (from Teorica_4 slide 36):
 *   m=70 kg, k=1e4 N/m, gamma=100 kg/s, tf=5 s
 *   r(0)=1 m, v(0)=-A*gamma/(2m) m/s  with A=1
 *   Force: f = -k*r - gamma*v
 *   Analytical: r(t) = exp(-gamma/(2m)*t) * cos(omega*t)
 */
public class OscillatorSimulation {

    static final double M     = 70.0;
    static final double K     = 1e4;
    static final double GAMMA = 100.0;
    static final double TF    = 5.0;
    static final double R0    = 1.0;
    static final double V0    = -GAMMA / (2.0 * M);  // = -5/7 m/s

    // Gear PC order-5 corrector coefficients for f(r, v) (velocity-dependent)
    static final double[] GEAR_ALPHA = {3.0/16, 251.0/360, 1.0, 11.0/18, 1.0/6, 1.0/60};

    // ── Force & analytical ──────────────────────────────────────────────────

    static double force(double r, double v) {
        return -K * r - GAMMA * v;
    }

    static double analytical(double t) {
        double omega = Math.sqrt(K / M - GAMMA * GAMMA / (4.0 * M * M));
        return Math.exp(-GAMMA / (2.0 * M) * t) * Math.cos(omega * t);
    }

    // ── Euler ────────────────────────────────────────────────────────────────

    static double[] runEuler(double dt) {
        int steps = (int) Math.round(TF / dt);
        double[] pos = new double[steps + 1];
        double r = R0, v = V0;
        pos[0] = r;
        for (int i = 1; i <= steps; i++) {
            double a = force(r, v) / M;
            double rNew = r + v * dt + 0.5 * a * dt * dt;
            double vNew = v + a * dt;
            r = rNew;
            v = vNew;
            pos[i] = r;
        }
        return pos;
    }

    // ── Verlet original ──────────────────────────────────────────────────────
    // Uses backward-difference v(t) ≈ [r(t) - r(t-dt)]/dt to evaluate the
    // velocity-dependent force; r(t-dt) is seeded via reversed Euler.

    static double[] runVerlet(double dt) {
        int steps = (int) Math.round(TF / dt);
        double[] pos = new double[steps + 1];
        double r = R0, v = V0;
        double a0 = force(r, v) / M;
        // r at t = -dt from reversed Euler
        double rPrev = r - v * dt + 0.5 * a0 * dt * dt;
        pos[0] = r;

        for (int i = 1; i <= steps; i++) {
            double vBack = (r - rPrev) / dt;       // backward-difference velocity
            double a = force(r, vBack) / M;
            double rNext = 2.0 * r - rPrev + a * dt * dt;
            rPrev = r;
            r = rNext;
            pos[i] = r;
        }
        return pos;
    }

    // ── Beeman ───────────────────────────────────────────────────────────────
    // For velocity-dependent forces: predict v, evaluate force, correct v.

    static double[] runBeeman(double dt) {
        int steps = (int) Math.round(TF / dt);
        double[] pos = new double[steps + 1];
        double r = R0, v = V0;
        double a = force(r, v) / M;
        // Seed a(t-dt) via reversed Euler
        double rPrev = r - v * dt + 0.5 * a * dt * dt;
        double vPrev = v - a * dt;
        double aPrev = force(rPrev, vPrev) / M;
        pos[0] = r;

        for (int i = 1; i <= steps; i++) {
            double rNext = r + v * dt + (2.0 / 3.0) * a * dt * dt - (1.0 / 6.0) * aPrev * dt * dt;
            // Predict velocity for force evaluation
            double vPred = v + (3.0 / 2.0) * a * dt - (0.5) * aPrev * dt;
            double aNext = force(rNext, vPred) / M;
            double vNext = v + (1.0 / 3.0) * aNext * dt + (5.0 / 6.0) * a * dt - (1.0 / 6.0) * aPrev * dt;
            aPrev = a;
            a = aNext;
            r = rNext;
            v = vNext;
            pos[i] = r;
        }
        return pos;
    }

    // ── Gear Predictor-Corrector order 5 ────────────────────────────────────
    // Stores scaled derivatives: c[q] = r^(q)(t) * dt^q / q!
    // Predictor: Pascal-triangle expansion.
    // Corrector: c[q] += alpha[q] * deltaR2, where deltaR2 = a_actual*dt^2/2 - c[2].

    static double[] runGearPC(double dt) {
        int steps = (int) Math.round(TF / dt);
        double[] pos = new double[steps + 1];

        // Compute initial derivatives for f = -k*r - gamma*v
        double r0  = R0, r1 = V0;
        double r2  = force(r0, r1) / M;
        double r3  = (-K * r1 - GAMMA * r2) / M;
        double r4  = (-K * r2 - GAMMA * r3) / M;
        double r5  = (-K * r3 - GAMMA * r4) / M;

        // Scale to c[q] = r^(q) * dt^q / q!
        double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
        double[] c = {
            r0,
            r1 * dt,
            r2 * dt2 / 2.0,
            r3 * dt3 / 6.0,
            r4 * dt4 / 24.0,
            r5 * dt5 / 120.0
        };
        pos[0] = c[0];

        for (int i = 1; i <= steps; i++) {
            // ─ Predict (Pascal triangle) ─
            double[] cp = new double[6];
            cp[0] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5];
            cp[1] =        c[1] + 2*c[2] + 3*c[3] + 4*c[4] + 5*c[5];
            cp[2] =               c[2] + 3*c[3] + 6*c[4] + 10*c[5];
            cp[3] =                      c[3] + 4*c[4] + 10*c[5];
            cp[4] =                             c[4] + 5*c[5];
            cp[5] =                                    c[5];

            // ─ Evaluate ─
            double rPred = cp[0];
            double vPred = cp[1] / dt;               // cp[1] = v * dt  →  v = cp[1]/dt
            double aActual  = force(rPred, vPred) / M;
            double deltaR2  = aActual * dt2 / 2.0 - cp[2];

            // ─ Correct ─
            for (int q = 0; q < 6; q++) {
                c[q] = cp[q] + GEAR_ALPHA[q] * deltaR2;
            }
            pos[i] = c[0];
        }
        return pos;
    }

    // ── MSE ──────────────────────────────────────────────────────────────────

    static double mse(double[] pos, double dt) {
        double sum = 0;
        for (int i = 0; i < pos.length; i++) {
            double diff = pos[i] - analytical(i * dt);
            sum += diff * diff;
        }
        return sum / pos.length;
    }

    // ── Main ─────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws IOException {
        String outDir = (args.length > 0) ? args[0] : "output/system1";
        new File(outDir).mkdirs();

        // 1.2 – Trajectories at a reference dt
        double dtRef = 1e-3;
        writeTrajectories(outDir + "/trajectories.csv", dtRef);

        // 1.3 – MSE vs dt (log-log convergence study)
        double[] dtValues = {1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5};
        writeMSE(outDir + "/mse_vs_dt.csv", dtValues);

        System.out.println("System 1 done → " + outDir);
    }

    static void writeTrajectories(String path, double dt) throws IOException {
        int steps = (int) Math.round(TF / dt);
        double[] euler  = runEuler(dt);
        double[] verlet = runVerlet(dt);
        double[] beeman = runBeeman(dt);
        double[] gear   = runGearPC(dt);

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("time,r_euler,r_verlet,r_beeman,r_gear,r_analytical");
            for (int i = 0; i <= steps; i++) {
                double t = i * dt;
                pw.printf(Locale.US, "%.8f,%.10f,%.10f,%.10f,%.10f,%.10f%n",
                        t, euler[i], verlet[i], beeman[i], gear[i], analytical(t));
            }
        }
        System.out.printf(Locale.US,
                "dt=%.1e | MSE: Euler=%.3e  Verlet=%.3e  Beeman=%.3e  Gear=%.3e%n",
                dt, mse(euler, dt), mse(verlet, dt), mse(beeman, dt), mse(gear, dt));
    }

    static void writeMSE(String path, double[] dtValues) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("dt,mse_euler,mse_verlet,mse_beeman,mse_gear");
            for (double dt : dtValues) {
                double mseE = mse(runEuler(dt),  dt);
                double mseV = mse(runVerlet(dt), dt);
                double mseB = mse(runBeeman(dt), dt);
                double mseG = mse(runGearPC(dt), dt);
                pw.printf(Locale.US, "%.2e,%.8e,%.8e,%.8e,%.8e%n", dt, mseE, mseV, mseB, mseG);
                System.out.printf(Locale.US,
                        "dt=%.1e | MSE: Euler=%.3e  Verlet=%.3e  Beeman=%.3e  Gear=%.3e%n",
                        dt, mseE, mseV, mseB, mseG);
            }
        }
    }
}
