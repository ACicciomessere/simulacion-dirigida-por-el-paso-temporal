package ar.edu.itba.sds.system2;

import java.io.*;

/**
 * Entry point for System 2.
 *
 * Usage (via Maven):
 *   mvn exec:java -Dexec.mainClass=ar.edu.itba.sds.system2.System2Main \
 *       -Dexec.args="-N 300 -dt 0.001 -dt2 1.0 -tf 500 -k 1000 -seeds 42,43,44 -out output/system2"
 *
 * Or via the JAR:
 *   java -cp target/tp4-molecular-dynamics-1.0-SNAPSHOT.jar \
 *       ar.edu.itba.sds.system2.System2Main [options]
 *
 * Options:
 *   -N       <int>        number of particles (default 200)
 *   -dt      <double>     integration step   (default 0.001 s)
 *   -dt2     <double>     output step        (default 1.0 s)
 *   -tf      <double>     total time         (default 500 s)
 *   -k       <double>     elastic constant   (default 1000 N/m)
 *   -seeds   <s1,s2,...>  random seeds       (default 42)
 *   -out     <dir>        base output dir    (default output/system2)
 *
 * Multi-N sweep (for task 1.1):
 *   -Nlist   <n1,n2,...>  run for each N, one seed, writes timing.csv
 */
public class System2Main {

    public static void main(String[] args) throws IOException {

        // ── Defaults ────────────────────────────────────────────────────────
        int     N       = 200;
        double  dt      = 0.001;
        double  dt2     = 1.0;
        double  tf      = 500.0;
        double  k       = 1e3;
        long[]  seeds   = {42L};
        String  outBase = "output/system2";
        int[]   Nlist   = null;

        // ── Parse ────────────────────────────────────────────────────────────
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-N":     N       = Integer.parseInt(args[++i]);     break;
                case "-dt":    dt      = Double.parseDouble(args[++i]);   break;
                case "-dt2":   dt2     = Double.parseDouble(args[++i]);   break;
                case "-tf":    tf      = Double.parseDouble(args[++i]);   break;
                case "-k":     k       = Double.parseDouble(args[++i]);   break;
                case "-out":   outBase = args[++i];                        break;
                case "-seeds": seeds   = parseLongs(args[++i]);           break;
                case "-Nlist": Nlist   = parseInts(args[++i]);            break;
                default:       break;
            }
        }

        new File(outBase).mkdirs();

        if (Nlist != null) {
            // ── Timing sweep 1.1 ──────────────────────────────────────
            runTimingSweep(Nlist, dt, dt2, tf, k, seeds[0], outBase);
        } else {
            runRealizations(N, dt, dt2, tf, k, seeds, outBase);
        }
    }

    // ── Multiple realizations for one N ─────────────────────────────────────

    static void runRealizations(int N, double dt, double dt2, double tf, double k,
                                 long[] seeds, String outBase) throws IOException {
        String dir = outBase + "/N" + N + "_k" + formatK(k);
        new File(dir).mkdirs();

        System.out.printf("Running N=%d, k=%.0f, %d realization(s)%n", N, k, seeds.length);

        for (long seed : seeds) {
            String runDir = dir + "/seed" + seed;
            MDSimulation sim = new MDSimulation(N, k, dt, dt2, tf, seed);
            long ms = sim.run(runDir);
            System.out.printf("  seed=%-8d  %.2f s%n", seed, ms / 1000.0);
        }
    }

    // ── Timing sweep: one seed, varying N ───────────────────────────────────

    static void runTimingSweep(int[] Nlist, double dt, double dt2, double tf, double k,
                                long seed, String outBase) throws IOException {
        String timingFile = outBase + "/timing_k" + formatK(k) + ".csv";
        System.out.printf("Timing sweep: k=%.0f, seed=%d%n", k, seed);

        try (PrintWriter pw = new PrintWriter(new FileWriter(timingFile))) {
            pw.println("N,elapsed_ms");
            for (int n : Nlist) {
                String runDir = outBase + "/timing/N" + n + "_k" + formatK(k);
                MDSimulation sim = new MDSimulation(n, k, dt, dt2, tf, seed);
                long ms = sim.run(runDir);
                pw.printf("%d,%d%n", n, ms);
                System.out.printf("  N=%-5d  %.2f s%n", n, ms / 1000.0);
            }
        }
        System.out.println("Timing written → " + timingFile);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    static String formatK(double k) {
        long ki = Math.round(k);
        return String.valueOf(ki);
    }

    static long[] parseLongs(String csv) {
        String[] parts = csv.split(",");
        long[] arr = new long[parts.length];
        for (int i = 0; i < parts.length; i++) arr[i] = Long.parseLong(parts[i].trim());
        return arr;
    }

    static int[] parseInts(String csv) {
        String[] parts = csv.split(",");
        int[] arr = new int[parts.length];
        for (int i = 0; i < parts.length; i++) arr[i] = Integer.parseInt(parts[i].trim());
        return arr;
    }
}
