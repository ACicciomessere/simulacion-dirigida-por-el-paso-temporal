package ar.edu.itba.sds.system2;

public class Particle {
    public final int id;
    public double x, y;
    public double vx, vy;
    public double ax, ay;      
    public final double radius;
    public final double mass;
    public boolean isUsed;      // false = fresh (green), true = used (red)

    public Particle(int id, double x, double y, double vx, double vy, double radius, double mass) {
        this.id     = id;
        this.x      = x;
        this.y      = y;
        this.vx     = vx;
        this.vy     = vy;
        this.radius = radius;
        this.mass   = mass;
        this.isUsed = false;
    }

    public double distFromOrigin() {
        return Math.sqrt(x * x + y * y);
    }

    public double speedSq() {
        return vx * vx + vy * vy;
    }
}
