/** Dependency-free test runner (no JUnit): assertion helpers + main that
 *  exits non-zero on the first failing group, zero when everything passes. */
public final class TestLeapYears {

    private static int failures = 0;

    private static void check(String what, boolean actual, boolean expected) {
        if (actual != expected) {
            System.err.println("FAIL: " + what
                    + " (expected " + expected + ", got " + actual + ")");
            failures++;
        }
    }

    public static void main(String[] args) {
        check("1996 is a leap year", LeapYears.isLeap(1996), true);
        check("1997 is not a leap year", LeapYears.isLeap(1997), false);
        check("2004 is a leap year", LeapYears.isLeap(2004), true);
        check("2000 is a leap year (divisible by 400)", LeapYears.isLeap(2000), true);
        check("1900 is not a leap year (century without 400)", LeapYears.isLeap(1900), false);
        check("2100 is not a leap year (century without 400)", LeapYears.isLeap(2100), false);
        try {
            LeapYears.isLeap(0);
            System.err.println("FAIL: isLeap(0) must throw");
            failures++;
        } catch (IllegalArgumentException expected) {
            /* ok */
        }
        if (failures > 0) {
            System.err.println(failures + " assertion(s) failed");
            System.exit(1);
        }
        System.out.println("leap-year tests passed");
    }
}
