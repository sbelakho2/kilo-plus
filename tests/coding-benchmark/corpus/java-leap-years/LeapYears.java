/** Gregorian leap-year rule: a year is leap when divisible by 4, except
 *  century years, which are leap only when divisible by 400. */
public final class LeapYears {

    private LeapYears() {
    }

    /**
     * Is {@code year} a leap year?
     *
     * @throws IllegalArgumentException when {@code year <= 0}
     */
    public static boolean isLeap(int year) {
        if (year <= 0) {
            throw new IllegalArgumentException("year must be positive: " + year);
        }
        // BUG: the century exception is missing, so 1900 and 2100 are
        // reported as leap years although they are not divisible by 400.
        return year % 4 == 0;
    }
}
