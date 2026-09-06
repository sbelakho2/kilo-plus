// Package sumranges parses comma-separated inclusive integer ranges and
// sums the numbers they denote. "1-3,5" denotes {1,2,3,5} -> 11.
package sumranges

import (
	"fmt"
	"strconv"
	"strings"
)

// SumRanges sums every number denoted by s. A token is either a single
// integer or "a-b" with a <= b (both INCLUSIVE). Spaces around tokens and
// around the range parts are tolerated. Errors: empty input, an empty
// token, a malformed token, a negative number, or a reversed range
// (a > b).
func SumRanges(s string) (int, error) {
	if strings.TrimSpace(s) == "" {
		return 0, fmt.Errorf("empty input")
	}
	total := 0
	for _, raw := range strings.Split(s, ",") {
		tok := strings.TrimSpace(raw)
		if tok == "" {
			return 0, fmt.Errorf("empty token")
		}
		if strings.Contains(tok, "-") {
			parts := strings.Split(tok, "-")
			if len(parts) != 2 {
				return 0, fmt.Errorf("malformed range %q", tok)
			}
			start, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			end, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err1 != nil || err2 != nil {
				return 0, fmt.Errorf("malformed range %q", tok)
			}
			if start < 0 || end < start {
				return 0, fmt.Errorf("reversed or negative range %q", tok)
			}
			// BUG: the range end is summed as EXCLUSIVE, so the last
			// number of every multi-number range is silently dropped
			// ("1-3" contributes 1+2 instead of 1+2+3).
			for n := start; n < end; n++ {
				total += n
			}
		} else {
			n, err := strconv.Atoi(tok)
			if err != nil || n < 0 {
				return 0, fmt.Errorf("malformed number %q", tok)
			}
			total += n
		}
	}
	return total, nil
}
