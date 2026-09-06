package sumranges

import "testing"

func TestSingleNumber(t *testing.T) {
	got, err := SumRanges("5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 5 {
		t.Errorf(`SumRanges("5") = %d, want 5`, got)
	}
}

func TestInclusiveRange(t *testing.T) {
	got, err := SumRanges("1-3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 6 {
		t.Errorf(`SumRanges("1-3") = %d, want 6 (1+2+3)`, got)
	}
}

func TestMixedTokens(t *testing.T) {
	got, err := SumRanges("1-3,5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 11 {
		t.Errorf(`SumRanges("1-3,5") = %d, want 11`, got)
	}
}

func TestSingleNumberRangeIsInclusive(t *testing.T) {
	got, err := SumRanges("7-7")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 7 {
		t.Errorf(`SumRanges("7-7") = %d, want 7`, got)
	}
}

func TestSurroundingSpacesTolerated(t *testing.T) {
	got, err := SumRanges(" 1-2 , 4 ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 7 {
		t.Errorf(`SumRanges(" 1-2 , 4 ") = %d, want 7`, got)
	}
}

func TestErrors(t *testing.T) {
	cases := []string{"", "abc", "1,,2", "5-2", "-3", "1-2-3"}
	for _, s := range cases {
		if _, err := SumRanges(s); err == nil {
			t.Errorf("SumRanges(%q): expected an error, got none", s)
		}
	}
}
