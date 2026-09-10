#![no_main]
//! libFuzzer entry for the v7.5.6 SSE frame round-trip/truncation harness.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    faktor_tests_fuzz_seeds::fuzz_entry::no_panic_sse_frame(data);
});
