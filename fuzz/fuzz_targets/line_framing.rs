#![no_main]
//! libFuzzer entry for the provider line-framing harness (P0-75).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    faktor_tests_fuzz_seeds::fuzz_entry::no_panic_provider_line_framing(data);
});
