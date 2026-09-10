#![no_main]
//! libFuzzer entry for the frozen v7.5.6 compat DTO decode harness.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    faktor_tests_fuzz_seeds::fuzz_entry::no_panic_compat_dto(data);
});
