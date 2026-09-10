#![no_main]
//! libFuzzer entry for the tool-call JSON repair/parse harness.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    faktor_tests_fuzz_seeds::fuzz_entry::no_panic_tool_json(data);
});
