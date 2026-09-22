pub struct ExactScope<'a> {
    flag: &'a std::sync::atomic::AtomicBool,
    prev: bool,
}

impl<'a> ExactScope<'a> {
    pub(crate) fn set(flag: &'a std::sync::atomic::AtomicBool, on: bool) -> Self {
        let prev = flag.load(std::sync::atomic::Ordering::Relaxed);
        flag.store(on, std::sync::atomic::Ordering::Relaxed);
        ExactScope { flag, prev }
    }
}

impl Drop for ExactScope<'_> {
    fn drop(&mut self) {
        self.flag
            .store(self.prev, std::sync::atomic::Ordering::Relaxed);
    }
}


#[cfg(test)]
mod exact_scope_tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn error_path_restores_verify_exact() {
        // TOOTH (hermes finding, fixed 2026-08-23): dspark_spec_session_burst called
        // set_verify_exact(true)/(false) manually with `?`s in between — any error left
        // the engine latched in the decode-exact matmul program for every later request.
        // The RAII scope must restore across an error propagation.
        let flag = AtomicBool::new(false);
        let failing = |flag: &AtomicBool| -> Result<(), &'static str> {
            let _scope = super::ExactScope::set(flag, true);
            assert!(flag.load(Ordering::Relaxed), "scope arms the flag");
            Err("draft forward failed")? // the `?` exit the manual pair leaked on
        };
        assert!(failing(&flag).is_err());
        assert!(
            !flag.load(Ordering::Relaxed),
            "error propagation must restore the pre-scope value"
        );
        // Nested/previous-value contract: a scope entered while already ON restores ON.
        let flag = AtomicBool::new(true);
        {
            let _scope = super::ExactScope::set(&flag, true);
        }
        assert!(flag.load(Ordering::Relaxed));
        // Early drop ends the scope exactly where the manual `false` used to sit.
        let flag = AtomicBool::new(false);
        let scope = super::ExactScope::set(&flag, true);
        drop(scope);
        assert!(!flag.load(Ordering::Relaxed));
    }
}
