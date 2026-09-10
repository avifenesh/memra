#!/bin/sh
# Red-arm demonstration, five injections, one per way the context could misresolve.
# Each injection is applied alone, the tests are run, and the tree is restored.
set -eu
cd /data/ctxlane
export PATH=/data/cargo/bin:$PATH RUSTUP_HOME=/data/rustup CARGO_HOME=/data/cargo
W=crates/memra-server/src/worker.rs
D=crates/memra-server/src/dsv4_serve.rs

inject() { # $1 = file, $2 = python snippet
  cp "$1" "/tmp/$(basename "$1").green"
  python3 -c "$2"
}
restore() { cp "/tmp/$(basename "$1").green" "$1"; }

run() { cargo test -j 24 -p memra-server --lib -- --test-threads 8 "$1" 2>&1 | grep -E "^test |^test result|panicked at|left:|right:|assertion"; }

echo "=== INJECTION 1: the pre-fix literal, unset resolves to 8192"
inject $W "
p='$W'; s=open(p).read()
old='''        None => {
            if model_ctx == 0 {'''
new='''        None => {
            if false {'''
assert s.count(old)==1
old2='''            Ok(model_ctx)
        }
    }
}'''
new2='''            Ok(8192) // RED ARM
        }
    }
}'''
assert s.count(old2)==1
open(p,'w').write(s.replace(old,new,1).replace(old2,new2,1))
"
run ctx || true
run declared_context_tests || true
restore $W

echo "=== INJECTION 2: an undeclared window silently acquires a default"
inject $W "
p='$W'; s=open(p).read()
old='''            if model_ctx == 0 {
                return Err('''
new='''            if false {
                return Err('''
assert s.count(old)==1
old2='''            Ok(model_ctx)
        }
    }
}'''
new2='''            Ok(if model_ctx == 0 { 8192 } else { model_ctx }) // RED ARM
        }
    }
}'''
assert s.count(old2)==1
open(p,'w').write(s.replace(old,new,1).replace(old2,new2,1))
"
run declared_context_tests || true
restore $W

echo "=== INJECTION 3: a beyond-ceiling declaration is clamped instead of refused"
inject $W "
p='$W'; s=open(p).read()
old='''            if model_ctx > ENGINE_MAX_CTX {
                return Err(format!('''
new='''            if false {
                return Err(format!('''
assert s.count(old)==1
old2='''            Ok(model_ctx)
        }
    }
}'''
new2='''            Ok(model_ctx.min(ENGINE_MAX_CTX)) // RED ARM: silent narrowing
        }
    }
}'''
assert s.count(old2)==1
open(p,'w').write(s.replace(old,new,1).replace(old2,new2,1))
"
run declared_context_tests || true
restore $W

echo "=== INJECTION 4: a present but unusable declaration is treated as absent"
inject $D "
p='$D'; s=open(p).read()
old='''        Some(0) | None => DeclaredContext::Unusable(value.to_string()),'''
new='''        Some(0) | None => DeclaredContext::Absent, // RED ARM'''
assert s.count(old)==1
open(p,'w').write(s.replace(old,new,1))
"
run declared_context_tests || true
restore $D

echo "=== INJECTION 5: the config read is swallowed"
inject $D "
p='$D'; s=open(p).read()
old='''    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!(\"dsv4 config {} unreadable: {e}\", path.display()))?;'''
new='''    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| \"{}\".to_string()); // RED ARM'''
assert s.count(old)==1
open(p,'w').write(s.replace(old,new,1))
"
run declared_context_tests || true
restore $D

echo "=== RESTORED, green re-run"
run ctx
run declared_context_tests
