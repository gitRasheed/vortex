// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use xshell::Shell;
use xshell::cmd;

use crate::schemas::FLATBUFFERS;
use crate::schemas::SchemaCrate;

/// The FlatBuffers compiler. Override with `FLATC` to use a build that is not on `PATH`.
fn flatc() -> String {
    std::env::var("FLATC").unwrap_or_else(|_| "flatc".to_string())
}

/// The Rust module path, relative to the owning crate's root, under which generated code finds the
/// schemas it includes from other crates.
///
/// Crates with cross-crate includes provide the corresponding `deps` module by hand; see
/// `vortex-ipc/src/flatbuffers.rs`.
const INCLUDE_PREFIX: &str = "flatbuffers::deps";

pub fn generate_fbs() -> anyhow::Result<()> {
    let sh = Shell::new()?;
    for schema_crate in FLATBUFFERS {
        generate(&sh, schema_crate)?;
    }
    Ok(())
}

fn generate(sh: &Shell, schema_crate: &SchemaCrate) -> anyhow::Result<()> {
    let schema_dir = schema_crate.schema_dir("flatbuffers");
    let out_dir = schema_crate.generated_dir();
    std::fs::create_dir_all(&out_dir)?;

    let mut schemas = Vec::with_capacity(schema_crate.schemas.len());
    for schema in schema_crate.schemas {
        let path = schema_dir.join(schema);
        if !path.exists() {
            anyhow::bail!("schema not found: {path:?}");
        }
        schemas.push(path);
    }

    let includes: Vec<_> = schema_crate
        .include_dirs("flatbuffers")
        .into_iter()
        .flat_map(|dir| [Path::new("-I").to_path_buf(), dir])
        .collect();

    let flatc = flatc();
    cmd!(
        sh,
        "{flatc} --rust --filename-suffix '' --include-prefix {INCLUDE_PREFIX} {includes...} -o {out_dir} {schemas...}"
    )
    .run()?;

    Ok(())
}
