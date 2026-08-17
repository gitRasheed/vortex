// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::schemas::PROTO;
use crate::schemas::SchemaCrate;

pub fn generate_proto() -> anyhow::Result<()> {
    for schema_crate in PROTO {
        generate(schema_crate)?;
    }
    Ok(())
}

fn generate(schema_crate: &SchemaCrate) -> anyhow::Result<()> {
    let schema_dir = schema_crate.schema_dir("proto");
    let out_dir = schema_crate.generated_dir();
    std::fs::create_dir_all(&out_dir)?;

    for schema in schema_crate.schemas {
        let path = schema_dir.join(schema);
        if !path.exists() {
            anyhow::bail!("schema not found: {path:?}");
        }
    }

    // `protox` parses the schemas in pure Rust, so regenerating needs no `protoc` install.
    let file_descriptors =
        protox::compile(schema_crate.schemas, schema_crate.include_dirs("proto"))?;

    prost_build::Config::new()
        .out_dir(out_dir)
        .compile_fds(file_descriptors)?;

    Ok(())
}
