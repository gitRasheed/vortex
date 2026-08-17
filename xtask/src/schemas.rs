// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

/// A crate that owns schema definitions, and the crates whose definitions those schemas reference.
///
/// Dependencies are declared rather than discovered, so adding an `include`/`import` that crosses a
/// crate boundary means adding the owning crate to `depends_on` here.
pub struct SchemaCrate {
    /// The owning crate's directory, relative to the workspace root.
    pub krate: &'static str,
    /// The schemas to compile, relative to the crate's schema directory.
    pub schemas: &'static [&'static str],
    /// Crates whose schema directories must be on the include path.
    pub depends_on: &'static [&'static str],
}

impl SchemaCrate {
    /// The directory holding this crate's schemas.
    pub fn schema_dir(&self, kind: &str) -> PathBuf {
        workspace_root().join(self.krate).join(kind)
    }

    /// The directory generated bindings are written to.
    pub fn generated_dir(&self) -> PathBuf {
        workspace_root()
            .join(self.krate)
            .join("src")
            .join("generated")
    }

    /// The schema directories to search when resolving includes, ours first.
    pub fn include_dirs(&self, kind: &str) -> Vec<PathBuf> {
        std::iter::once(self.schema_dir(kind))
            .chain(
                self.depends_on
                    .iter()
                    .map(|krate| workspace_root().join(krate).join(kind)),
            )
            .collect()
    }
}

/// FlatBuffers schemas, keyed by the crate that owns them.
pub const FLATBUFFERS: &[SchemaCrate] = &[
    SchemaCrate {
        krate: "vortex-array",
        schemas: &["vortex-array/array.fbs", "vortex-dtype/dtype.fbs"],
        depends_on: &[],
    },
    SchemaCrate {
        krate: "vortex-layout",
        schemas: &["vortex-layout/layout.fbs"],
        depends_on: &[],
    },
    SchemaCrate {
        krate: "vortex-file",
        schemas: &["vortex-file/footer.fbs"],
        depends_on: &["vortex-array", "vortex-layout"],
    },
    SchemaCrate {
        krate: "vortex-ipc",
        schemas: &["vortex-serde/message.fbs"],
        depends_on: &["vortex-array"],
    },
];

/// Protocol Buffers schemas, keyed by the crate that owns them.
pub const PROTO: &[SchemaCrate] = &[SchemaCrate {
    krate: "vortex-array",
    schemas: &["dtype.proto", "scalar.proto", "expr.proto"],
    depends_on: &[],
}];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}
