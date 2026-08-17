# xtask - swiss army knife builder

This crate is not published and is only used by developers.

It automates a number of tasks that a project maintainer might need to do, for example
code generation.

You can run `cargo xtask -h` to get a list of supported commands.

## Current commands

### `generate-fbs`

This regenerates the checked-in FlatBuffers bindings under `src/generated` in each crate that owns
`.fbs` schemas. It must be run every time changes are made to one of the .fbs files, or if any are
added/deleted. It requires the `flatc` compiler on `PATH`, or its location in `FLATC`.

### `generate-proto`

This regenerates the checked-in Protocol Buffers bindings under `src/generated` in each crate that
owns `.proto` schemas. It must be run every time changes are made to one of the .proto files, or if
any are added/deleted. Parsing is done in pure Rust, so no `protoc` install is needed.

## Adding a schema

`src/schemas.rs` lists which crate owns which schemas, and which other crates' schema directories
each one includes from. A schema that grows a cross-crate `include`/`import` needs the owning crate
added to its `depends_on` list there.
