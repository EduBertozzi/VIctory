//! The vertex shader that draws what a mesh program run as compute wrote
//! ([`super::MeshLayout`]): vertex `SV_VertexID` is record `SV_VertexID` of
//! the vertex records (`t0` in [`MESH_OUT_SPACE`]), and each of its output
//! elements is that record's four dwords for the element, so the mesh
//! program's output signature (and the title's pixel shader linked to it)
//! stays as it is.

use super::bitcode::{Attr, Cast, Term};
use super::container::SigElement;
use super::shader::{MESH_OUT_SPACE, Shader};

/// The vertex shader for a mesh program with these output elements (in
/// signature order, as the program was lowered with) and `vertex_dwords`
/// dwords per record.
pub(crate) fn vertex_shader(outputs: &[SigElement], vertex_dwords: u32) -> Vec<u8> {
    let mut s = Shader::new(false);
    s.inputs = vec![SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    let e = s.entry;
    let i32t = s.t.i32;
    let args = [s.m.i32(0), s.m.i32(0), s.m.i8(0), s.m.undef(i32t)];
    let vertex = s.op(e, "loadInput.i32", Attr::ReadNone, i32t, 4, &args);
    let record = s.m.i32(vertex_dwords * 4);
    let base = s.m.bin(e, super::bitcode::Bin::Mul, vertex, record);
    let h = s.mesh_vertices_handle();
    for (id, el) in outputs.iter().enumerate() {
        for col in 0..el.cols {
            let at = s.m.i32((id as u32 * 4 + u32::from(col)) * 4);
            let off = s.m.bin(e, super::bitcode::Bin::Add, base, at);
            let (u, mask, al) = (s.m.undef(i32t), s.m.i8(1), s.m.i32(4));
            let r = s.op(
                e,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                s.t.res_ret,
                139,
                &[h, off, u, mask, al],
            );
            let mut x = s.m.extract(e, r, 0);
            let suffix = if el.comp == 9 {
                ".f32"
            } else {
                x = s.m.cast(e, Cast::Bitcast, x, i32t);
                ".i32"
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), x];
            let void = s.t.void;
            s.op(
                e,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                void,
                5,
                &args,
            );
        }
    }
    s.outputs = outputs
        .iter()
        .cloned()
        .map(|mut el| {
            el.usage = (1 << el.cols) - 1;
            el
        })
        .collect();
    s.m.terminate(e, Term::Ret);
    debug_assert_eq!(MESH_OUT_SPACE, super::MESH_OUT_SPACE);
    s.finish()
}
