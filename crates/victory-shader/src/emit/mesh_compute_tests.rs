//! A mesh program run as compute ([`super::translate_mesh_as_compute`]):
//! a synthetic GFX10 mesh program lowered both ways, and the compute form
//! dispatched on the host GPU (and WARP) to read back what it wrote.

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::container::SigElement;
use super::host::Host;
use super::{HostShader, MESH_OUT_SPACE, MeshLayout, lower, validator};
use crate::native::{HwStage, MeshIo, NativeProgram, StageIo, Usage, VsOutput};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constants as operands.
const ZERO: u16 = 0x80;
const MINUS_ONE: u16 = 0xC1;
const ONE_F: u16 = 0xF2;
const TWO_F: u16 = 0xF4;
/// A literal dword follows the instruction.
const LITERAL: u16 = 0xFF;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u16) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | (src0 as u32 & 0xFF)
}

/// VOP1 `op vdst, src0` (`src0`: below 256 SGPRs and constants, 256 on VGPRs).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// `exp target, v[srcs]` with the channels `srcs` names enabled.
fn exp(target: u8, srcs: [Option<u8>; 4], done: bool) -> [u32; 2] {
    let mut en = 0;
    let mut w1 = 0;
    for (k, s) in srcs.iter().enumerate() {
        if let Some(v) = s {
            en |= 1 << k;
            w1 |= u32::from(*v) << (8 * k);
        }
    }
    [
        0xF800_0000 | u32::from(done) << 11 | u32::from(target) << 4 | en,
        w1,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// One triangle per group from three lanes. Lane n exports vertex n at
/// (n, 2, 0, 1) with `TEXCOORD0` (n, 1, 2, 0), and primitive n: lane 0
/// the triangle (0, 1, 2), lanes 1 and 2 culled ones.
fn program() -> NativeProgram {
    let mut w = Vec::new();
    w.extend(vop3(0x365, 0, MINUS_ONE, ZERO, 0)); // v_mbcnt_lo_u32_b32 v0, -1, 0: the lane
    w.push(vop1(0x006, 1, vgpr(0))); // v_cvt_f32_u32 v1, v0
    w.push(vop1(0x001, 2, TWO_F)); // v_mov_b32 v2, 2.0
    w.push(vop1(0x001, 3, ZERO)); // v_mov_b32 v3, 0
    w.push(vop1(0x001, 4, ONE_F)); // v_mov_b32 v4, 1.0
    // v6 = 0x00200400 + lane * 0x7FDFFC00: lane 0 the indices 0, 1, 2;
    // lanes 1 and 2 have bit 31 set (culled).
    w.extend([s_mov_b32(10, LITERAL), 0x7FDF_FC00]);
    w.extend([s_mov_b32(11, LITERAL), 0x0020_0400]);
    w.extend(vop3(0x169, 6, vgpr(0), 10, 0)); // v_mul_lo_u32 v6, v0, s10
    w.push(vop2(0x25, 6, 11, 6)); // v_add_nc_u32 v6, s11, v6
    w.extend(exp(20, [Some(6), None, None, None], true)); // exp prim
    w.extend(exp(32, [Some(1), Some(4), Some(2), Some(3)], false)); // exp param0
    w.extend(exp(12, [Some(1), Some(2), Some(3), Some(4)], true)); // exp pos0
    w.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            clip_cntl: 0,
            threads: 3,
            max_vertices: 3,
            max_primitives: 3,
            topology: 2,
            lds_bytes: 0,
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn outputs() -> Vec<SigElement> {
    let element = |name: &str, kind, interp, row| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    vec![
        element("SV_Position", 3, 4, 0),
        element("TEXCOORD", 0, 2, 1),
    ]
}

/// The program lowered as a mesh shader or, with `as_compute`, a compute
/// shader with its layout; validated when `dxil.dll` is there.
fn lowered(as_compute: bool) -> (HostShader, MeshLayout) {
    let native = program();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let outputs = outputs();
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: outputs.clone(),
        prims: Vec::new(),
        mesh_stride: 3,
        mesh_as_compute: as_compute,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Ms,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let layout = MeshLayout {
        threads: 3,
        max_vertices: 3,
        max_primitives: 3,
        vertex_dwords: 4 * outputs.len() as u32,
        primitive_dwords: 0,
    };
    (shader, layout)
}

/// The program kind in the DXIL part's version word (5 compute, 13 mesh).
fn program_kind(container: &[u8]) -> u32 {
    let c = crate::dxbc::Container::parse(container).expect("container");
    let dxil = c.part(crate::dxbc::FourCC::DXIL).expect("DXIL part");
    u32::from_le_bytes(dxil.data[0..4].try_into().unwrap()) >> 16
}

#[test]
fn a_mesh_program_lowers_as_compute_too() {
    let (mesh, _) = lowered(false);
    let (cs, layout) = lowered(true);
    assert_eq!(program_kind(&mesh.dxil), 13);
    assert_eq!(program_kind(&cs.dxil), 5);
    assert_eq!(layout.bytes(2), [2 * 3 * 8 * 4, 2 * 3 * 12, 0]);
}

/// Runs the compute form for `groups` groups and returns `u0` and `u1`.
fn run(
    host: &Host,
    cs: &HostShader,
    layout: &MeshLayout,
    groups: u32,
) -> Result<[Vec<u32>; 2], String> {
    let err = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    let sizes = layout.bytes(groups).map(|b| b.max(4));
    let params: Vec<D3D12_ROOT_PARAMETER1> = (0..3)
        .map(|r| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: r,
                    RegisterSpace: MESH_OUT_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        })
        .collect();
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                ..Default::default()
            },
        },
    };
    // SAFETY: D3D12 calls with descriptions that outlive them; the resources
    // stay alive until the fence says the GPU is done.
    unsafe {
        let mut blob = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(err)?;
        let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).map_err(err)?;
        let pso = host.compute_pipeline_with(&root, &cs.dxil)?;
        let buffer = |heap, size: u64, flags, state| -> Result<ID3D12Resource, String> {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut r,
            )
            .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let mut outs = Vec::new();
        let mut readbacks = Vec::new();
        for &size in &sizes {
            outs.push(buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                size,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
            readbacks.push(buffer(
                D3D12_HEAP_TYPE_READBACK,
                size,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?);
        }
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        list.SetComputeRootSignature(&root);
        for (k, out) in outs.iter().enumerate() {
            list.SetComputeRootUnorderedAccessView(k as u32, out.GetGPUVirtualAddress());
        }
        list.Dispatch(groups, 1, 1);
        for (out, readback) in outs.iter().zip(&readbacks) {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(out),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }]);
            list.CopyResource(readback, out);
        }
        list.Close().map_err(err)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            std::thread::yield_now();
        }
        let read = |r: &ID3D12Resource, size: u64| -> Result<Vec<u32>, String> {
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).map_err(err)?;
            let v = std::slice::from_raw_parts(p.cast::<u32>(), size as usize / 4).to_vec();
            r.Unmap(0, None);
            Ok(v)
        };
        Ok([
            read(&readbacks[0], sizes[0])?,
            read(&readbacks[1], sizes[1])?,
        ])
    }
}

#[test]
fn a_mesh_program_run_as_compute_writes_its_vertices_and_triangles() {
    // One test at a time on the devices (see `valu_tests::run_cases`).
    let (cs, layout) = lowered(true);
    let groups = 2;
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let [vertices, indices] =
            run(&host, &cs, &layout, groups).unwrap_or_else(|e| panic!("{label}: {e}"));
        let f = f32::to_bits;
        let mut want_vertices = Vec::new();
        let mut want_indices = Vec::new();
        for g in 0..groups {
            for lane in 0..3u32 {
                let n = lane as f32;
                // SV_Position, then TEXCOORD0.
                want_vertices.extend([f(n), f(2.0), f(0.0), f(1.0)]);
                want_vertices.extend([f(n), f(1.0), f(2.0), f(0.0)]);
                want_indices.extend(match lane {
                    0 => [3 * g, 3 * g + 1, 3 * g + 2],
                    _ => [0, 0, 0],
                });
            }
        }
        assert_eq!(vertices, want_vertices, "{label}: vertices");
        assert_eq!(indices, want_indices, "{label}: indices");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
