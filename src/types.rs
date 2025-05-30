use cgmath::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, view_proj: cgmath::Matrix4<f32>) {
        self.view_proj = view_proj.into();
    }
}

// Cube vertices with colors
pub const VERTICES: &[Vertex] = &[
    // Front face (red)
    Vertex { position: [-1.0, -1.0,  1.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 1.0, -1.0,  1.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 1.0,  1.0,  1.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-1.0,  1.0,  1.0], color: [1.0, 0.0, 0.0] },
    
    // Back face (green)
    Vertex { position: [-1.0, -1.0, -1.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 1.0, -1.0, -1.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 1.0,  1.0, -1.0], color: [0.0, 1.0, 0.0] },
    Vertex { position: [-1.0,  1.0, -1.0], color: [0.0, 1.0, 0.0] },
];

pub const INDICES: &[u16] = &[
    // Front face
    0, 1, 2,  2, 3, 0,
    // Back face
    4, 6, 5,  6, 4, 7,
    // Left face
    4, 0, 3,  3, 7, 4,
    // Right face
    1, 5, 6,  6, 2, 1,
    // Top face
    3, 2, 6,  6, 7, 3,
    // Bottom face
    4, 5, 1,  1, 0, 4,
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub position: [f32; 3],
    pub color: [f32; 3], 
    pub scale: f32,
    pub _padding: f32, // Ensure 16-byte alignment for GPU
}

impl InstanceData {
    pub fn new(position: [f32; 3], color: [f32; 3], scale: f32) -> Self {
        Self {
            position,
            color,
            scale,
            _padding: 0.0,
        }
    }
    
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Instance position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Instance color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Instance scale
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
} 