mod utils;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, console, Performance};
use wgpu::util::DeviceExt;
use cgmath::prelude::*;
use std::collections::VecDeque;

// WASM-compatible performance timing
fn performance() -> Performance {
    web_sys::window()
        .expect("should have a window in this context")
        .performance()
        .expect("performance should be available")
}

fn now() -> f64 {
    performance().now()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
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

// Cube vertices with colors
const VERTICES: &[Vertex] = &[
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

const INDICES: &[u16] = &[
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
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, view_proj: cgmath::Matrix4<f32>) {
        self.view_proj = view_proj.into();
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct PerformanceSnapshot {
    pub timestamp: f64,        // Milliseconds since session start
    pub fps: f64,              // Current FPS
    pub frame_time_ms: f64,    // Average frame time in window
    pub min_frame_time: f64,   // Min frame time in window  
    pub max_frame_time: f64,   // Max frame time in window
    pub frame_count: u32,      // Total frames since start
}

struct PerformanceTracker {
    frame_times: VecDeque<(f64, f64)>, // (timestamp, duration) pairs in milliseconds
    session_start: f64,
    last_frame_start: Option<f64>,
    last_export: f64,
    total_frames: u32,
    
    // Export interval (100ms = 10Hz)
    export_interval: f64,
}

impl PerformanceTracker {
    fn new() -> Self {
        let now = now();
        Self {
            frame_times: VecDeque::new(),
            session_start: now,
            last_frame_start: None,
            last_export: now,
            total_frames: 0,
            export_interval: 100.0, // 100ms
        }
    }
    
    fn start_frame(&mut self) {
        self.last_frame_start = Some(now());
    }
    
    fn end_frame(&mut self) -> Option<PerformanceSnapshot> {
        if let Some(frame_start) = self.last_frame_start.take() {
            let frame_end = now();
            let frame_duration = frame_end - frame_start;
            
            // Add this frame to our rolling window
            self.frame_times.push_back((frame_start, frame_duration));
            self.total_frames += 1;
            
            // Remove frames older than 1 second for rolling calculation
            let one_second_ago = frame_end - 1000.0; // 1 second = 1000ms
            while let Some(&(timestamp, _)) = self.frame_times.front() {
                if timestamp < one_second_ago {
                    self.frame_times.pop_front();
                } else {
                    break;
                }
            }
            
            // Check if we should export a snapshot
            if frame_end - self.last_export >= self.export_interval {
                self.last_export = frame_end;
                return Some(self.create_snapshot(frame_end));
            }
        }
        None
    }
    
    fn create_snapshot(&self, now: f64) -> PerformanceSnapshot {
        if self.frame_times.is_empty() {
            return PerformanceSnapshot {
                timestamp: now - self.session_start,
                fps: 0.0,
                frame_time_ms: 0.0,
                min_frame_time: 0.0,
                max_frame_time: 0.0,
                frame_count: self.total_frames,
            };
        }
        
        // Calculate metrics from rolling window (last 1 second)
        let window_size = self.frame_times.len() as f64;
        let fps = window_size; // frames in 1 second = FPS
        
        let durations: Vec<f64> = self.frame_times
            .iter()
            .map(|(_, duration)| *duration)
            .collect();
        
        let avg_frame_time = durations.iter().sum::<f64>() / window_size;
        let min_frame_time = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_frame_time = durations.iter().fold(0.0f64, |a, &b| a.max(b));
        
        PerformanceSnapshot {
            timestamp: now - self.session_start,
            fps,
            frame_time_ms: avg_frame_time,
            min_frame_time,
            max_frame_time,
            frame_count: self.total_frames,
        }
    }
}

#[wasm_bindgen]
pub struct CubeRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    
    // Camera controls
    camera_distance: f32,
    camera_pan_x: f32,
    camera_pan_y: f32,
    camera_rotation_x: f32, // Pitch (up/down rotation)
    camera_rotation_y: f32, // Yaw (left/right rotation)
    
    // Background color
    background_color: wgpu::Color,
    
    // Performance tracking
    performance_tracker: PerformanceTracker,
}

#[wasm_bindgen]
impl CubeRenderer {
    fn parse_hex_color(hex: &str) -> Result<wgpu::Color, JsValue> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return Err(JsValue::from_str("Invalid hex color format. Expected #RRGGBB"));
        }
        
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| JsValue::from_str("Invalid hex color format"))?;
        
        Ok(wgpu::Color {
            r: r as f64 / 255.0,
            g: g as f64 / 255.0,
            b: b as f64 / 255.0,
            a: 1.0,
        })
    }

    #[wasm_bindgen(constructor)]
    pub async fn new(canvas: HtmlCanvasElement) -> Result<CubeRenderer, JsValue> {
        Self::new_with_background(canvas, "#dddddd").await
    }

    #[wasm_bindgen]
    pub async fn new_force_webgl(canvas: HtmlCanvasElement) -> Result<CubeRenderer, JsValue> {
        Self::new_force_webgl_with_background(canvas, "#dddddd").await
    }

    #[wasm_bindgen]
    pub async fn new_with_background(canvas: HtmlCanvasElement, background_color: &str) -> Result<CubeRenderer, JsValue> {
        Self::new_with_backend_and_background(canvas, false, background_color).await
    }

    #[wasm_bindgen]
    pub async fn new_force_webgl_with_background(canvas: HtmlCanvasElement, background_color: &str) -> Result<CubeRenderer, JsValue> {
        Self::new_with_backend_and_background(canvas, true, background_color).await
    }

    async fn new_with_backend_and_background(canvas: HtmlCanvasElement, force_webgl: bool, background_color: &str) -> Result<CubeRenderer, JsValue> {
        utils::set_panic_hook();
        
        // Parse background color from hex string
        let bg_color = Self::parse_hex_color(background_color)?;
        
        console::log_1(&format!("Canvas width: {}, height: {}", canvas.width(), canvas.height()).into());

        let width = canvas.width();
        let height = canvas.height();

        if force_webgl {
            console::log_1(&"🔧 TESTING: Forcing WebGL backend".into());
            return Self::create_webgl_renderer(canvas, width, height, bg_color).await;
        }

        // Try WebGPU first, fall back to WebGL if it fails
        console::log_1(&"🚀 Attempting to use WebGPU backend...".into());
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                    fence_behavior: wgpu::GlFenceBehavior::default(),
                },
                ..Default::default()
            },
        });

        // Clone canvas for potential fallback use
        let canvas_clone = canvas.clone();
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

        let adapter_result = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter_result {
            Ok(adapter) => {
                let backend = adapter.get_info().backend;
                console::log_1(&format!("✅ Adapter acquired successfully using: {:?}", backend).into());
                adapter
            }
            Err(e) => {
                console::log_1(&format!("❌ WebGPU adapter request failed: {:?}", e).into());
                console::log_1(&"🔄 Falling back to WebGL...".into());
                
                // Use the reusable WebGL function for fallback
                return Self::create_webgl_renderer(canvas_clone, width, height, bg_color).await;
            }
        };

        Self::create_with_adapter_and_surface(adapter, surface, width, height, bg_color).await
    }

    async fn create_webgl_renderer(
        canvas: HtmlCanvasElement,
        width: u32,
        height: u32,
        bg_color: wgpu::Color,
    ) -> Result<CubeRenderer, JsValue> {
        let webgl_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions {
                gl: wgpu::GlBackendOptions {
                    gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                    fence_behavior: wgpu::GlFenceBehavior::default(),
                },
                ..Default::default()
            },
        });
        
        let webgl_surface = webgl_instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("Failed to create WebGL surface: {:?}", e)))?;
        
        let webgl_adapter = webgl_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&webgl_surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to request WebGL adapter: {:?}", e)))?;
        
        console::log_1(&format!("✅ WebGL adapter acquired successfully using: {:?}", webgl_adapter.get_info().backend).into());
        
        Self::create_with_adapter_and_surface(webgl_adapter, webgl_surface, width, height, bg_color).await
    }

    async fn create_with_adapter_and_surface(
        adapter: wgpu::Adapter,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        bg_color: wgpu::Color,
    ) -> Result<CubeRenderer, JsValue> {
        // Log adapter information
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        console::log_1(&format!("📊 Adapter backend: {:?}", adapter_info.backend).into());
        console::log_1(&format!("📊 Adapter limits: max_inter_stage_shader_components = {}", adapter_limits.max_inter_stage_shader_components).into());

        // Choose appropriate limits based on backend
        let device_limits = match adapter_info.backend {
            wgpu::Backend::BrowserWebGpu => {
                console::log_1(&"🚀 Using WebGPU limits".into());
                wgpu::Limits::default()
            }
            wgpu::Backend::Gl => {
                console::log_1(&"🔧 Using WebGL2 downlevel limits".into());
                wgpu::Limits::downlevel_webgl2_defaults()
            }
            _ => {
                console::log_1(&"⚠️ Unknown backend, using default limits".into());
                wgpu::Limits::default()
            }
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: device_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

        console::log_1(&"✅ Device created successfully".into());

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create uniform buffer
        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        console::log_1(&"🎉 CubeRenderer created successfully!".into());

        Ok(Self {
            surface,
            device,
            queue,
            config,
            width,
            height,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            camera_distance: 5.0,
            camera_pan_x: 0.0,
            camera_pan_y: 0.0,
            camera_rotation_x: 0.0,
            camera_rotation_y: 0.0,
            background_color: bg_color,
            performance_tracker: PerformanceTracker::new(),
        })
    }

    #[wasm_bindgen]
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            console::log_1(&format!("Resized canvas to width: {}, height: {}", width, height).into());
        }
    }

    #[wasm_bindgen]
    pub fn zoom(&mut self, delta: f32) {
        // Zoom by adjusting camera distance
        // Positive delta = zoom in (get closer), negative = zoom out (get farther)
        let zoom_sensitivity = 0.1;
        let zoom_factor = 1.0 + (delta * zoom_sensitivity);
        
        self.camera_distance /= zoom_factor;
        
        // Clamp camera distance to reasonable bounds
        self.camera_distance = self.camera_distance.clamp(1.0, 50.0);
    }

    #[wasm_bindgen]
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        // Pan by adjusting the target position
        // Scale pan sensitivity based on camera distance (further = larger pan movements)
        let pan_sensitivity = 0.001 * self.camera_distance;
        
        // Invert deltaX and deltaY so dragging feels like moving the object directly
        self.camera_pan_x -= delta_x * pan_sensitivity; // Invert X for natural movement
        self.camera_pan_y += delta_y * pan_sensitivity; // Invert Y for natural movement
    }

    #[wasm_bindgen]
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        // Rotate camera around target (orbital rotation)
        let rotation_sensitivity = 0.001;
        
        // Invert deltas so dragging feels like rotating the object directly
        self.camera_rotation_y -= delta_x * rotation_sensitivity; // Invert Yaw for natural rotation
        self.camera_rotation_x -= delta_y * rotation_sensitivity; // Invert Pitch for natural rotation
        
        // Clamp pitch to prevent gimbal lock
        self.camera_rotation_x = self.camera_rotation_x.clamp(-1.5, 1.5);
    }

    #[wasm_bindgen]
    pub fn render(&mut self) -> Result<(), JsValue> {
        // Start performance tracking for this frame
        self.performance_tracker.start_frame();
        
        // Create transformation matrices
        let aspect = self.width as f32 / self.height as f32;
        let proj = cgmath::perspective(cgmath::Deg(45.0), aspect, 0.1, 100.0);
        
        // Create orbital camera system
        // 1. Apply rotations around the target
        let rotation_y = cgmath::Matrix4::from_angle_y(cgmath::Rad(self.camera_rotation_y));
        let rotation_x = cgmath::Matrix4::from_angle_x(cgmath::Rad(self.camera_rotation_x));
        let rotation_matrix = rotation_y * rotation_x;
        
        // 2. Position camera at distance from target
        let camera_offset = cgmath::Vector3::new(0.0, 0.0, self.camera_distance);
        let rotated_offset = rotation_matrix.transform_vector(camera_offset);
        
        // 3. Apply pan offset to target position
        let target = cgmath::Point3::new(self.camera_pan_x, self.camera_pan_y, 0.0);
        let camera_position = target + rotated_offset;
        
        // 4. Create view matrix
        let view = cgmath::Matrix4::look_at_rh(
            camera_position,
            target,
            cgmath::Vector3::unit_y(),
        );

        // Static model matrix (object stays at origin)
        let model = cgmath::Matrix4::identity();

        let view_proj = proj * view * model;
        self.uniforms.update_view_proj(view_proj);

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        let output = self.surface
            .get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Failed to get surface texture: {:?}", e)))?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // End performance tracking and return snapshot if available
        // Note: We don't return the snapshot from render() to avoid affecting performance
        // JavaScript will poll for snapshots separately
        self.performance_tracker.end_frame();

        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_performance_snapshot(&mut self) -> Option<PerformanceSnapshot> {
        // Force create a snapshot for JavaScript to consume
        // This is called periodically from JavaScript, not every frame
        if !self.performance_tracker.frame_times.is_empty() {
            let now = now();
            Some(self.performance_tracker.create_snapshot(now))
        } else {
            None
        }
    }
} 